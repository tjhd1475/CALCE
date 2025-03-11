"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

import os
import sys
import re

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from peft import LoraConfig, get_peft_model
import wandb

sys.path.append(sys.path[0] + "/..")
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.common.dist_utils import is_main_process
from lavis.models.blip2_mr_models.utils import (
    format_wandb_log_images_and_predictions,
    post_process,
    post_process_TAL,
    get_timestamps_as_seconds_integers,
    get_timestamps_as_relative_integers,
    get_timestamps_as_seconds_floats,
    get_timestamps_as_relative_floats,
    get_timestamps_as_framenumbers,
)

# set the environment variable TOKENIZERS_PARALLELISM = false
# to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@registry.register_model("blip2_mr")
class BLIP2_MR(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        num_beams=5,
        prompt="",
        max_txt_len=200,
        apply_lemmatizer=False,
        input_time_format="seconds_integers",
        interleave_data=False,
        frame_token_aggregation=None,
        task="lora",
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.task = task
        if "TAL" in task:
            self.post_process = post_process_TAL
        else:
            self.post_process = post_process
        self.use_lora = True if "lora" in task else False
        self.use_wandb = True if wandb.run is not None else False
        self.log_samples_every_n = 3000
        self.log_samples_every_n_eval = 1000

        self.input_time_format = input_time_format
        self.interleave_data = interleave_data
        self.frame_token_aggregation = frame_token_aggregation

        if self.use_wandb and is_main_process():
            self.wandb_table_data = []
            self.wandb_table_data_eval = []

        ### Vision backbone ######################################################
        (
            self.visual_encoder,
            self.ln_vision,
        ) = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        # freeze ViT
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        ##########################################################################

        ### Text backbone ########################################################
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        # Depending on the tokenizer, some numbers are represented as 2 tokens
        # this is annoying and needs to be fixed
        # fairly dirty fix, is to just replace them with the closest number that is not "annoying"
        self.annoying_numbers, _ = self.find_annoying_numbers(self.t5_tokenizer, 200)
        self.annoying_numbers_replacement_dict = (
            self.find_annoying_numbers_replacement_dict(self.annoying_numbers)
        )

        logging.info(
            "Annoying numbers and their replacement: {}".format(
                self.annoying_numbers_replacement_dict
            )
        )

        ### LORA ##########

        if self.use_lora:
            # If only targeting attention blocks of the model
            # target_modules = ["q", "v"]

            # If targeting all linear layers
            model_modules = str(self.t5_model.modules)
            pattern = r"\((\w+)\): Linear"
            linear_layer_names = re.findall(pattern, model_modules)

            names = []
            # Print the names of the Linear layers
            for name in linear_layer_names:
                names.append(name)
            target_modules = list(set(names))

            lora_config = LoraConfig(
                r=8,
                target_modules=target_modules,
                lora_alpha=8,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.t5_model = get_peft_model(self.t5_model, lora_config)
            self.t5_model.print_trainable_parameters()
        else:
            # freeze T5
            for name, param in self.t5_model.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16()

        ##########################################################################

        ### Q-Former for Image Embeddings ########################################
        self.multimodal_Qformer = False

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        if not self.multimodal_Qformer:
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer_tokenizer = self.init_tokenizer()

        self.num_query_token = num_query_token
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        ##########################################################################

        self.max_txt_len = max_txt_len

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_beams = num_beams

        # Seperator token ">" when placed in the middle of a sentence
        self.seperator_token = self.t5_tokenizer.convert_tokens_to_ids(">")
        self.pad_token_id = self.t5_tokenizer.pad_token_id

        if "qformer_freeze" in self.task:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.query_tokens.requires_grad = False
            self.t5_proj.requires_grad = False

        if is_main_process() and self.use_wandb:
            if self.use_lora:
                wandb.watch(self.t5_model, log="all")
            if not "qformer_freeze" in self.task:
                wandb.watch(self.Qformer, log="all")
                wandb.watch(self.t5_proj, log="all")

    def forward(
        self,
        samples,
    ):
        image = samples["video"]
        timestamps, durations = (
            samples["timestamps"],
            samples["duration"],
        )
        video_prompt_end = samples["video_prompt_end"]
        query_prompt, task_prompt = samples["query_prompt"], samples["task_prompt"]
        answer = samples["relevant_windows"]

        # uniform sampling
        b, t, c, w, h = image.shape
        image = image.reshape(-1, c, w, h)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))  # bt, n, c
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )  # bt n c

        ### Apply Q-Former for Image Embeddings ####################################
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if self.multimodal_Qformer:
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.Qformer_tokenizer(
                [
                    q for q in query_prompt for _ in range(t)
                ],  # apply query to each frame
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            frames_for_projection = output.last_hidden_state[
                :, : query_tokens.size(1), :
            ]
        else:
            frames_after_qformer = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            frames_for_projection = frames_after_qformer.last_hidden_state

        frames_for_t5 = self.t5_proj(frames_for_projection)

        # TODO: Use average pooling to aggregate the 32 embeddings of one frame
        if self.frame_token_aggregation:
            assert self.frame_token_aggregation in [
                "mean",
                False,
            ], "Invalid aggregation method, please choose from ['mean']"
            frames_for_t5 = frames_for_t5.mean(dim=1, keepdim=True)

        # reshape the frames for t5 from (bt, n, c) to (b, t * n, c)
        frames_for_t5 = frames_for_t5.reshape(
            b, t, frames_for_t5.shape[-2], -1
        )  # b, t, n, c
        frames_atts_for_t5 = torch.ones(frames_for_t5.size()[:-1], dtype=torch.long).to(
            image.device
        )  # b, t, n
        frames_for_t5 = frames_for_t5.reshape(
            b, -1, frames_for_t5.shape[-1]
        )  # b, t * n, c
        frames_atts_for_t5 = frames_atts_for_t5.reshape(b, -1)  # b, t * n

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embs_mr, inputs_atts_mr, video_prompt = self.prompt_concatenation(
                timestamps,
                durations,
                frames_for_t5,
                frames_atts_for_t5,
                video_prompt_end,
                query_prompt,
                task_prompt,
            )

            ### Encode answer ################################################
            output_tokens_mr = self.t5_tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            targets_mr = output_tokens_mr.input_ids.masked_fill(
                output_tokens_mr.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            output_tokens_mask_mr = output_tokens_mr.attention_mask

            ### Apply moment retrieval prompt ######################################
            outputs_loc = self.t5_model(
                inputs_embeds=inputs_embs_mr,
                attention_mask=inputs_atts_mr,
                decoder_attention_mask=output_tokens_mask_mr,
                return_dict=True,
                labels=targets_mr,
            )
            loss = outputs_loc.loss

            # write the following to a wandb table
            if self.use_wandb and is_main_process():
                log = {}
                log["train/log_likelihood_loss"] = loss.item()
                # Log images and predictions
                if samples["iters"] % self.log_samples_every_n == 0:
                    pred = self.t5_tokenizer.batch_decode(
                        torch.argmax(outputs_loc.logits, dim=-1)
                    )
                    out, self.wandb_table_data = (
                        format_wandb_log_images_and_predictions(
                            samples=samples,
                            wandb_table_data=self.wandb_table_data,
                            pred=pred,
                            video_prompt=video_prompt,
                            post_process_fn=self.post_process,
                            input_time_format=self.input_time_format,
                            interleave_data=True,
                            train_data=True,
                        )
                    )
                    log.update(out)
                # Log iteration
                wandb.log(log)

        return {"loss": loss}

    def prompt_concatenation(
        self,
        timestamps,
        durations,
        frames_for_t5,
        frames_atts_for_t5,
        video_prompt_end,
        query_prompt,
        task_prompt,
    ):

        ### video prompt
        # </vid> = <extra_id_0>\n
        if "only_frames" in self.task:
            assert (
                not self.input_time_format
            ), "Set input_time_format to False in the config to use only frames without timestamps."
            # prompt will at the end look as follows:
            # <vid> f1 > f2 > ... > fT > </vid>\n
            video_prompt = ["<vid>" for _ in range(len(timestamps))]
            video_prompt_end = ["<extra_id_0>\n" for _ in range(len(video_prompt_end))]
        elif "add_duration" in self.task:
            # prompt will at the end look as follows:
            # <vid> f1 > f2 > ... > fT > duration </vid>\n
            video_prompt_end = [
                "{}<extra_id_0>\n".format(">" + d.item()) for d in durations
            ]
            video_prompt = ["<vid>" for _ in range(len(timestamps))]

        if self.input_time_format == "framenumbers":
            timestamps, durations, video_prompt = get_timestamps_as_framenumbers(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        elif self.input_time_format == "relative_floats":
            timestamps, durations, video_prompt = get_timestamps_as_relative_floats(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        elif self.input_time_format == "relative_integers":
            timestamps, durations, video_prompt = get_timestamps_as_relative_integers(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        elif self.input_time_format == "seconds_integers":
            timestamps, durations, video_prompt = get_timestamps_as_seconds_integers(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        elif self.input_time_format == "seconds_floats":
            timestamps, durations, video_prompt = get_timestamps_as_seconds_floats(
                timestamps, durations, self.annoying_numbers_replacement_dict
            )

        # </vid> = <extra_id_0>\n
        video_prompt_end_tokens = self.t5_tokenizer(
            video_prompt_end,
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_t5.device)
        video_prompt_end_embs = self.t5_model.encoder.embed_tokens(
            video_prompt_end_tokens.input_ids
        )

        ### query_prompt + task_prompt
        # Question: q
        # Given the video and the query, find the relevant windows.
        # Relevant windows: [start_time, end_time]

        # concatenate query_prompt and task_prompt (list[str])
        if "no_task_prompt" in self.task:
            text_prompt = [q for q in query_prompt]
        else:
            text_prompt = [q + t for q, t in zip(query_prompt, task_prompt)]

        text_prompt_tokens = self.t5_tokenizer(
            text_prompt,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(frames_for_t5.device)
        text_prompt_embs = self.t5_model.encoder.embed_tokens(
            text_prompt_tokens.input_ids
        )

        if self.interleave_data:
            assert (
                "integer" in self.input_time_format
            ), "Interleaving only works with integer time formats where one number is one token."

            # get the tokens and embeddings for the timestamps and durations
            batch_timestamps_tokens = []
            batch_timestamps_embs = []
            for t in timestamps:
                tokens, embs = self.get_clean_timestamp_tokens_and_embs(t)
                batch_timestamps_tokens.append(tokens)
                batch_timestamps_embs.append(embs)

            batch_duration_tokens, batch_duration_embs = (
                self.get_clean_timestamp_tokens_and_embs(torch.tensor(durations))
            )

            interleaved_video_prompt_embs = []
            video_prompt = []

            b, t_n, c = frames_for_t5.shape
            if self.frame_token_aggregation:
                n = 1
            else:
                n = self.num_query_token
            t = t_n // n

            # iterate over the batch
            for j, (timestamp_embs, frame_embs) in enumerate(
                zip(batch_timestamps_embs, frames_for_t5)
            ):
                interleaved_prompt = torch.tensor([]).to(frames_for_t5.device)
                _video_prompt = []
                # iternate over the number of frames -> t
                for i in range(t):
                    # each frame has n tokens -> shape (t*n, c)
                    frame_emb = frame_embs[i * n : i * n + n]

                    timestamp_emb = timestamp_embs[i]

                    # for logging of input design
                    _video_prompt.append(
                        "f{i}-".format(i=i)
                        + self.t5_tokenizer.decode(batch_timestamps_tokens[j][i])
                        + ">"
                    )

                    # frame i and corresponding timestamp
                    frame_and_time = torch.cat(
                        [
                            frame_emb,
                            timestamp_emb,
                        ]
                    )
                    # add frame and timestamp "pair" to the interleaved prompt
                    interleaved_prompt = torch.cat([interleaved_prompt, frame_and_time])

                # add one more seperator and the duration tokens to the interleaved prompt
                seperator_emb = self.t5_model.encoder.embed_tokens(
                    torch.tensor(self.seperator_token)
                    .to(frames_for_t5.device)
                    .unsqueeze(0)
                )

                duration_emb = batch_duration_embs[j]
                interleaved_prompt = torch.cat(
                    [interleaved_prompt, seperator_emb, duration_emb]
                )

                # batch level list of interleaved video prompt
                interleaved_video_prompt_embs.append(interleaved_prompt)

                # for logging of input design
                # append the decoded video duration
                video_prompt.append(
                    "".join(_video_prompt)
                    + self.t5_tokenizer.decode(batch_duration_tokens[j])
                )

            # if interleaved_video_prompt_embs elements are not same length, pad them
            # should only be necessary if we allow for timestamp numbers to be tokenize to more than 1 token
            max_len = max([len(i) for i in interleaved_video_prompt_embs])
            for i in range(len(interleaved_video_prompt_embs)):
                if len(interleaved_video_prompt_embs[i]) < max_len:
                    padding = self.pad_token_id * torch.ones(
                        max_len - len(interleaved_video_prompt_embs[i]),
                        interleaved_video_prompt_embs[i].shape[-1],
                    ).to(frames_for_t5.device)
                    interleaved_video_prompt_embs[i] = torch.cat(
                        [padding, interleaved_video_prompt_embs[i]]
                    )

            interleaved_video_prompt_embs = torch.stack(
                interleaved_video_prompt_embs
            ).to(frames_for_t5.device)

            ### Concatenate interleaved_video_prompt, video_prompt_end, text_prompt
            inputs_embs_mr = torch.cat(
                [
                    interleaved_video_prompt_embs,
                    video_prompt_end_embs,
                    text_prompt_embs,
                ],
                dim=1,
            )

            interleaved_video_prompt_attn_embs = torch.ones(
                interleaved_video_prompt_embs.size()[:-1], dtype=torch.long
            ).to(frames_for_t5.device)
            interleaved_video_prompt_attn_embs = (
                interleaved_video_prompt_attn_embs.reshape(b, -1)
            )

            inputs_atts_mr = torch.cat(
                [
                    interleaved_video_prompt_attn_embs,
                    video_prompt_end_tokens.attention_mask,
                    text_prompt_tokens.attention_mask,
                ],
                dim=1,
            )
        else:

            video_prompt_tokens = self.t5_tokenizer(
                video_prompt,
                padding="longest",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(frames_for_t5.device)
            video_prompt_embs = self.t5_model.encoder.embed_tokens(
                video_prompt_tokens.input_ids
            )

            ### Concatenate video_prompt, frames_for_t5, video_prompt_end, text_prompt
            inputs_embs_mr = torch.cat(
                [
                    video_prompt_embs,
                    frames_for_t5,
                    video_prompt_end_embs,
                    text_prompt_embs,
                ],
                dim=1,
            )

            inputs_atts_mr = torch.cat(
                [
                    video_prompt_tokens.attention_mask,
                    frames_atts_for_t5,
                    video_prompt_end_tokens.attention_mask,
                    text_prompt_tokens.attention_mask,
                ],
                dim=1,
            )

            video_prompt = [
                p + "frames" + end_p
                for (p, end_p) in zip(video_prompt, video_prompt_end)
            ]

        return inputs_embs_mr, inputs_atts_mr, video_prompt

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=50,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        out = {}
        image = samples["video"]
        timestamps, durations = (
            samples["timestamps"],
            samples["duration"],
        )
        qid = samples["query_id"]
        video_prompt_end = samples["video_prompt_end"]
        query_prompt, task_prompt = samples["query_prompt"], samples["task_prompt"]
        answer = samples["relevant_windows"]

        # uniform sampling
        b, t, c, w, h = image.shape
        image = image.reshape(-1, c, w, h)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))  # bt, n, c
        _, n, _ = image_embeds.shape
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )  # bt n c

        ### Apply Q-Former for Image Embeddings ####################################
        query_tokens_qa = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        frames_after_qformer = self.Qformer.bert(
            query_embeds=query_tokens_qa,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        frames_for_t5 = self.t5_proj(frames_after_qformer.last_hidden_state)

        if self.frame_token_aggregation:
            assert self.frame_token_aggregation in [
                "mean"
            ], "Invalid aggregation method, please choose from ['mean']"
            frames_for_t5 = frames_for_t5.mean(dim=1, keepdim=True)

        # reshape the frames for t5 from (bt, n, c) to (b, t * n, c)
        frames_for_t5 = frames_for_t5.reshape(
            b, t, frames_for_t5.shape[-2], -1
        )  # b, t, n, c
        frames_atts_for_t5 = torch.ones(frames_for_t5.size()[:-1], dtype=torch.long).to(
            image.device
        )  # b, t, n
        frames_for_t5 = frames_for_t5.reshape(
            b, -1, frames_for_t5.shape[-1]
        )  # b, t * n, c
        frames_atts_for_t5 = frames_atts_for_t5.reshape(b, -1)  # b, t * n

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embs_mr, inputs_atts_mr, video_prompt = self.prompt_concatenation(
                timestamps,
                durations,
                frames_for_t5,
                frames_atts_for_t5,
                video_prompt_end,
                query_prompt,
                task_prompt,
            )

            ### Apply moment retrieval prompt ######################################
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embs_mr,
                attention_mask=inputs_atts_mr,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                output_attentions=output_attentions,
            )

            # tokenizer decode outputs
            pred_ans = self.t5_tokenizer.batch_decode(
                outputs[0], skip_special_tokens=True
            )

        # if duration is Tensor, convert to list
        if isinstance(samples["duration"], torch.Tensor):
            out["duration"] = samples["duration"].tolist()
        else:
            out["duration"] = samples["duration"]

        if (
            self.input_time_format == "relative_integers"
            or self.input_time_format == "relative_floats"
        ):
            prediction = [self.post_process(pred) for pred in pred_ans]
            out["prediction"] = self.convert_to_absolute_time(
                prediction, out["duration"]
            )
        else:
            out["prediction"] = [self.post_process(pred) for pred in pred_ans]

        out["raw_prediction"] = pred_ans
        out["answer"] = answer
        out["qid"] = qid

        # write the following to a wandb table
        if self.use_wandb and is_main_process():
            # Log images and predictions
            if samples["iters"] % self.log_samples_every_n_eval == 0:
                table, self.wandb_table_data_eval = (
                    format_wandb_log_images_and_predictions(
                        samples=samples,
                        wandb_table_data=self.wandb_table_data_eval,
                        pred=pred_ans,
                        video_prompt=video_prompt,
                        post_process_fn=self.post_process,
                        input_time_format=self.input_time_format,
                        interleave_data=True,
                        train_data=False,
                    )
                )
                # Log iteration
                wandb.log(table)

        return out

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs,
    ):
        image = samples["image"]
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        num_beams = cfg.get("num_beams", 5)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        input_time_format = cfg.get("input_time_format", "seconds_integers")
        interleave_data = cfg.get("interleave_data", True)
        frame_token_aggregation = cfg.get("frame_token_aggregation", None)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_len", 200)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        task = cfg.get("task", "qformer_freeze_lora")

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            num_beams=num_beams,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            input_time_format=input_time_format,
            interleave_data=interleave_data,
            frame_token_aggregation=frame_token_aggregation,
            task=task,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
            logging.info("load pretrained weights from %s" % pretrain_path)

            # get finetuned lora adapter
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
            logging.info("load finetuned weights from %s" % finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
            logging.info("load pretrained weights from %s" % pretrain_path)

    def find_annoying_numbers(
        self,
        tokenizer=T5TokenizerFast.from_pretrained("google/flan-t5-xl", local_files_only=True),
        range_end=300,
    ):
        """
        Find numbers that are tokenized in more than one token by the T5 tokenizer.

        Args:
            tokenizer: A tokenizer object from the transformers library.
            range_end: The range of numbers to check.

        Returns:
            annoying_numbers: A list of numbers that are tokenized in more than one token.
            annoying_numbers_spance: A list of numbers that are tokenized in more than one token, but the first token is a space.
        """

        annoying_numbers = []
        annoying_numbers_spance = []
        for i in range(0, range_end):
            tokens = tokenizer(
                str(i),
                padding="longest",
                add_special_tokens=False,
                truncation=True,
                max_length=300,
                return_tensors="pt",
            )

            n_tokens = len(tokens["input_ids"].tolist()[0])

            if n_tokens > 1:
                if tokens["input_ids"].tolist()[0][0] == 3:
                    annoying_numbers_spance.append(i)
                else:
                    annoying_numbers.append(i)

        return annoying_numbers, annoying_numbers_spance

    def find_annoying_numbers_replacement_dict(self, annoying_numbers):
        """
        Find a the closes integer replacement for numbers that are tokenized in more than one token by the T5 tokenizer.

        Args:
            annoying_numbers: A list of numbers that are tokenized in more than one token.

        Returns:
            annoying_numbers_replacement_dict: A dictionary with the number as key and the replacement as value.
        """

        annoying_numbers_replacement_dict = {}
        for i in annoying_numbers:
            for j in range(100):
                if (i + j) not in annoying_numbers:
                    new_i = i + j
                    break
                if (i - j) not in annoying_numbers:
                    new_i = i - j
                    break

            annoying_numbers_replacement_dict[i] = new_i

        return annoying_numbers_replacement_dict

    def get_clean_timestamp_tokens_and_embs(self, timestamps):
        """
        Tokenize timestamps and clean up removing the special tokens.
        Return a list of tokenized and embedded timestamps.
        Each timestamp can be one or more tokens.

        Args:
            timestamps: A list of timestamps.

        Returns:
            tokens: A list of cleaned tokenized timestamps.
            token_embs list(torch.tensor()): A list of token embeddings.
        """

        # This es extremely slow, but no idea how to best do it otherwise...
        tokens = self.t5_tokenizer(
            [str(t.item()) for t in timestamps], add_special_tokens=False
        )["input_ids"]

        # remove the leading 3 in each tokenized timestamp if there is one
        tokens = [token[1:] if token[0] == 3 else token for token in tokens]

        # concatenate all tokens into one list
        # and create a mask to know when tokens are from different timestamps
        concatenated_tokens = []
        concatenated_tokens_mask = []
        for i, token in enumerate(tokens):
            concatenated_tokens.extend(token)
            concatenated_tokens_mask.extend([i] * len(token))

        token_embs = []

        token_embs_tmp = self.t5_model.encoder.embed_tokens(
            torch.tensor(concatenated_tokens).to(self.device)
        )

        # group the token embeddings by timestamp
        token_embs = [[] for _ in range(len(timestamps))]
        for i, mask in enumerate(concatenated_tokens_mask):
            if token_embs[mask] == []:
                token_embs[mask] = token_embs_tmp[i].unsqueeze(0)
            else:
                token_embs[mask] = torch.cat(
                    [token_embs[mask], token_embs_tmp[i].unsqueeze(0)],
                    dim=0,
                )

        return tokens, token_embs


if __name__ == '__main__':
    model = BLIP2_MR(
        interleave_data=True
    )