"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

import os
import sys
import re,copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from peft import LoraConfig, get_peft_model
import numpy as np
import wandb

sys.path.append(sys.path[0] + "/..")
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.common.dist_utils import is_main_process
from lavis.models.blip2_calce_models.utils import (
    format_wandb_log_images_and_predictions,
    post_process,
    post_process_TAL,
    get_timestamps_as_seconds_integers,
    moment_str_to_list
)

from lavis.models.blip2_calce_models.cluster import CTM

# set the environment variable TOKENIZERS_PARALLELISM = false
# to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@registry.register_model("blip2_calce_stage1")
class BLIP2_CALCE_STAGE1(Blip2Base):
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
        format_target_amount=30,
        interleave_data=False,
        frame_token_aggregation=None,
        task="lora",
        key_frame_selection="event",
        num_clusters=12,
        bias1=1,
        debug=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.task = task
        self.post_process = post_process
        self.use_lora = True if "lora" in task else False
        self.use_wandb = True if wandb.run is not None else False
        self.log_samples_every_n = 4000
        self.log_samples_every_n_eval = 1500

        self.input_time_format = input_time_format
        self.format_target_amount = format_target_amount
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
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model,local_files_only=True)
        t5_config = T5Config.from_pretrained(t5_model,local_files_only=True)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config,local_files_only=True
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

        ##########################################################################
        self.num_query_token = num_query_token
        self.query_combine_rate = 4

        ### Q-Former for Image Embeddings ########################################
        self.tokenizer = self.init_tokenizer()
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features,debug
        )
        self.query_proj = nn.Linear(self.num_query_token, int(self.num_query_token / self.query_combine_rate))
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None

        if "qformer_freeze" in self.task:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.query_tokens.requires_grad = False
            self.t5_proj.requires_grad = False

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

        self.max_txt_len = max_txt_len

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_beams = num_beams

        # Seperator token ">" when placed in the middle of a sentence
        self.pad_token_id = self.t5_tokenizer.pad_token_id
        self.video_sign_part = "<extra_id_0>"
        self.video_sign_frame = "," # unused
        self.video_sign_time = ","  # unused
        self.video_sign_duration = ">"
        self.subtitle_sign_part = "<extra_id_1>"
        self.subtitle_sign_time_start = "["
        self.subtitle_sign_text = "|"   # unused
        self.subtitle_sign_time_end = "]"

        self.key_frame_selection = key_frame_selection
        # cluster
        self.num_cluster_center = num_clusters
        self.cluster = CTM(sample_ratio=self.num_cluster_center)
        self.bias1 = bias1

        if is_main_process() and self.use_wandb:
            if self.use_lora:
                wandb.watch(self.t5_model, log="all")
            if not "qformer_freeze" in self.task:
                wandb.watch(self.Qformer, log="all")
                wandb.watch(self.t5_proj, log="all")

    keyword_set = ['talk','speak','say','describe','explain','chat','shout','scream']
    def is_talk_relative(self,query):
        return any(keyword in query for keyword in self.keyword_set)

    def select_key_frame(self,embeds,batch_subtitle_durations,durations,query):
        b, t = embeds.shape[:2] # b, t, n, c
        if self.key_frame_selection == 'cluster':
            ####### cluster #########
            embeds = torch.mean(embeds, dim=2, keepdim=False).reshape(b, t, -1).clone()   # b, t, c
            cluster_dict = {'x': embeds,
                            'token_num': embeds.size(1),
                            'idx_token': torch.arange(t)[None, :].repeat(b, 1),
                            'agg_weight': embeds.new_ones(b, t, 1),
                            'mask': None}
            cluster_dict = self.cluster(cluster_dict)
            batch_cluster_idx = cluster_dict['idx_token']
        elif self.key_frame_selection == 'information':
            embeds = torch.mean(embeds, dim=2, keepdim=False).reshape(b, t, -1).clone()  # b, t, c
            embeds_shift = embeds[:, 1:, :]
            embeds = embeds[:, :-1, :]
            batch_sub_sim = F.cosine_similarity(embeds, embeds_shift, dim=-1, eps=1e-8)  # b, t-1
            batch_sub_sim = F.softmax(batch_sub_sim, dim=-1)


        ###### get key frame ######
        batch_key_frame_idx, batch_non_key_frame_idx = [], []
        for bid, (subtitle_durations, duration, q) in enumerate(zip(batch_subtitle_durations, durations,
                                                                query)):
            if self.key_frame_selection == 'cluster':
                cluster_idx = batch_cluster_idx[bid]
                # native key frame (cluster edges)
                key_frame_mask = torch.zeros(t, dtype=torch.int)
                for i in range(1, t):
                    if cluster_idx[i - 1] != cluster_idx[i]:  # frame between different cluster is key frame
                        key_frame_mask[i - 1] = 1
                        key_frame_mask[i] = 1
                key_frame_mask[0] = 1
                key_frame_mask[-1] = 1
                key_frame_mask = self.two_side_bias_padding(key_frame_mask, self.bias1)
                # subtitle exist frame is key frame for talk relative
            elif self.key_frame_selection == 'information':
                key_frame_mask = torch.zeros(t, dtype=torch.int)
                sub_sim = batch_sub_sim[bid]
                sim_rank = torch.argsort(sub_sim, dim=-1, descending=False)
                top_sim_rank = sim_rank[:t//3]
                key_frame_mask[top_sim_rank] = 1
                key_frame_mask[top_sim_rank+1] = 1
                key_frame_mask = self.two_side_bias_padding(key_frame_mask, self.bias1)
            elif self.key_frame_selection == 'interval':
                key_frame_mask = torch.remainder(torch.arange(t,dtype=torch.int), 2)
            else:
                key_frame_mask = torch.zeros(t, dtype=torch.int)

            if self.is_talk_relative(q) and subtitle_durations:
                key_frame_mask = torch.zeros(t, dtype=torch.int)
                for subtitle_duration in subtitle_durations:
                    start = int(subtitle_duration[0] * t / duration)
                    end = min(int(subtitle_duration[1] * t / duration), t - 1)
                    for i in range(start, end + 1):
                        key_frame_mask[i] = 1

            non_key_frame_mask = 1 - key_frame_mask
            key_frame_idx = torch.nonzero(key_frame_mask).reshape(-1).tolist()
            non_key_frame_idx = torch.nonzero(non_key_frame_mask).reshape(-1).tolist()
            batch_key_frame_idx.append(key_frame_idx)
            batch_non_key_frame_idx.append(non_key_frame_idx)
        return batch_key_frame_idx, batch_non_key_frame_idx

    def forward(
        self,
        samples,
    ):
        qid = samples["query_id"]
        video = samples["video"]
        timestamps, durations = samples["timestamps"].tolist(),samples["duration"].tolist()
        batch_subtitles = samples["subtitles"]
        batch_subtitle_durations = samples["subtitle_durations"]
        query, task_prompt = samples["query_prompt"], samples["task_prompt"]
        item_prompt = samples['item_prompt'] if 'item' in self.task else None

        relevant_windows = samples["relevant_windows"]
        num_iters = samples["iters"]

        device = video.device

        # uniform sampling
        b, t, c, w, h = video.shape
        video = video.reshape(-1, c, w, h)  # bt, c, w, h
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            video_embeds = self.ln_vision(self.visual_encoder(video))  # bt, n, c
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(device)  # bt n c

        if self.input_time_format == 'frame_indices':
            durations_orig = copy.deepcopy(durations)
            relevant_windows_orig = copy.deepcopy(relevant_windows)
            timestamps, relevant_windows, durations = self.absolute_times_to_frame_indices(timestamps, relevant_windows, durations, amount=self.format_target_amount)

        # ####### cluster #########
        # cluster_features = torch.mean(video_embeds, dim=1, keepdim=False).reshape(b, t, -1).clone()
        # cluster_dict = {'x': cluster_features,
        #               'token_num': cluster_features.size(1),
        #               'idx_token': torch.arange(t)[None, :].repeat(b, 1),
        #               'agg_weight': cluster_features.new_ones(b, t, 1),
        #               'mask': None}
        # cluster_dict = self.cluster(cluster_dict)
        # batch_cluster_idx = cluster_dict['idx_token']
        #
        # ###### get key frame ######
        # batch_key_frame_idx, batch_non_key_frame_idx = [], []
        # for cluster_idx, subtitle_durations,duration,q in zip(batch_cluster_idx,batch_subtitle_durations,durations,query):
        #     # native key frame (cluster edges)
        #     key_frame_mask = torch.zeros(t, dtype=torch.int)
        #     for i in range(1, t):
        #         if cluster_idx[i - 1] != cluster_idx[i]:  # frame between different cluster is key frame
        #             key_frame_mask[i - 1] = 1
        #             key_frame_mask[i] = 1
        #     key_frame_mask[0] = 1
        #     key_frame_mask[-1] = 1
        #     key_frame_mask = self.two_side_bias_padding(key_frame_mask,self.bias)
        #     # subtitle exist frame is key frame for talk relative
        #     if self.is_talk_relative(q) and subtitle_durations:
        #         key_frame_mask = torch.zeros(t, dtype=torch.int)
        #         for subtitle_duration in subtitle_durations:
        #             start = int(subtitle_duration[0] * t / duration)
        #             end = min(int(subtitle_duration[1] * t / duration),t-1)
        #             for i in range(start,end+1):
        #                 key_frame_mask[i] = 1
        #
        #     non_key_frame_mask = 1 - key_frame_mask
        #     key_frame_idx = torch.nonzero(key_frame_mask).reshape(-1).tolist()
        #     non_key_frame_idx = torch.nonzero(non_key_frame_mask).reshape(-1).tolist()
        #     batch_key_frame_idx.append(key_frame_idx)
        #     batch_non_key_frame_idx.append(non_key_frame_idx)

        batch_key_frame_idx, batch_non_key_frame_idx = self.select_key_frame(video_embeds.reshape(b,t,*video_embeds.shape[1:]),batch_subtitle_durations,durations,query)

        ### Apply Q-Former for Image Embeddings ####################################
        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)
        frames_after_qformer = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        frames_for_projection = frames_after_qformer.last_hidden_state  # bt, n_query, d
        # non key frame compress
        frames_for_projection = frames_for_projection.reshape(b,t,self.num_query_token,-1)  # b,t, n_query, d
        batch_frames,batch_frames_atts = [], [] # b, t, n_query or n_query/combine_ratio, c; b, t, n_query or n_query/combine_ratio
        for key_frame_idx, non_key_frame_idx, frames in zip(batch_key_frame_idx,batch_non_key_frame_idx,frames_for_projection):
            key_frames,non_key_frames = frames[key_frame_idx], frames[non_key_frame_idx] # t_k, n_query, d; t_n, n_query, d
            non_key_frames = self.query_proj(non_key_frames.transpose(-1, -2)).transpose(-1, -2)    # t_n, n_query / combine_ratio, d
            key_frames = self.t5_proj(key_frames) # t_k, n_query, c
            non_key_frames = self.t5_proj(non_key_frames) # t_n, n_query / combine_ratio, c
            frames, frames_atts = [], []
            t_n, t_k = 0, 0
            for t_i in range(t):
                if t_k<len(key_frame_idx) and t_i == key_frame_idx[t_k]:
                    frames.append(key_frames[t_k])
                    t_k += 1
                else:
                    frames.append(non_key_frames[t_n])
                    t_n += 1
                frames_atts.append(torch.ones(frames[-1].shape[:-1], dtype=torch.long, device=device))
            assert len(frames) == t and len(frames_atts) == t
            batch_frames.append(frames)
            batch_frames_atts.append(frames_atts)

        # del video_embeds, video, cluster_features, frames_for_projection

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            ### Construct moment retrival prompt ######################################
            inputs_embs, inputs_atts, video_prompt, text_prompt, subtitle_prompt, embed_lens = self.mr_prompt_concatenation(
                timestamps, durations, batch_frames, batch_frames_atts, batch_subtitles, query, task_prompt,
                item_prompt=item_prompt,device=device)
            # print(f"{num_iters}:{qid}:{list(inputs_embs.shape)}:{embed_lens}:{query}:{subtitle_prompt}")
            ### Encode mr answer ################################################
            for a, ap in self.annoying_numbers_replacement_dict.items():
                relevant_windows = [mw.replace(str(a), str(ap)) for mw in relevant_windows]

            output_tokens = self.t5_tokenizer(
                relevant_windows,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            output_tokens_mask = output_tokens.attention_mask

            ### Apply moment retrival prompt ######################################
            outputs_loc = self.t5_model(
                inputs_embeds=inputs_embs,
                attention_mask=inputs_atts,
                decoder_attention_mask=output_tokens_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs_loc.loss
            pred_windows = self.t5_tokenizer.batch_decode(
                torch.argmax(outputs_loc.logits, dim=-1)
            )

        # write the following to a wandb table
        if self.use_wandb and is_main_process():
            log = {}
            log["train/log_likelihood_loss"] = loss.item()
            # Log images and predictions
            if samples["iters"] % self.log_samples_every_n == 0:
                out, self.wandb_table_data = (
                    format_wandb_log_images_and_predictions(
                        samples=samples,
                        wandb_table_data=self.wandb_table_data,
                        pred=pred_windows,
                        target=relevant_windows,
                        video_prompt=video_prompt,
                        text_prompt=text_prompt,
                        post_process_fn=self.post_process,
                        train_data=True,
                        etc={
                            "subtitle_prompt": subtitle_prompt
                        }
                    )
                )
                log.update(out)
            # Log iteration
            wandb.log(log)

        pred_windows_processed = [self.post_process(pred_window, duration) for pred_window, duration in
                                  zip(pred_windows, durations)]
        iou = [self.get_iou(gt, pred, n_frms=d) for gt, pred, d in
               zip(relevant_windows, pred_windows_processed, durations)]
        miou = sum(iou) / b
        if self.input_time_format == 'frame_indices':
            durations = durations_orig
            relevant_windows = relevant_windows_orig
            timestamps,pred_windows_processed=self.frame_indices_to_absolute_times(timestamps, pred_windows_processed, durations, amount=self.format_target_amount)

        result_str = [f'{_qid}: \traw_pred: {pw}\tpred: {pwp}\t gt: {rw}' for _qid, pw, pwp, rw in
                      zip(qid, pred_windows, pred_windows_processed, relevant_windows)]
        print(f"{num_iters}: {list(inputs_embs.shape)}: loss:{loss.item():.3f}: iou:{miou:.3f}\n" + "\n".join(result_str))
        return {"loss": loss}

    def two_side_bias_padding(self,clusters,bias=3):
        n_frames = len(clusters)
        new_clusters = torch.zeros(n_frames, dtype=torch.int,device=clusters.device)
        for i in range(n_frames):
            if clusters[i] > 0:
                for j in range(i - bias, i + bias+1):
                    if j >= 0 and j < n_frames:
                        new_clusters[j] = 1
        return new_clusters

    def mr_prompt_concatenation(
        self,
        timestamps,
        durations,
        batch_frames,
        batch_frames_atts,
        batch_subtitles,
        query_prompt,
        task_prompt,
        item_prompt=None,
        subtitle_topk=5,
        subtitle_threshold=0.9,
        device=None
    ):

        b = len(durations)
        def word_embedding(word):
            token = self.t5_tokenizer(word,add_special_tokens=False).input_ids
            token = [t for t in token if t!=3]
            return token, self.t5_model.encoder.embed_tokens(
                torch.tensor(token)
                .to(device)
            )

        # do filter for subtitle
        base_query = [q[7:] for q in query_prompt]
        batch_query_prompts_tokens = self.tokenizer(
            base_query,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        batch_query_prompts_embeds = self.Qformer.bert(
            batch_query_prompts_tokens.input_ids,
            attention_mask=batch_query_prompts_tokens.attention_mask,
            return_dict=True,
        )
        batch_query_prompts_embeds = batch_query_prompts_embeds.last_hidden_state[:, 0, :]  # b, dim
        for bid, (query_prompt_embs, subtitles) in enumerate(zip(batch_query_prompts_embeds, batch_subtitles)):
            query_prompt_embs = query_prompt_embs.unsqueeze(0)
            subtitles_text = [subtitle['text'] for subtitle in subtitles]  # n_subtitle, subtitle_len
            if len(subtitles_text) == 0:
                subtitles = [{'start':-1,'end':-1,'text':'*None*'}]
            else:
                subtitles_text_tokens = self.tokenizer(
                    subtitles_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(device)
                subtitles_text_embs = self.Qformer.bert(
                    subtitles_text_tokens.input_ids,
                    attention_mask=subtitles_text_tokens.attention_mask,
                    return_dict=True,
                )
                subtitles_text_embs = subtitles_text_embs.last_hidden_state[:, 0, :]  # subs_len , dim
                sims = F.cosine_similarity(query_prompt_embs, subtitles_text_embs)
                sim_indices = torch.argsort(sims, descending=True)[:subtitle_topk]
                sim_indices = [sim_index for sim_index in sim_indices if sims[sim_index]>subtitle_threshold]
                subtitles = [subtitles[i] for i in sim_indices]
                if len(subtitles) == 0:
                    subtitles = [{'start': -1, 'end': -1, 'text': '*None*'}]
            batch_subtitles[bid] = subtitles

        del batch_query_prompts_embeds

        # get the tokens and embeddings for the subtitle (start, text, end)
        batch_subtitle_start_timestamps, _, _ = get_timestamps_as_seconds_integers(
            [[s['start'] for s in subtitle] for subtitle in batch_subtitles], durations,
            self.annoying_numbers_replacement_dict
        )
        batch_subtitle_end_timestamps, _, _ = get_timestamps_as_seconds_integers(
            [[s['end'] for s in subtitle] for subtitle in batch_subtitles], durations,
            self.annoying_numbers_replacement_dict
        )
        batch_subtitle_texts = [[s['text'] for s in subtitles] for subtitles in batch_subtitles]

        batch_subtitles_start_tokens, batch_subtitles_end_tokens = [], []
        batch_subtitles_start_embs, batch_subtitles_end_embs = [], []
        batch_subtitles_text_tokens, batch_subtitles_text_embs = [], []
        for sst, set, st in zip(batch_subtitle_start_timestamps, batch_subtitle_end_timestamps,batch_subtitle_texts):
            tokens, embs = self.get_clean_timestamp_tokens_and_embs(sst)
            batch_subtitles_start_tokens.append(tokens)
            batch_subtitles_start_embs.append(embs)
            tokens, embs = self.get_clean_timestamp_tokens_and_embs(set)
            batch_subtitles_end_tokens.append(tokens)
            batch_subtitles_end_embs.append(embs)
            one_batch_subtitle_text_tokens = []
            one_batch_subtitle_text_embs = []
            for subtitle in st:
                subtitle_text_tokens = self.t5_tokenizer(
                    [subtitle],
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(device)
                subtitle_text_embs = self.t5_model.encoder.embed_tokens(
                    subtitle_text_tokens.input_ids
                )
                one_batch_subtitle_text_tokens.append(subtitle_text_tokens.input_ids[0])
                one_batch_subtitle_text_embs.append(subtitle_text_embs[0])
            batch_subtitles_text_embs.append(one_batch_subtitle_text_embs)
            batch_subtitles_text_tokens.append(one_batch_subtitle_text_tokens)

        if item_prompt:
            text_prompt = [i + it + t for i,it, t in zip(query_prompt, item_prompt, task_prompt)]
        else:
            text_prompt = [i + t for i, t in zip(query_prompt, task_prompt)]
        text_prompt_tokens = self.t5_tokenizer(
            text_prompt,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        text_prompt_embs = self.t5_model.encoder.embed_tokens(
            text_prompt_tokens.input_ids
        )


        # get the tokens and embeddings for the timestamps and durations

        timestamps, durations, video_prompt = get_timestamps_as_seconds_integers(
            timestamps, durations, self.annoying_numbers_replacement_dict
        )

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
        subtitle_prompt_embs = []
        subtitle_prompt = []

        video_sign_part_token,video_sign_part_emb = word_embedding(self.video_sign_part)
        # video_sign_frame_token,video_sign_frame_emb = word_embedding(self.video_sign_frame)
        # video_sign_time_token,video_sign_time_emb = word_embedding(self.video_sign_time)
        video_sign_duration_token,video_sign_duration_emb = word_embedding(self.video_sign_duration)
        subs_sign_part_token,subs_sign_part_emb = word_embedding(self.subtitle_sign_part)
        subs_sign_text_token, subs_sign_text_emb = word_embedding(self.subtitle_sign_text)
        subs_sign_time_start_token,subs_sign_time_start_emb = word_embedding(self.subtitle_sign_time_start)
        subs_sign_time_end_token,subs_sign_time_end_emb = word_embedding(self.subtitle_sign_time_end)

        # iterate over the batch
        for j, (timestamp_embs, frame_embs,subs_start_embs,subs_text_embs,subs_end_embs) in enumerate(
            zip(batch_timestamps_embs, batch_frames,
                batch_subtitles_start_embs, batch_subtitles_text_embs, batch_subtitles_end_embs
                )
        ):
            interleaved_embed = torch.tensor([]).to(device)
            _video_prompt = []
            t = len(frame_embs)
            for i in range(t):
                frame_emb = frame_embs[i]
                timestamp_emb = timestamp_embs[i]

                # for logging of input design
                _video_prompt.append(f"[f{i}-{frame_emb.shape[0]}]{timestamps[j][i]}")

                # frame i and corresponding timestamp
                frame_and_time = torch.cat([
                    frame_emb,timestamp_emb
                ])
                # add frame and timestamp "pair" to the interleaved prompt
                interleaved_embed = torch.cat([interleaved_embed, frame_and_time])

            duration_emb = batch_duration_embs[j]
            interleaved_embed = torch.cat(
                [interleaved_embed, video_sign_duration_emb, duration_emb, video_sign_part_emb]
            )
            interleaved_video_prompt_embs.append(interleaved_embed)
            _video_prompt.append(f"{self.video_sign_duration}{durations[j]}{self.video_sign_part}")
            video_prompt.append("".join(_video_prompt))

            ##### construct embed of subtitle
            subtitle_prompt_emb = torch.tensor([]).to(device)
            _subtitle_prompt = []
            for i,(subs_start_emb, subs_text_emb, subs_end_emb) in enumerate(zip(subs_start_embs,subs_text_embs,subs_end_embs)):
                one_subtitle_prompt = torch.cat([subs_sign_time_start_emb,subs_start_emb,
                                                 subs_text_emb,
                                                 subs_end_emb,subs_sign_time_end_emb])
                subtitle_prompt_emb = torch.cat([subtitle_prompt_emb, one_subtitle_prompt])
                _subtitle_prompt.append(f'{self.subtitle_sign_time_start}{self.t5_tokenizer.decode(batch_subtitles_start_tokens[j][i])}'
                                        f'{self.t5_tokenizer.decode(batch_subtitles_text_tokens[j][i])}'
                                        f'{self.t5_tokenizer.decode(batch_subtitles_end_tokens[j][i])}{self.subtitle_sign_time_end}')

            subtitle_prompt_emb = torch.cat([subtitle_prompt_emb,subs_sign_part_emb])
            subtitle_prompt_embs.append(subtitle_prompt_emb)
            _subtitle_prompt.append(self.subtitle_sign_part)
            subtitle_prompt.append("".join(_subtitle_prompt))

        # if interleaved_video_prompt_embs elements are not same length, pad them
        # should only be necessary if we allow for timestamp numbers to be tokenize to more than 1 token

        max_vid_len = max([len(i) for i in interleaved_video_prompt_embs])
        interleaved_video_prompt_attn_embs = torch.ones((b, max_vid_len), dtype=torch.long).to(device)
        for i in range(len(interleaved_video_prompt_embs)):
            if len(interleaved_video_prompt_embs[i]) < max_vid_len:
                pad_len = max_vid_len - len(interleaved_video_prompt_embs[i])
                padding = self.pad_token_id * torch.ones(
                    pad_len,
                    interleaved_video_prompt_embs[i].shape[-1],
                ).to(device)
                interleaved_video_prompt_embs[i] = torch.cat(
                    [padding, interleaved_video_prompt_embs[i]]
                )
                interleaved_video_prompt_attn_embs[i,:pad_len] = 0

        interleaved_video_prompt_embs = torch.stack(
            interleaved_video_prompt_embs
        ).to(device)

        max_subtitle_len = max([len(i) for i in subtitle_prompt_embs])
        subtitle_prompt_attn_embs = torch.ones((b, max_subtitle_len), dtype=torch.long).to(device)
        for i in range(len(subtitle_prompt_embs)):
            if len(subtitle_prompt_embs[i]) < max_subtitle_len:
                pad_len = max_subtitle_len - len(subtitle_prompt_embs[i])
                padding = self.pad_token_id * torch.ones(
                    pad_len,
                    subtitle_prompt_embs[i].shape[-1],
                ).to(device)
                subtitle_prompt_embs[i] = torch.cat(
                    [padding, subtitle_prompt_embs[i]]
                )
                subtitle_prompt_attn_embs[i,:pad_len] = 0

        subtitle_prompt_embs = torch.stack(
            subtitle_prompt_embs
        ).to(device)

        ### Concatenate interleaved_video_prompt, video_prompt_end, text_prompt
        inputs_embs_mr = torch.cat(
            [
                interleaved_video_prompt_embs,
                subtitle_prompt_embs,
                text_prompt_embs,
            ],
            dim=1,
        )

        embed_lens = [interleaved_video_prompt_embs.shape[1],subtitle_prompt_embs.shape[1],text_prompt_embs.shape[1]]

        inputs_atts_mr = torch.cat(
            [
                interleaved_video_prompt_attn_embs,
                subtitle_prompt_attn_embs,
                text_prompt_tokens.attention_mask,
            ],
            dim=1,
        )

        return inputs_embs_mr, inputs_atts_mr, video_prompt, text_prompt, subtitle_prompt, embed_lens

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=70,  # 50
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
    ):
        out = {}
        qid = samples["query_id"]
        video = samples["video"]
        timestamps, durations = samples["timestamps"].tolist(), samples["duration"].tolist()
        batch_subtitles = samples["subtitles"]
        batch_subtitle_durations = samples["subtitle_durations"]
        query, task_prompt = samples["query_prompt"], samples["task_prompt"]
        item_prompt = samples['item_prompt'] if 'item' in self.task else None

        relevant_windows = samples["relevant_windows"]
        num_iters = samples["iters"]

        device = video.device
        # uniform sampling
        b, t, c, w, h = video.shape
        video = video.reshape(-1, c, w, h)
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            video_embeds = self.ln_vision(self.visual_encoder(video))  # bt, n, c
        _, n, _ = video_embeds.shape
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(device)  # bt n c

        if self.input_time_format == 'frame_indices':
            durations_orig = copy.deepcopy(durations)
            relevant_windows_orig = copy.deepcopy(relevant_windows)
            timestamps, relevant_windows, durations =self.absolute_times_to_frame_indices(timestamps, relevant_windows, durations, amount=self.format_target_amount)
        ####### cluster #########
        # cluster_features = torch.mean(video_embeds, dim=1, keepdim=False).reshape(b, t, -1).clone()
        # cluster_dict = {'x': cluster_features,
        #                 'token_num': cluster_features.size(1),
        #                 'idx_token': torch.arange(t)[None, :].repeat(b, 1),
        #                 'agg_weight': cluster_features.new_ones(b, t, 1),
        #                 'mask': None}
        # cluster_dict = self.cluster(cluster_dict)
        # batch_cluster_idx = cluster_dict['idx_token']
        #
        # ###### get key frame ######
        # batch_key_frame_idx, batch_non_key_frame_idx = [], []
        # for cluster_idx,subtitle_durations,duration,q in zip(batch_cluster_idx, batch_subtitle_durations, durations,query):
        #     # native key frame (cluster edges)
        #     key_frame_mask = torch.zeros(t, dtype=torch.int)
        #     for i in range(1, t):
        #         if cluster_idx[i - 1] != cluster_idx[i]:  # frame between different cluster is key frame
        #             key_frame_mask[i - 1] = 1
        #             key_frame_mask[i] = 1
        #     key_frame_mask[0] = 1
        #     key_frame_mask[-1] = 1
        #     key_frame_mask_copy = key_frame_mask.clone()
        #     for i in range(0, t):
        #         for bi in range(-self.bias, self.bias + 1):
        #             if 0 <= i + bi < t and key_frame_mask_copy[i + bi] > 0:
        #                 key_frame_mask[i] = 1
        #                 break
        #     # subtitle exist frame is key frame for talk relative
        #     if self.is_talk_relative(q) and subtitle_durations:
        #         key_frame_mask = torch.zeros(t, dtype=torch.int)
        #         for subtitle_duration in subtitle_durations:
        #             start = int(subtitle_duration[0] * t / duration)
        #             end = min(int(subtitle_duration[1] * t / duration),t-1)
        #             for i in range(start, end + 1):
        #                 key_frame_mask[i] = 1
        #
        #     non_key_frame_mask = 1 - key_frame_mask
        #     key_frame_idx = torch.nonzero(key_frame_mask).reshape(-1).tolist()
        #     non_key_frame_idx = torch.nonzero(non_key_frame_mask).reshape(-1).tolist()
        #     batch_key_frame_idx.append(key_frame_idx)
        #     batch_non_key_frame_idx.append(non_key_frame_idx)

        batch_key_frame_idx, batch_non_key_frame_idx = self.select_key_frame(video_embeds.reshape(b, t, *video_embeds.shape[1:]), batch_subtitle_durations, durations, query)

        ### Apply Q-Former for Image Embeddings ####################################
        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)
        frames_after_qformer = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        frames_for_projection = frames_after_qformer.last_hidden_state
        # non key frame compress
        frames_for_projection = frames_for_projection.reshape(b, t, self.num_query_token, -1)  # b,t, n_query, d
        batch_frames, batch_frames_atts = [], []  # b, t, n_query or n_query/combine_ratio, c; b, t, n_query or n_query/combine_ratio
        for key_frame_idx, non_key_frame_idx, frames in zip(batch_key_frame_idx, batch_non_key_frame_idx, frames_for_projection):
            key_frames, non_key_frames = frames[key_frame_idx], frames[
                non_key_frame_idx]  # t_k, n_query, d; t_n, n_query, d
            non_key_frames = self.query_proj(non_key_frames.transpose(-1, -2)).transpose(-1,
                                                                                         -2)  # t_n, n_query / combine_ratio, d
            key_frames = self.t5_proj(key_frames)  # t_k, n_query, c
            non_key_frames = self.t5_proj(non_key_frames)  # t_n, n_query / combine_ratio, c
            frames, frames_atts = [], []
            t_n, t_k = 0, 0
            for t_i in range(t):
                if t_k<len(key_frame_idx) and t_i == key_frame_idx[t_k]:
                    frames.append(key_frames[t_k])
                    t_k += 1
                else:
                    frames.append(non_key_frames[t_n])
                    t_n += 1
                frames_atts.append(torch.ones(frames[-1].shape[:-1], dtype=torch.long, device=device))
            assert len(frames) == t and len(frames_atts) == t
            batch_frames.append(frames)
            batch_frames_atts.append(frames_atts)

        # del video_embeds, video, cluster_features, frames_for_projection

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            ### Construct moment retrival prompt ######################################
            inputs_embs, inputs_atts, video_prompt, text_prompt, subtitle_prompt, embed_lens = self.mr_prompt_concatenation(
                timestamps, durations, batch_frames, batch_frames_atts, batch_subtitles, query, task_prompt,
                item_prompt=item_prompt, device=device)
            # print(f"{num_iters}:{qid}:{list(inputs_embs.shape)}:{embed_lens}:{query}:{subtitle_prompt}")
            ### Apply moment retrival prompt ######################################
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embs,
                attention_mask=inputs_atts,
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

            # tokenizer decode mr outputs
            pred_windows = self.t5_tokenizer.batch_decode(
                outputs[0], skip_special_tokens=True
            )
            pred_windows_processed = [self.post_process(pred_window, duration) for pred_window, duration in zip(pred_windows, durations)]

        # write the following to a wandb table
        if self.use_wandb and is_main_process():
            # Log images and predictions
            if num_iters % self.log_samples_every_n_eval == 0:
                table, self.wandb_table_data_eval = (
                    format_wandb_log_images_and_predictions(
                        samples=samples,
                        wandb_table_data=self.wandb_table_data_eval,
                        pred=pred_windows,
                        target=relevant_windows,
                        video_prompt=video_prompt,
                        text_prompt=text_prompt,
                        post_process_fn=self.post_process,
                        train_data=False,
                        etc={
                            "subtitle_prompt": subtitle_prompt
                        }
                    )
                )
                # Log iteration
                wandb.log(table)

        if self.input_time_format == 'frame_indices':
            durations = durations_orig
            relevant_windows = relevant_windows_orig
            timestamps, pred_windows_processed = self.frame_indices_to_absolute_times(timestamps, pred_windows_processed, durations, amount=self.format_target_amount)

        out["duration"] = durations
        out["raw_pred_windows"] = pred_windows
        out["pred_windows"] = pred_windows_processed
        out["gt_windows"] = relevant_windows
        out["qid"] = qid
        out["query"] = samples["query_prompt"]

        result_str = [f'{_qid}: \traw_pred: {pw}\tpred: {pwp}\t gt: {rw}' for _qid, pw, pwp, rw in
                      zip(qid, pred_windows, pred_windows_processed, relevant_windows)]
        print(f"{num_iters}: {list(inputs_embs.shape)}\n" + "\n".join(result_str))
        return out

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
        format_target_amount = cfg.get("format_target_amount",30)
        interleave_data = cfg.get("interleave_data", True)
        frame_token_aggregation = cfg.get("frame_token_aggregation", None)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_len", 200)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        task = cfg.get("task", "qformer_freeze_lora")
        num_clusters = cfg.get("num_clusters", 12)
        bias1 = cfg.get("bias1", 1)
        key_frame_selection = cfg.get("key_frame_selection","cluster")

        debug = cfg.get("debug", False)

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
            format_target_amount=format_target_amount,
            interleave_data=interleave_data,
            frame_token_aggregation=frame_token_aggregation,
            key_frame_selection=key_frame_selection,
            num_clusters=num_clusters,
            bias1=bias1,
            task=task,
            debug=debug
        )
        if not debug:
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
        tokenizer=T5TokenizerFast.from_pretrained("google/flan-t5-xl",local_files_only=True),
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
        if len(timestamps) == 0:
            return [],[]
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

    def get_iou(self,gt,pred,n_frms):
        gt_s = np.zeros(n_frms+1, dtype=int)
        pred_s = np.zeros(n_frms+1, dtype=int)
        if isinstance(gt,str):
            gt = moment_str_to_list(gt)
        if isinstance(pred,str):
            pred = moment_str_to_list(pred)
        for relevant_window in gt:
            start, end = relevant_window
            start = int(start)
            end = int(end)
            gt_s[start:end] = 1
        for relevant_window in pred:
            start, end = relevant_window
            start = int(start)
            end = int(end)
            pred_s[start:end] = 1
        iou = np.sum(gt_s & pred_s) / np.sum(gt_s | pred_s)
        return iou

    def two_side_bias_padding(self,clusters,bias=5):
        n_frames = len(clusters)
        new_clusters = torch.zeros(n_frames, dtype=torch.int,device=clusters.device)
        for i in range(n_frames):
            if clusters[i] > 0:
                for j in range(i - bias, i + bias+1):
                    if j >= 0 and j < n_frames:
                        new_clusters[j] = 1
        return new_clusters

    def absolute_times_to_frame_indices(self, batch_absolute_times, batch_absolute_time_relevant_windows, batch_duration, amount=20):
        batch_frame_indices = []
        batch_frame_index_relevant_windows = []

        for absolute_times, absolute_time_relevant_windows,  duration in zip(batch_absolute_times,batch_absolute_time_relevant_windows,batch_duration):
            absolute_time_relevant_windows = moment_str_to_list(absolute_time_relevant_windows)
            frame_index_relevant_windows = [[round((t*amount/duration)) for t in w] for w in absolute_time_relevant_windows]
            batch_frame_index_relevant_windows.append(str(frame_index_relevant_windows))
            frame_indices = [round((t*amount/duration)) for t in absolute_times]
            batch_frame_indices.append(frame_indices)
        batch_duration = [amount]*len(batch_duration)
        return batch_frame_indices,batch_frame_index_relevant_windows, batch_duration

    def frame_indices_to_absolute_times(self, batch_frame_indices, batch_frame_index_pred_windows, batch_duration, amount=20):
        batch_absolute_times = []
        batch_absolute_time_pred_windows = []
        for frame_indices, frame_index_pred_windows, duration in zip(batch_frame_indices,batch_frame_index_pred_windows, batch_duration):
            absolute_time_pred_windows = [[(t/amount)*duration for t in w] for w in frame_index_pred_windows]
            batch_absolute_time_pred_windows.append(absolute_time_pred_windows)
            absolute_times = [(fi/amount)*duration for fi in frame_indices]
            batch_absolute_times.append(absolute_times)
        return batch_absolute_times,batch_absolute_time_pred_windows
