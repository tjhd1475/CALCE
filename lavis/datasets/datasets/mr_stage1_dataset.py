import os, random,copy, re
from datetime import datetime


import torch
import numpy as np

from lavis.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate

class MRStage1Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, etc=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, etc)
        self.srt_root = vis_root.replace("videos","srts")
        if 'sampling' in etc:
            sampling = etc['sampling']
            assert etc['sampling'] in ['random','fix','all','keyword','no_keyword']
            if sampling != 'all':
                if sampling == 'random':
                    num_sampling = etc['num_sampling']
                    self.annotation = random.sample(self.annotation,num_sampling)
                elif sampling == 'fix':
                    qid_set = etc['qid_set']
                    self.annotation = [d for d in self.annotation if int(d['qid'].split('_')[1]) in set(qid_set)]
                elif sampling == 'keyword':
                    keyword_set = etc['key_set']
                    self.annotation = [d for d in self.annotation if any(keyword in d['query'] for keyword in keyword_set)]
                elif sampling == 'no_keyword':
                    keyword_set = etc['key_set']
                    self.annotation = [d for d in self.annotation if not
                                       any(keyword in d['query'] for keyword in keyword_set)]
                self._add_instance_ids()
        self.use_rewrite = etc.get('use_rewrite', False)
        if self.use_rewrite:
            self.indices = []
            for i1, anno in enumerate(self.annotation):
                for i2 in range(len(anno['rewrite'])):
                    self.indices.append([i1, i2])
            random.shuffle(self.indices)
        self.no_audio = etc.get('no_audio',False)

    def __len__(self):
        if self.use_rewrite:
            return len(self.indices)
        else:
            return len(self.annotation)

    def __getitem__(self, index):
        if self.use_rewrite:
            i1, i2 = self.indices[index]
            ann = self.annotation[i1]
            ann['query'] = ann['rewrite'][i2]
        else:
            ann = self.annotation[index]

        # set video clip if 'start'&'end' timestamp in data
        if "start" in ann:
            start, end = float(ann["start"]), float(ann["end"])
            # start, end = int(float(ann["start"]) * 100), int(float(ann["end"]) * 100)
            clip = [start, end]
        else:
            clip = None

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname + ".mp4")
        srt_path = os.path.join(self.srt_root,vname+".srt")

        frms, indices, fps = self.vis_processor(video_path, clip_proposal=clip)

        duration = ann["duration"]
        if self.no_audio:
            subtitles, subtitle_durations = [], []
        else:
            subtitles, subtitle_durations = self.read_srt_file(srt_path,duration)
        query = ann["query"]
        items = ann["items"]
        relevant_windows = str(ann["relevant_windows"])

        query_prompt = "Query: " + query + "\n"
        item_prompt = "Items: " + ",".join(items) + "\n"
        task_prompt = "Given the video and the subtitle and the query, find the query relevant windows.\nRelevant windows: "\

        # generate video prompt in the following format:
        # <vid><t><t+1><t+2>…<duration>[frame embeddings]</vid>
        # where <vid> is the video id, and <t> are the timestamps of each frame
        frms = frms.permute(1, 0, 2, 3)
        timestamps = [float(idx / fps) for idx in indices]


        # timestamps = [round(t, 2) for t in timestamps]
        # timestamps.append(duration)
        timestamps = torch.tensor(timestamps)

        duration = torch.tensor(duration)

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": frms,
            "duration": duration,
            "query_id": ann["qid"],
            "timestamps": timestamps,
            "query_prompt": query_prompt,
            "item_prompt": item_prompt,
            "task_prompt": task_prompt,
            "relevant_windows": relevant_windows,
            "subtitles":subtitles,
            "subtitle_durations":subtitle_durations
        }

    from datetime import datetime

    def parse_time_to_seconds(self,srt_time):
        time_format = "%H:%M:%S,%f"  # SRT 时间格式
        dt = datetime.strptime(srt_time, time_format)
        total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000
        return total_seconds

    filter_words = ['the', 'a', 'o', 'oh', 'yeah', 'uh', 'ah', 'yep', 'uh-no', 'hmm']

    # filter_words = ['i','to','the','a','you','and','were','was','we','are','is','he','she','him','his','uh','o','yeah','yes','oh','this','that','our']
    def text_filter(self,subtitle, filter_non_eng=0.8):
        def is_english_word(word):
            return bool(re.match(r'^[a-zA-Z]+$', word))

        text = subtitle['text']
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        words = text.split()
        english_words = [word for word in words if is_english_word(word)]
        if len(english_words) < len(words) * filter_non_eng:
            return None
        # 过滤掉包含特定词的单词
        filtered_words = [word for word in words if word not in self.filter_words]
        if len(filtered_words) == 0:
            return None
        # 将过滤后的单词连接回句子
        subtitle['text'] = ' '.join(filtered_words)
        return subtitle['text']
        # return text

    def read_srt_file(self,file_path, duration, t=2.5):
        subtitles = []

        def add(subtitle=None):
            if subtitle and 'text' in subtitle:
                subtitles.append(copy.deepcopy(subtitle))

            subtitle.clear()

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            subtitle = {}

            for line in lines:
                line = line.strip()
                if line.isdigit():  # 这是字幕的序号
                    add(subtitle)
                elif '-->' in line:  # 这是时间戳
                    start_time, end_time = line.split(' --> ')
                    subtitle['start'] = self.parse_time_to_seconds(start_time)
                    subtitle['end'] = self.parse_time_to_seconds(end_time)
                elif line:  # 这是字幕文本
                    if 'text' in subtitle:
                        subtitle['text'] += ' ' + line
                    else:
                        subtitle['text'] = line

            # 添加最后一个字幕
            add(subtitle)


        combined_subtitles = []
        i = 0
        while i < len(subtitles):
            j = i + 1
            now_end = subtitles[i]['end']
            while j < len(subtitles):
                if subtitles[j]['start'] == now_end and subtitles[j]['text'] == subtitles[i]['text']:
                    now_end = subtitles[j]['end']
                    j += 1
                else:
                    break
            j -= 1
            combined_subtitles.append({'start':subtitles[i]['start'],'text':subtitles[i]['text'],'end':subtitles[j]['end']})
            i = j + 1

        subtitle_durations = []
        i = 0
        while i < len(subtitles):
            j = i + 1
            now_end = subtitles[i]['end']
            while j < len(subtitles):
                if subtitles[j]['start'] <= now_end + t:
                    now_end = subtitles[j]['end']
                    j += 1
                else:
                    break
            j -= 1
            subtitle_durations.append([subtitles[i]['start'],subtitles[j]['end']])
            i = j + 1
        combined_subtitles = [subtitle for subtitle in combined_subtitles if self.text_filter(subtitle)]
        return combined_subtitles, subtitle_durations

    def collater(self, samples):
        subtitles = [copy.deepcopy(sample['subtitles']) for sample in samples]
        subtitle_durations = [copy.deepcopy(sample['subtitle_durations']) for sample in samples]
        for sample in samples:
            del sample['subtitles'], sample['subtitle_durations']
        samples = default_collate(samples)
        samples['subtitles'] = subtitles
        samples['subtitle_durations'] = subtitle_durations
        return samples

if __name__ == '__main__':
    from lavis.processors.blip_processors import Blip2VideoTrainProcessor, BlipQuestionProcessor
    vis_pro = Blip2VideoTrainProcessor(n_frms=75, image_size=224)
    txt_pro = BlipQuestionProcessor(max_words=50)
    dataset = MRStage1Dataset(vis_pro,txt_pro,
                          '/data2/xiepeiyu/qvhighlights/videos',
                          ['/data2/xiepeiyu/qvhighlights/annotations/lavis/val_mrhd.json']
                          )
    item = dataset.__getitem__(2)