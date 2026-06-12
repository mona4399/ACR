from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import imageio.v3 as iio
import csv

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().split() for line in lines]
    paths, labels = zip(*data)
    return paths, labels

def load_txt_file_kinetics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().rsplit(' ', 1) for line in lines]
    paths, labels = zip(*data)
    return paths, labels

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', eval=False, cfg=None, cfg_flow=None, sample_dur=10, dataset='HMDB', datapath=''):
        self.base_path = datapath
        self.split = split
        self.interval = 9
        self.sample_dur = sample_dur

        # build the data pipeline
        if split == 'train' and not eval:
            train_pipeline = cfg.data.train.pipeline
            self.pipeline = Compose(train_pipeline)
            train_pipeline_flow = cfg_flow.data.train.pipeline
            self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.pipeline = Compose(val_pipeline)
            val_pipeline_flow = cfg_flow.data.val.pipeline
            self.pipeline_flow = Compose(val_pipeline_flow)

        if dataset == "HAC":
            self.samples = []
            self.labels = []
            train_file_name = "splits/" + dataset + "_" + split + "_only_cartoon.csv"
            with open(train_file_name) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.samples.append(row[0])
                    self.labels.append(row[1])
        else:
            train_file_name = "splits/" + dataset + "_" + split + ".txt"

            if dataset == "Kinetics" or dataset == "Kinetics100":
                self.samples, self.labels = load_txt_file_kinetics(train_file_name)
            else:
                self.samples, self.labels = load_txt_file(train_file_name)

        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.dataset = dataset

    def __getitem__(self, index):
        video_path = ''
        if self.dataset == "Kinetics" or self.dataset == "Kinetics100":
            video_file = self.base_path + 'Kinetics-600-train/' + self.samples[index]
        elif self.dataset == "HAC":
            video_file = self.base_path + 'cartoon/videos/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/defocus_blur/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/frost/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/brightness/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/pixelate/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/jpeg_compression/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/gaussian_noise_5/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/gaussian_noise/' + self.samples[index]
        else:
            video_file = self.base_path + 'video/' + self.samples[index]

        vid = iio.imread(video_file, plugin="pyav")

        frame_num = vid.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index,
            video=vid,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl,
            modality=modality)
        data, frame_inds = self.pipeline(data)

        if self.dataset == "Kinetics" or self.dataset == "Kinetics100":
            v_name_pure = self.samples[index][:-4]
        elif self.dataset == "HMDB":
            v_name = self.samples[index]
            start_index = v_name.index('/') + 1
            end_index = v_name.index('.')
            v_name_pure = v_name[start_index:end_index]
        elif self.dataset == "UCF":
            v_name = self.samples[index]
            end_index = v_name.index('.')
            v_name_pure = v_name[:end_index]

        if self.dataset == "Kinetics" or self.dataset == "Kinetics100":
            video_file_x = self.base_path + 'Kinetics-600-train-flow/'+ v_name_pure + '_flow_x.mp4'
            video_file_y = self.base_path + 'Kinetics-600-train-flow/'+ v_name_pure + '_flow_y.mp4'
        elif self.dataset == "HAC":
            video_file_x = self.base_path + 'cartoon/flow/'+ self.samples[index][:-4] + '_flow_x.mp4'
            video_file_y = self.base_path + 'cartoon/flow/'+ self.samples[index][:-4] + '_flow_y.mp4'
        else:
            video_file_x = self.base_path + 'flow/' + v_name_pure + '_flow_x.mp4'
            video_file_y = self.base_path + 'flow/' + v_name_pure + '_flow_y.mp4'

        vid_x = iio.imread(video_file_x, plugin="pyav")
        vid_y = iio.imread(video_file_y, plugin="pyav")

        frame_num = vid_x.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
        modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
        start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame)
        flow = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index_flow,
            video=vid_x,
            video_y=vid_y,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl_flow,
            modality=modality_flow)
        flow, frame_inds_flow = self.pipeline_flow(flow)

        label1 = int(self.labels[index])

        return data, flow, label1, index


    def __len__(self):
        return len(self.samples)

