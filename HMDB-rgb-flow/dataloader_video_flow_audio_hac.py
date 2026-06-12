from mmaction.datasets.pipelines import Compose
import torch.utils.data
import csv
import soundfile as sf
from scipy import signal
import numpy as np
import os
import imageio.v3 as iio

def get_spectrogram_piece(samples, start_time, end_time, duration, samplerate, training=False):
    start1 = start_time / duration * len(samples)
    end1 = end_time / duration * len(samples)
    start1 = int(np.round(start1))
    end1 = int(np.round(end1))
    samples = samples[start1:end1]

    resamples = samples[:160000]
    if len(resamples) == 0:
        resamples = np.zeros((160000))
    while len(resamples) < 160000:
        resamples = np.tile(resamples, 10)[:160000]

    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)

    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    interval = 9
    if training is True:
        noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
        spectrogram = spectrogram + noise
        start1 = np.random.choice(256 - interval, (1,))[0]
        spectrogram[start1:(start1 + interval), :] = 0

    return spectrogram



class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', eval=False, cfg=None, cfg_flow=None, sample_dur=10, use_video=True, use_flow=True, use_audio=True, dataset='HAC', datapath=''):
        self.base_path = datapath
        self.split = split
        self.interval = 9
        self.sample_dur = sample_dur
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_flow = use_flow


        # build the data pipeline
        if split == 'train':
            if self.use_video:
                train_pipeline = cfg.data.train.pipeline
                self.pipeline = Compose(train_pipeline)
            if self.use_flow:
                train_pipeline_flow = cfg_flow.data.train.pipeline
                self.pipeline_flow = Compose(train_pipeline_flow)
            self.train = True
        else:
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline)
            if self.use_flow:
                val_pipeline_flow = cfg_flow.data.val.pipeline
                self.pipeline_flow = Compose(val_pipeline_flow)
            self.train = False

        if dataset == "HAC":
            self.samples = []
            self.labels = []
            train_file_name = "splits/" + dataset + "_" + split + "_only_cartoon.csv"
            with open(train_file_name) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.samples.append(row[0])
                    self.labels.append(row[1])

        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.dataset = dataset

    def __getitem__(self, index):
        video_path = ''
        if self.dataset == "HAC":
            video_file = self.base_path + 'cartoon/videos/' + self.samples[index]
            #video_file = self.base_path + 'cartoon/video-C/defocus_blur/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/frost/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/brightness/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/pixelate/' + self.samples[index]
            # video_file = self.base_path + 'cartoon/video-C/jpeg_compression/' + self.samples[index]
            #video_file = self.base_path + 'cartoon/video-C/gaussian_noise_5/' + self.samples[index]
        
        if self.use_video:
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


        if self.dataset == "HAC":
            video_file_x = self.base_path + 'cartoon/flow/'+ self.samples[index][:-4] + '_flow_x.mp4'
            video_file_y = self.base_path + 'cartoon/flow/'+ self.samples[index][:-4] + '_flow_y.mp4'
        
        if self.use_flow:
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

        if self.use_audio:
            audio_path = self.base_path + 'cartoon/audio/' + self.samples[index][:-4] + '.wav'
            #audio_path = self.base_path + 'cartoon/audio-C/wind/' + self.samples[index][:-4] + '.wav'
            if self.use_video:
                start_time = frame_inds[0] / 24.0
                end_time = frame_inds[-1] / 24.0
            else:
                start_time = frame_inds_flow[0] / 24.0
                end_time = frame_inds_flow[-1] / 24.0
            samples, samplerate = sf.read(audio_path)
            duration = len(samples) / samplerate

            spectrogram = get_spectrogram_piece(samples,start_time,end_time,duration,samplerate,training=self.train)

        label1 = int(self.labels[index])

        if self.use_video and self.use_flow and self.use_audio:
            return data, flow, spectrogram.astype(np.float32), label1, index
        elif self.use_video and self.use_flow:
            return data, flow, 0, label1, index
        elif self.use_video and self.use_audio:
            return data, 0, spectrogram.astype(np.float32), label1, index
        elif self.use_flow and self.use_audio:
            return 0, flow, spectrogram.astype(np.float32), label1, index


    def __len__(self):
        return len(self.samples)
