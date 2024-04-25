import os
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import librosa
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from util import load_index, qtile_normalize
from modules.peak_extractor import Analyzer

class FPaintDataset(Dataset):
    def __init__(self, cfg, path, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.cfg = cfg
        self.norm = cfg['norm']
        self.sample_rate = cfg['fs']
        self.dur = cfg['dur']
        self.n_fft = cfg['n_fft']
        self.hop_len = cfg['hop_len']
        self.win_len = cfg['win_len']
        self.n_frames = cfg['n_frames']
        self.size = cfg['train_sz'] if train else cfg['val_sz']
        if train:
            self.filenames = load_index(cfg, path, mode="train")
        else:
            self.filenames = load_index(cfg, path, mode="valid")
        self.ignore_idx = []

    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx+1]
        
        datapath = self.filenames[str(idx)]
        try:
            audio, sr = librosa.load(datapath, sr=self.sample_rate, mono=True)
            if self.norm:
                audio = qtile_normalize(audio, self.norm)
        except Exception as e:
            print(f"Error loading {datapath}: {e}")
            self.ignore_idx.append(idx)
            return self[idx+1]
        
        clip_frames = int(self.dur * self.sample_rate)

        if self.train:
            try:
                start = np.random.randint(0, len(audio) - clip_frames)
                audio = audio[start:start+clip_frames]
            except:
                print(f"Audio length is {len(audio)/self.sample_rate} seconds. Skipping {datapath}")
                self.ignore_idx.append(idx)
                return self[idx+1]

        spec = np.abs(librosa.stft(audio, 
                                    n_fft=self.n_fft, 
                                    win_length=self.win_len,
                                    hop_length=self.hop_len))

        # Get rid of extra bin
        spec = spec[:-1, :]
        # Pad to n_frames
        if spec.shape[1] < self.n_frames:
            pad = np.zeros((spec.shape[0], self.n_frames - spec.shape[1]))
            spec = np.concatenate([spec, pad], axis=1)
        elif spec.shape[1] > self.n_frames:
            spec = spec[:, :self.n_frames]

        # Peak-picking
        analyzer = Analyzer(cfg=self.cfg)
        peaks, _ = analyzer.find_peaks(sgram=spec, backward=False)

        if self.transform is not None:
            spec = self.transform(spec)
            
        target = torch.from_numpy(librosa.amplitude_to_db(spec)).unsqueeze(0).float()
        peaks = torch.from_numpy(peaks).unsqueeze(0).float()
        return peaks, target
    

    def __len__(self):
        return len(self.filenames)