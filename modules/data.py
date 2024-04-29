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
    def __init__(self, cfg, path, transform=None, train=False, mode=None):
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
        self.silence = cfg['silence']
        self.size = cfg['train_sz'] if train else cfg['val_sz']
        if mode is None:
            if train:
                self.filenames = load_index(cfg, path, mode="train")
            else:
                self.filenames = load_index(cfg, path, mode="valid")
        else:
            self.filenames = load_index(cfg, path, mode=mode)
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
            except:
                print(f"Audio length is {len(audio)/self.sample_rate} seconds. Skipping {datapath}")
                self.ignore_idx.append(idx)
                return self[idx+1]
        else:
            start = int(2.0 * self.sample_rate) # Arbitrary start point for validation

        audio = audio[start:start+clip_frames]

        if np.max(abs(audio)) < self.silence:
            print("Silence detected. Skipping...")
            return self[idx + 1]
        # spec = np.abs(librosa.stft(audio, 
        #                             n_fft=self.n_fft, 
        #                             win_length=self.win_len,
        #                             hop_length=self.hop_len))

        # # Get rid of extra bin
        # spec = spec[:-1, :]

        # Compute constant-Q spectrogram
        spec = np.abs(librosa.cqt(audio, 
                           sr=self.sample_rate, 
                           hop_length=self.hop_len, 
                           n_bins=self.cfg['n_bins'], 
                           bins_per_octave=36))

        spec_dB = librosa.amplitude_to_db(spec, ref=np.max)

        # Min max normalization
        spec_min = np.min(spec_dB)
        spec_max = np.max(spec_dB)
        spec_dB = (spec_dB - spec_min) / (spec_max - spec_min)
        
        # Pad to 256 frequency bins
        if spec.shape[0] < 256:
            pad = np.zeros((256 - spec.shape[0], spec.shape[1]))
            spec = np.concatenate([spec, pad], axis=0)
            spec_dB = np.concatenate([spec_dB, pad], axis=0)

        # Pad to n_frames
        if spec.shape[1] < self.n_frames:
            pad = np.zeros((spec.shape[0], self.n_frames - spec.shape[1]))
            spec = np.concatenate([spec, pad], axis=1)
            spec_dB = np.concatenate([spec_dB, pad], axis=1)
        elif spec.shape[1] > self.n_frames:
            spec = spec[:, :self.n_frames]
            spec_dB = spec_dB[:, :self.n_frames]

        # Peak-picking
        analyzer = Analyzer(cfg=self.cfg)
        mask, _ = analyzer.find_peaks(sgram=spec, backward=False)
        peaks = mask * spec_dB

        if self.transform is not None:
            spec_dB = self.transform(spec_dB)

        assert spec_dB.shape == (256, self.n_frames), f"Expected shape (256, {self.n_frames}), but got {spec.shape}"
            
        # target = torch.from_numpy(librosa.amplitude_to_db(spec)).unsqueeze(0).float()
        target = torch.from_numpy(spec_dB).unsqueeze(0).float()
        peaks = torch.from_numpy(peaks).unsqueeze(0).float()

        if self.train:
            return peaks, target
        else:
            return peaks, spec_min, spec_max, datapath  

    def __len__(self):
        return len(self.filenames)