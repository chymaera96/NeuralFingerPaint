import torch
from util import qtile_normalize
import numpy as np
import librosa
from modules.peak_extractor import Analyzer

# Given, model and input audio, compute the output audio
def generate_audio(cfg, model, audio, device='cuda'):
    model.eval()

    # audio = qtile_normalize(audio, cfg['norm'])
    clip_frames = int(cfg['dur'] * cfg['fs'])
    audio = audio[:clip_frames]
    spec = np.abs(librosa.cqt(audio, 
                        sr=cfg['fs'], 
                        hop_length=cfg['hop_len'], 
                        n_bins=cfg['n_bins'], 
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
    if spec.shape[1] < cfg['n_frames']:
        pad = np.zeros((spec.shape[0], cfg['n_frames']  - spec.shape[1]))
        spec = np.concatenate([spec, pad], axis=1)
        spec_dB = np.concatenate([spec_dB, pad], axis=1)
    elif spec.shape[1] > cfg['n_frames']:
        spec = spec[:, :cfg['n_frames']]
        spec_dB = spec_dB[:, :cfg['n_frames']]

    # Peak-picking
    analyzer = Analyzer(cfg=cfg)
    mask, _ = analyzer.find_peaks(sgram=spec, backward=False)
    peaks = mask * spec_dB

    target = torch.from_numpy(spec_dB).unsqueeze(0).unsqueeze(0).float()
    peaks = torch.from_numpy(peaks).unsqueeze(0).unsqueeze(0).float()

    peaks = peaks.to(device)
    target = target.to(device)
    noise = torch.randn(peaks.size(), device=device)

    with torch.no_grad():
        output = model(torch.cat([peaks, noise], dim=1))
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        output = output[:252, :]
        output = output * (spec_max - spec_min) + spec_min
        output = librosa.db_to_amplitude(output)
        output_audio = librosa.griffinlim_cqt(output, sr=cfg['fs'], hop_length=cfg['hop_len'], bins_per_octave=36)

    real_spec = librosa.amplitude_to_db(spec, ref=np.max)[:252, :]

    return output_audio, real_spec, output 
