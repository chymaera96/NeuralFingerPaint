import numpy as np
import torch
from scipy import linalg

def eval_fad(cfg, model, data, device):
    model.eval()
    fad = 0
    n = 0
    for idx, (input, target) in enumerate(data):
        if idx % 100 == 0:
            print(f"Step[{idx}/{len(data)}]")
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            fake_spec = model(input)
        fad += compute_fad(fake_spec, target)
        n += 1
    fad /= n

    return fad

def compute_fad(fake, real, eps=1e-6):
    fake = fake.cpu().numpy()
    real = real.cpu().numpy()
    fake = fake.reshape(fake.shape[0], -1)
    real = real.reshape(real.shape[0], -1)
    mu1 = np.mean(fake, axis=0)
    mu2 = np.mean(real, axis=0)
    sigma1 = np.cov(fake, rowvar=False)
    sigma2 = np.cov(real, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    fad = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fad