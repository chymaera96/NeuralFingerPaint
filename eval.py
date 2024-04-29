import numpy as np
import torch
import argparse
from scipy import linalg

from src.pix2pixGAN import Generator
from util import load_config, load_ckp
from modules.data import FPaintDataset


parser = argparse.ArgumentParser(description='Neural Fingerpaint FAD Evaluation')
parser.add_argument('--config', default='config/default.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--ckp', default='test', type=str,
                    help='path to checkpoint')

def eval_fad(cfg, model, data, device):
    model.eval()
    fad = 0
    n = 0
    for idx, (input, target) in enumerate(data):
        if idx % 10 == 0:
            print(f"Step[{idx}/{len(data)}]")
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            noise = torch.randn(input.size(), device=device)
            fake_spec = model(torch.cat([input, noise], dim=1))
        fad += compute_fad(fake_spec, target)
        n += 1
    fad /= n

    return fad

def compute_fad(fake, real, eps=1e-6):
    fake = fake.view(fake.shape[0], -1)
    real = real.view(real.shape[0], -1)
    mu1 = torch.mean(fake, dim=0)
    mu2 = torch.mean(real, dim=0)
    sigma1 = torch_cov(fake, rowvar=False, compute_full=False)
    sigma2 = torch_cov(real, rowvar=False, compute_full=False)
    diff = mu1 - mu2
    tr_covmean = sqrtm(sigma1 * sigma2)
    if not torch.isfinite(tr_covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = torch.eye(sigma1.shape[0]).to(fake.device) * eps
        tr_covmean = sqrtm((sigma1 + offset) * (sigma2 + offset))
    fad = diff.dot(diff) + sigma1 + sigma2 - 2 * tr_covmean
    return fad


def torch_cov(m, y=None, rowvar=False, compute_full=True):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=0)
    x = m - m_exp
    if compute_full:
        cov = 1 / (x.size(0) - 1) * x.t().mm(x)
    else:
        cov = torch.mean(x ** 2, dim=0) - m_exp ** 2
    return cov

def sqrtm(matrix):
    # This function is not available in PyTorch, so we need to implement it ourselves
    # Here we use the method described in https://github.com/msubhransu/matrix-sqrt
    _, s, v = torch.svd(matrix)
    return v.mm(torch.diag(s.sqrt())).mm(v.t())

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_dir = cfg['train_dir']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading dataset...")
    val_dataset = FPaintDataset(cfg=cfg, path=valid_dir, train=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    # Load model
    model = Generator()
    model = model.to(device)
    checkpoint = torch.load(f'{args.ckp}', map_location=device)
    model.load_state_dict(checkpoint['gen_state_dict'])

    # Evaluate
    fad = eval_fad(cfg, model, val_loader, device)
    print(f"FAD: {fad}")


if __name__ == '__main__':
    main()