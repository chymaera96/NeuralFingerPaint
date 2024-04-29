import numpy as np
import torch
import argparse
from scipy import linalg
from torcheval.metrics import FrechetInceptionDistance as FID

from src.pix2pixGAN import Generator
from util import load_config, load_ckp
from modules.data import FPaintDataset


parser = argparse.ArgumentParser(description='Neural Fingerpaint FAD Evaluation')
parser.add_argument('--config', default='config/default.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--ckp', default='test', type=str,
                    help='path to checkpoint')
parser.add_argument('--density', default=None, type=float,    
                    help='Density of the mask')
parser.add_argument('--metric', default='FID', type=str,
                    help='Metric to evaluate: FID or FAD')


def eval_fd(cfg, model, data, device, measure='FID'):
    model.eval()
    fad = 0
    n = 0
    fid = FID(device=device)  # Initialize FID calculator
    for idx, (input, target) in enumerate(data):
        if idx % 10 == 0:
            print(f"Step[{idx}/{len(data)}]")
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            noise = torch.randn(input.size(), device=device)
            fake_spec = model(torch.cat([input, noise], dim=1))
            if measure == 'FID':
                # Copy channel to 3 channels
                fake_spec = torch.cat([fake_spec, fake_spec, fake_spec], dim=1)
                target = torch.cat([target, target, target], dim=1)
                fid.update(True, target)
                fid.update(False, fake_spec)
                # fid.update(fake_spec, target)  # Update FID with new samples
            else:
                fad += compute_fad(fake_spec, target)
                n += 1
    if measure == 'FID':
        fad = fid.compute()  # Compute FID
    else:
        fad /= n

    return fad


def compute_fad(fake, real, eps=1e-6):
    fake = fake.view(fake.shape[0], -1)
    real = real.view(real.shape[0], -1)
    mu1 = torch.mean(fake, dim=0)
    mu2 = torch.mean(real, dim=0)
    sigma1 = torch_cov(fake, rowvar=False)
    sigma2 = torch_cov(real, rowvar=False)
    diff = mu1 - mu2
    tr_covmean = torch.trace(sqrtm(sigma1.mm(sigma2) + eps * torch.eye(sigma1.size(0)).to(fake.device)))
    fad = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return fad

def torch_cov(m, y=None, rowvar=False, chunk_size=100):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    if not rowvar:
        m = m.t()
    m_exp = torch.mean(m, dim=0)
    x = m - m_exp
    cov = torch.zeros((x.size(1), x.size(1)), device=x.device)
    for i in range(0, x.size(0), chunk_size):
        batch = x[i:i+chunk_size]
        cov += 1 / (x.size(0) - 1) * batch.t().mm(batch)
    return cov

def sqrtm(matrix):
    # This function is not available in PyTorch, so we need to implement it ourselves
    # Here we use the method described in https://github.com/msubhransu/matrix-sqrt
    _, s, v = torch.svd(matrix)
    return v.mm(torch.diag(s.sqrt())).mm(v.t())

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.density is not None:
        cfg['density'] = args.density
    val_dir = cfg['val_dir']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading dataset...")
    val_dataset = FPaintDataset(cfg=cfg, path=val_dir, train=True, mode="valid")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    # Load model
    model = Generator()
    model = model.to(device)
    checkpoint = torch.load(f'{args.ckp}', map_location=device)
    model.load_state_dict(checkpoint['gen_state_dict'])

    # Evaluate
    fad = eval_fd(cfg, model, val_loader, device, measure=args.metric)
    print(f"FAD: {fad}")


if __name__ == '__main__':
    main()