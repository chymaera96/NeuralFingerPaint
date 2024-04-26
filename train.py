import os
import numpy as np
import argparse
import torch
import librosa
import gc
import torch.nn.functional as F
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



from util import *
from modules.loss import hinge_loss_dis, hinge_loss_gen  
from modules.data import FPaintDataset
from src.pix2pixGAN import Generator, Discriminator

# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")


device = torch.device("cuda")


parser = argparse.ArgumentParser(description='Neural Fingerpaint Training')
parser.add_argument('--config', default='config/default.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--train_dir', default=None, type=str, metavar='PATH',
                    help='path to training data')
parser.add_argument('--val_dir', default=None, type=str, metavar='PATH',
                    help='path to validation data')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--ckp', default='test', type=str,
                    help='checkpoint_name')



def train(cfg, train_loader, discriminator, generator, dis_optimizer, gen_optimizer):
    discriminator.train()
    generator.train()
    dis_loss_epoch = 0
    gen_loss_epoch = 0

    for idx, (input, target) in enumerate(train_loader):

        dis_optimizer.zero_grad()
        input = input.to(device)
        target = target.to(device)

        # Real spectrogram
        dis_real_output = discriminator(input, target)  # target is spectrogram

        # Fake spectrogram
        noise = torch.randn(input.size(), device=device)
        fake_spec = generator(torch.cat([input, noise], dim=1))
        dis_fake_output = discriminator(input, fake_spec.detach())

        dis_loss = hinge_loss_dis(dis_real_output, dis_fake_output)
        # gradient_penalty = compute_gradient_penalty(discriminator, input, target, fake_spec)
        # dis_loss += cfg['lambda'] * gradient_penalty
        dis_loss.backward()
        dis_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        noise = torch.randn(input.size(), device=device)  # Generate new noise
        fake_spec = generator(torch.cat([input, noise], dim=1))
        gen_output = discriminator(input, fake_spec)
        gen_loss = hinge_loss_gen(gen_output)
        gen_loss.backward()
        gen_optimizer.step()

        dis_loss_epoch += dis_loss.item()
        gen_loss_epoch += gen_loss.item()

        if idx % 50 == 0:
            print(
                f"Step[{idx}/{len(train_loader)}] Loss D: {dis_loss.item()}, Loss G: {gen_loss.item()}"
            )

    return dis_loss_epoch, gen_loss_epoch


def save_generated_samples(cfg, generator, val_loader, ckp, epoch, save_path='data/generated_samples'):
    generator.eval()  # Set the generator to evaluation mode
    if not os.path.exists(f'{save_path}'):
        os.mkdir(f'{save_path}')
    # Load samples from val_loader
    input, path = next(iter(val_loader))
    input = input.to(device)

    with torch.no_grad():
        # Generate fake spectrograms
        noise = torch.randn(input.size(), device=device)
        fake_specs = generator(torch.cat([input, noise], dim=1))

    # Reconstruct audio from spectrograms using Griffin-Lim algorithm
    reconstructed_audios = []
    for fake_spec in fake_specs:
        fake_spec = fake_spec.squeeze().detach().cpu().numpy()
        assert len(fake_spec.shape) == 2, f"Expected 2D spectrogram, but got {fake_spec.shape}"
        if fake_spec.shape[0] == cfg['n_fft'] // 2 + 1:
            fake_spec = fake_spec[:-1, :]
        fake_audio = librosa.griffinlim(fake_spec, hop_length=cfg['hop_len'], win_length=cfg['win_len'])
        reconstructed_audios.append(fake_audio)

    # Save generated audio files
    for i, audio in enumerate(reconstructed_audios):
        audio_path = f'{save_path}/audio_model_{ckp}_epoch_{epoch}_sample_{path[0]}.wav'
        sf.write(audio_path, audio, cfg['fs'])
        # save spectrogram
        spec_path = f'{save_path}/spec_model_{ckp}_epoch_{epoch}_sample_{path[0]}.png'
        # Save librosa specshow spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(fake_specs[i].squeeze().detach().cpu().numpy(), ref=np.max),
                                    y_axis='log', x_axis='time', sr=cfg['fs'], hop_length=cfg['hop_len'])
        plt.colorbar(format='%+2.0f dB')
        plt.title('Generated Spectrogram')
        plt.tight_layout()
        plt.savefig(spec_path)
    generator.train()  # Set the generator back to training mode



def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    writer = SummaryWriter(f'runs/{args.ckp}')
    train_dir = cfg['train_dir']
    valid_dir = cfg['val_dir']

    
    # Hyperparameters
    batch_size = cfg['bsz_train']
    num_epochs = cfg['n_epochs']
    model_name = args.ckp
    random_seed = args.seed
    shuffle_dataset = True

    print("Loading dataset...")
    train_dataset = FPaintDataset(cfg=cfg, path=train_dir, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)
    
    val_dataset = FPaintDataset(cfg=cfg, path=valid_dir, train=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, drop_last=True)

    
    print("Creating new model...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    print(count_parameters(generator, 'generator'))
    print(count_parameters(discriminator, 'discriminator'))

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg['g_lr'], betas=(0.5, 0.9))
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg['d_lr'], betas=(0.5, 0.9))
       
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            generator, discriminator, gen_optimizer, dis_optimizer, start_epoch, gen_loss_log, dis_loss_log = load_ckp(args.resume, 
                                                                                                                       generator, 
                                                                                                                       discriminator, 
                                                                                                                       gen_optimizer, 
                                                                                                                       dis_optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    else:
        start_epoch = 0
        dis_loss_log = []
        gen_loss_log = []


    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        dis_loss_epoch, gen_loss_epoch = train(cfg, train_loader, discriminator, generator, dis_optimizer, gen_optimizer)
        if epoch % 5 == 0:
            save_generated_samples(cfg, generator, val_loader, ckp=args.ckp, epoch=epoch)
        writer.add_scalar("Discriminator Loss", dis_loss_epoch, epoch)
        writer.add_scalar("Generator Loss", gen_loss_epoch, epoch)
        dis_loss_log.append(dis_loss_epoch)
        gen_loss_log.append(gen_loss_epoch)

        checkpoint = {
            'epoch': epoch,
            'dis_loss': dis_loss_log,
            'gen_loss': gen_loss_log,
            'dis_state_dict': discriminator.state_dict(),
            'gen_state_dict': generator.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict()
        }
        save_ckp(checkpoint, model_name, model_folder, 'current')
        if epoch % 5 == 0:
            save_ckp(checkpoint, model_name, model_folder, str(epoch))
  
        
if __name__ == '__main__':
    main()