import os
import numpy as np
import argparse
import torch
import gc
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



from util import *
from modules.loss import hinge_loss_dis, hinge_loss_gen  
from modules.data import FPaintDataset
from pix2pixGAN import Generator, Discriminator

# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")


device = torch.device("cuda")


parser = argparse.ArgumentParser(description='Neural Fingerpaint Training')
parser.add_argument('--config', default=None, type=str,
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
        dis_loss.backward()
        dis_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        gen_output = discriminator(input, fake_spec)
        gen_loss = hinge_loss_gen(gen_output)
        gen_loss.backward()
        gen_optimizer.step()

        dis_loss_epoch += dis_loss.item()
        gen_loss_epoch += gen_loss.item()

        if idx % 100 == 0:
            print(
                f"Step[{idx}/{len(train_loader)}] Loss D: {dis_loss.item()}, Loss G: {gen_loss.item()}"
            )

    return dis_loss_epoch, gen_loss_epoch


def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    writer = SummaryWriter(f'runs/{args.ckp}')
    train_dir = cfg['train_dir']
    valid_dir = cfg['val_dir']

    
    # Hyperparameters
    batch_size = cfg['bsz_train']
    learning_rate = cfg['lr']
    num_epochs = cfg['n_epochs']
    model_name = args.ckp
    random_seed = args.seed
    shuffle_dataset = True

    print("Loading dataset...")
    train_dataset = FPaintDataset(cfg=cfg, path=train_dir, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    
    print("Creating new model...")
    generator = Generator(cfg).to(device)
    discriminator = Discriminator(cfg).to(device)

    print(count_parameters(generator, args.encoder))
    print(count_parameters(discriminator, args.encoder))

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
       
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            generator, 
            discriminator, 
            gen_optimizer, 
            dis_optimizer, 
            start_epoch, 
            gen_loss_log, 
            dis_loss_log = load_ckp(args.resume,  generator, discriminator, gen_optimizer, dis_optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    else:
        start_epoch = 0
        dis_loss_log = []
        gen_loss_log = []


    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        dis_loss_epoch, gen_loss_epoch = train(cfg, train_loader, discriminator, generator, dis_optimizer, gen_optimizer)
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
        if epoch % 10 == 0:
            save_ckp(checkpoint, model_name, model_folder, str(epoch))
  
        
if __name__ == '__main__':
    main()