import os
import torch
import numpy as np
import json
import glob
import soundfile as sf
import shutil
import yaml
from prettytable import PrettyTable

def load_index(cfg, data_dir, ext=['wav','mp3'], mode="train"):

    if not os.path.exists('data'):
        os.mkdir('data')
    if data_dir.endswith('.json'):
        print(f"=>Loading indices from index file {data_dir}")
        with open(data_dir, 'r') as fp:
            dataset = json.load(fp)
        return dataset
    
    print(f"=>Loading indices from {data_dir}")
    train_json_path = os.path.join('data', data_dir.split('/')[-1] + "_train.json")
    valid_json_path = os.path.join('data', data_dir.split('/')[-1] + "_valid.json")

    if mode == "train" and os.path.exists(train_json_path):
        print(f"Train index exists. Loading indices from {train_json_path}")
        with open(train_json_path, 'r') as fp:
            train = json.load(fp)
    
    elif mode != "train" and os.path.exists(valid_json_path):
        print(f"Valid index exists. Loading indices from {valid_json_path}")
        with open(valid_json_path, 'r') as fp:
            valid = json.load(fp)

    else:
        print(f"Creating new index files {train_json_path} and {valid_json_path}")
        train_len = cfg['train_sz']
        valid_len = cfg['val_sz']
        idx = 0
        train = {}
        for fpath in glob.iglob(os.path.join(data_dir,'**/*.*'), recursive=True):
            if len(train) >= train_len:
                break
            if fpath.split('.')[-1] in ext: 
                train[str(idx)] = fpath
                idx += 1

        with open(train_json_path, 'w') as fp:
            json.dump(train, fp)

        idx = 0
        valid = {}
        for fpath in glob.iglob(os.path.join(data_dir,'**/*.*'), recursive=True):
            if len(valid) >= valid_len:
                break
            if fpath.split('.')[-1] in ext and fpath not in list(train.values()): 
                valid[str(idx)] = fpath
                idx += 1

        with open(valid_json_path, 'w') as fp:
            json.dump(valid, fp)

    if mode == "train":
        dataset = train
    else:
        dataset = valid

    return dataset

def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + np.quantile(y,q=q))


def load_ckp(checkpoint_path, generator, discriminator, gen_optimizer, dis_optimizer):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['gen_state_dict'])
    discriminator.load_state_dict(checkpoint['dis_state_dict'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
    dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
    epoch = checkpoint['epoch']
    gen_loss = checkpoint['gen_loss']
    dis_loss = checkpoint['dis_loss']
    return generator, discriminator, gen_optimizer, dis_optimizer, epoch, gen_loss, dis_loss



def save_ckp(state,model_name,model_folder,text):
    if not os.path.exists(model_folder): 
        print("Creating checkpoint directory...")
        os.mkdir(model_folder)
    torch.save(state, "{}/model_{}_{}.pth".format(model_folder, model_name, text))

def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    return config

def count_parameters(model, encoder):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    # Write table in text file
    with open(f'model_summary_{encoder}.txt', 'w') as f:
        f.write(str(table))
    return total_params


def compute_r1_penalty(real_output, real_images, device):
    real_images.requires_grad_(True)
    real_grad_out = torch.ones_like(real_output, device=device)
    real_grad = torch.autograd.grad(
        outputs=real_output,
        inputs=real_images,
        grad_outputs=real_grad_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    r1_penalty = torch.sum(real_grad.pow(2), dim=[1, 2, 3])
    return r1_penalty

def main():
    
    path = '/import/c4dm-datasets/PiJAMA'
    # train_json_path = os.path.join('data', path.split('/')[-1] + "_train.json")
    # Recursively count all wav or mp3 files in the directory
    idx = 0
    for fpath in glob.iglob(os.path.join(path,'**/*.*'), recursive=True):
        if fpath.split('.')[-1] in ['wav','mp3']: 
            idx += 1

    print(f"Total files: {idx}")


if __name__ == "__main__":
    main()