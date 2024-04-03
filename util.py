import os
import torch
import numpy as np
import json
import glob
import soundfile as sf
import shutil
import yaml
from prettytable import PrettyTable

def load_index(data_dir, ext=['wav','mp3'], max_len=10000, inplace=False):
    dataset = {}

    if data_dir.endswith('.json'):
        print(f"=>Loading indices from index file {data_dir}")
        with open(data_dir, 'r') as fp:
            dataset = json.load(fp)
        return dataset
    
    print(f"=>Loading indices from {data_dir}")
    if inplace:
        json_path = os.path.join(data_dir, data_dir.split('/')[-1] + ".json")
    else:
        json_path = os.path.join('data', data_dir.split('/')[-1] + ".json")
    if not os.path.exists(json_path):
        idx = 0
        for fpath in glob.iglob(os.path.join(data_dir,'**/*.*'), recursive=True):
            if len(dataset) >= max_len:
                break
            if fpath.split('.')[-1] in ext: 
                dataset[str(idx)] = fpath
                idx += 1

        with open(json_path, 'w') as fp:
            json.dump(dataset, fp)
    
    else:
        print(f"Index exists. Loading indices from {json_path}")
        with open(json_path, 'r') as fp:
            dataset = json.load(fp)

    assert len(dataset) > 0
    return dataset

def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + torch.quantile(y,q=q))

def load_ckp(checkpoint_fpath, generator, discriminator, gen_optimizer, dis_optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

def load_checkpoint(checkpoint_path, generator, discriminator, gen_optimizer, dis_optimizer):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
    dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
    epoch = checkpoint['epoch']
    return generator, discriminator, gen_optimizer, dis_optimizer, epoch



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
