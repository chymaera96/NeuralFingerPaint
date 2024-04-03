import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_loss_dis(real, fake):
    return (torch.mean(F.relu(1 - real)) + torch.mean(F.relu(1 + fake))) / 2.0
 
def hinge_loss_gen(fake):
    return -torch.mean(fake)