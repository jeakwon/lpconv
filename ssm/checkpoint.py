import torch
from torchvision import models

def load_checkpoint(path_to_checkpoint):
    return torch.load(path_to_checkpoint, map_location='cpu')