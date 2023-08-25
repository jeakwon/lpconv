import torch
from .. import models

def load_checkpoint(path_to_checkpoint):
    checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
    model = getattr(models, checkpoint['args'].model).load_state_dict(checkpoint['model'])
    return model