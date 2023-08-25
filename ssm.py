import torch
import torch.nn as nn
import models 
from models.lpconv2 import LpConv2d

from functools import partial
import collections

def load_checkpoint(path_to_checkpoint):
    checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
    args = checkpoint['args']
    model = getattr(models, args.model)(num_classes=args.nb_classes)
    model.load_state_dict(checkpoint['model'])
    return model

def load_natural_scenes(path_to_data):
    return torch.load(path_to_data)


def get_activations(data, model):

    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    for name, m in model.named_modules():
        if type(m)==nn.Conv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

        elif type(m)==LpConv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

    # forward pass through the full dataset
    out = model(data)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(activation, 0) for name, activation in activations.items()}

    for name in activations.keys():
        activations[name] = activations[name].detach()

    return activations