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
    # modified from below
    # https://github.com/ShahabBakht/ventral-dorsal-model/blob/a959ac56650468894aa07a2e95eaf80250922791/RSM/deepModelsAnalysis.py#L592
    # https://github.com/ShahabBakht/ventral-dorsal-model/blob/a959ac56650468894aa07a2e95eaf80250922791/RSM/generate_SSM.py#L124C1-L137C33

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

    def center_activations(feature_dict):
        
        feature_dict_centered = dict()
        for layers, activation_arr in feature_dict.items():
            activation_flat = activation_arr.reshape((activation_arr.shape[0],-1))
            if torch.is_tensor(activation_flat):
                activation_flat = activation_flat.numpy()
                
            activation_mean_percolumn = np.mean(activation_flat,axis=0)
            activation_mean = np.tile(activation_mean_percolumn,(activation_flat.shape[0],1))
            activation_centered = activation_flat - activation_mean
            activation_centered_unflat = activation_centered.reshape((activation_arr.shape))
            feature_dict_centered[layers] = activation_centered_unflat
            
        return feature_dict_centered

    return center_activations(activations)
