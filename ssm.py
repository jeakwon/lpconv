import torch
import models 

def load_checkpoint(path_to_checkpoint):
    checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
    args = checkpoint['args']
    model = getattr(models, args.model)(num_classes=args.nb_classes)
    model.load_state_dict(checkpoint['model'])
    return model

def load_natural_scences(path_to_data):
    return torch.load(path_to_data)