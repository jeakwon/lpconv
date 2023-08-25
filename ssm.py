import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from functools import partial
import collections

import models 
from models.lpconv2 import LpConv2d

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

def sim_pearson(X):
    # https://github.com/ShahabBakht/ventral-dorsal-model/blob/a959ac56650468894aa07a2e95eaf80250922791/RSM/generate_SSM.py#L70
    # X is [dim, samples]
    dX = (X.T - np.mean(X.T, axis=0)).T
    sigma = np.sqrt(np.mean(dX**2, axis=1)) + 1e-7

    cor = np.dot(dX, dX.T)/(dX.shape[1]*sigma)
    cor = (cor.T/sigma).T

    return cor

def compute_similarity_matrices(feature_dict, layers=None):
    # https://github.com/ShahabBakht/ventral-dorsal-model/blob/a959ac56650468894aa07a2e95eaf80250922791/RSM/generate_SSM.py#L30

	'''
	feature_dict: a dictionary containing layer activations as numpy arrays
	layers: list of model layers for which features are to be generated

	Output: a dictionary containing layer activation similarity matrices as numpy arrays
	'''


	similarity_mat_dict = {}
	if layers is not None:
		for layer in layers:
			try:
				activation_arr = feature_dict[layer]
				activations_flattened = activation_arr.reshape((activation_arr.shape[0],-1))
				similarity_mat_dict[layer] = sim_pearson(activations_flattened) #np.corrcoef(activations_flattened)
			except Exception as e:
				print(layer)
				raise e
	else:
		for layer,activation_arr in feature_dict.items():
			try:
				activations_flattened = activation_arr.reshape((activation_arr.shape[0],-1))
				similarity_mat_dict[layer] = sim_pearson(activations_flattened) #np.corrcoef(activations_flattened)
			except Exception as e:
				print(layer,activation_arr.shape)
				raise e

	return similarity_mat_dict

def get_model_RSM(path_to_checkpoint, path_to_data):
    model = load_checkpoint(path_to_checkpoint)
    data = load_natural_scenes(path_to_data)
    activations = get_activations(data, model)
    model_RSM = compute_similarity_matrices(activations)
    return model_RSM

def load_brain_RSM(path):
    return torch.load(path)

def load_brain_RSM_noise_ceiling(path):
    noise_ceiling = torch.load(path)
    return pd.DataFrame(noise_ceiling)

def compute_ssm(similarity1, similarity2, num_shuffles=None, num_folds=None):
    # https://github.com/ShahabBakht/ventral-dorsal-model/blob/a959ac56650468894aa07a2e95eaf80250922791/RSM/generate_SSM.py#L96C1-L121C11
    '''
	similarity1: first similarity matrix as a numpy array of size n X n
	similarity2: second similarity matrix as a numpy array of size n X n
	num_shuffles: Number of shuffles to perform to generate a distribution of SSM values
	num_folds: Number of folds to split stimuli set into
    
	Output: the spearman rank correlation of the similarity matrices
    '''
    if num_shuffles is not None:
        raise NotImplementedError()
        
    if num_folds is not None:
	    raise NotImplementedError()
    
    try:
        from scipy.stats import spearmanr
        from scipy.stats import kendalltau
        lowertri_idx = np.tril_indices(similarity1.shape[0],k=-1)
        similarity1_lowertri = similarity1[lowertri_idx]
        similarity2_lowertri = similarity2[lowertri_idx]
        r,_ = kendalltau(similarity1_lowertri,similarity2_lowertri)
        return r
    except:
	    print("Error in calculating spearman correlation")
	    raise

def compute_reps(model_RSM, brain_RSM):
    rows = []
    for layer_name, layer_RSM in model_RSM.items():
        for struct_name, struct_RSM in brain_RSM.items():
            for session in range(struct_RSM.shape[2]):
                ssm = compute_ssm(layer_RSM, struct_RSM[:, :, session])
                row = dict(layer=layer_name, struct=struct_name, session=session, ssm=ssm)
                rows.append(row)

    df = pd.DataFrame(rows)
    return df

def noise_corrected_ssm(path_to_checkpoint, path_to_data, path_to_brain_rsm, path_to_brain_RSM_noise_ceiling):
    model_RSM = get_model_RSM(path_to_checkpoint, path_to_data)
    brain_RSM = load_brain_RSM(path_to_brain_rsm_noise)
    
    r = compute_reps(model_RSM, brain_RSM)

    noise_ceiling = load_brain_RSM_noise_ceiling(path_to_brain_RSM_noise_ceiling)

    return r, noise_ceiling