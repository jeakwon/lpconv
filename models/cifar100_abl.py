"""Lp-Conv p=2 in CIFAR-100"""
import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert
import copy

alexnet_conv_layers = {1:0,2:3,3:6,4:8,5:10}
# AlexNet
@register_model
def abl2345_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [2,3,4,5]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl1345_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1,3,4,5]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl1245_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1,2,4,5]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl1235_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1,2,3,5]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl1234_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1,2,3,4]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel


@register_model
def abl1_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl2_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [2]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl2_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [2]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl3_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [3]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl4_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [4]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl5_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [5]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl145_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1, 4, 5]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

@register_model
def abl123_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1, 2, 3]
    for target in ablation_targets:
        layer = alexnet_conv_layers[target]
        lpmodel.features[layer] = model.features[layer]
    return lpmodel

resnet18_conv_layers = {0:'conv1', 1:'layer1',2:'layer2',3:'layer3',4:'layer4'}
# ResNet18
@register_model
def abl1234_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1, 2, 3, 4]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel
    
def abl0234_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [0, 2, 3, 4]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel
    
def abl0134_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [0, 1, 3, 4]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel

def abl0124_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [0, 1, 2, 4]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel

def abl0123_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [0, 1, 2, 3]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel

def abl0_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [0]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel

def abl1_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [1]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel

def abl2_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [2]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel

def abl3_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [3]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel

def abl4_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32

    lpmodel = LpConvert( copy.deepcopy(model), log2p=1)
    ablation_targets = [4]
    for target in ablation_targets:
        layer = resnet18_conv_layers[target]
        setattr(lpmodel, layer, getattr(model, layer))
    return lpmodel
