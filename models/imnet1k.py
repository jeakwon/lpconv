import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert

@register_model
def alexnet_imnet1k(pretrained=True, **kwargs):
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/alexnet.pth"
    weights = torch.load(path)
    model = models.alexnet(weights=weights)
    return model

@register_model
def vgg16_bn_imnet1k(pretrained=True, **kwargs):
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/vgg16_bn.pth"
    weights = torch.load(path)
    model = models.vgg16_bn(weights=weights)
    return model

@register_model
def resnet18_imnet1k(pretrained=True, **kwargs):
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/resnet18.pth"
    weights = torch.load(path)
    model = models.resnet18(weights=weights)
    return model

@register_model
def resnet34_imnet1k(pretrained=True, **kwargs):
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/resnet34.pth"
    weights = torch.load(path)
    model = models.resnet34(weights=weights)
    return model

@register_model
def resnet50_imnet1k(pretrained=True, **kwargs):
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/resnet50.pth"
    weights = torch.load(path)
    model = models.resnet50(weights=weights)
    return model

@register_model
def densenet121_imnet1k(pretrained=True, **kwargs):
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/densenet121.pth"
    weights = torch.load(path)
    model = models.densenet121(weights=weights)
    return model

@register_model
def convnext_base_imnet1k(pretrained=True, **kwargs):
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/convnext_base.pth"
    weights = torch.load(path)
    model = models.convnext_base(weights=weights)
    return model
