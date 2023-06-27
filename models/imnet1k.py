import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv2 import LpConvert

@register_model
def alexnet_imnet1k(pretrained=True, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/alexnet.py#L65
    "acc@1": 56.522,
    "acc@5": 79.066,
    """
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/alexnet.pth"
    print(path)
    weights = torch.load(path)
    model = models.alexnet(weights=weights)
    return model

@register_model
def vgg16_bn_imnet1k(pretrained=True, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/vgg.py#L254
    "acc@1": 73.360,
    "acc@5": 91.516,
    """
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/vgg16_bn.pth"
    print(path)
    weights = torch.load(path)
    model = models.vgg16_bn(weights=weights)
    return model

@register_model
def resnet18_imnet1k(pretrained=True, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/resnet.py#L321
    "acc@1": 69.758,
    "acc@5": 89.078,
    """
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/resnet18.pth"
    print(path)
    weights = torch.load(path)
    model = models.resnet18(weights=weights)
    return model

@register_model
def resnet34_imnet1k(pretrained=True, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/resnet.py#L343
    "acc@1": 73.314,
    "acc@5": 91.420,
    """
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/resnet34.pth"
    print(path)
    weights = torch.load(path)
    model = models.resnet34(weights=weights)
    return model

@register_model
def resnet50_imnet1k(pretrained=True, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/resnet.py#L383
    "acc@1": 80.858,
    "acc@5": 95.434,
    """
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/resnet50.pth"
    print(path)
    weights = torch.load(path)
    model = models.resnet50(weights=weights)
    return model

@register_model
def resnext50_32x4d_imnet1k(pretrained=True, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/resnet.py#L494
    "acc@1": 77.618,
    "acc@5": 93.698,
    """
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/resnext50_32x4d.pth"
    print(path)
    weights = torch.load(path)
    model = models.resnext50_32x4d(weights=weights)
    return model

@register_model
def densenet121_imnet1k(pretrained=True, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/densenet.py#L275
    "acc@1": 74.434,
    "acc@5": 91.972,
    """
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/densenet121.pth"
    print(path)
    weights = torch.load(path)
    model = models.densenet121(weights=weights)
    return model

@register_model
def convnext_base_imnet1k(pretrained=True, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/convnext.py#L257
    "acc@1": 84.062,
    "acc@5": 96.870,
    """
    path = "/mnt/lustre/ibs/jeakwon/pretrained_models/convnext_base.pth"
    print(path)
    weights = torch.load(path)
    model = models.convnext_base(weights=weights)
    return model
