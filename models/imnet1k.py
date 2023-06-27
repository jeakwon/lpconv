import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv2 import LpConvert

@register_model
def alexnet_imnet1k(pretrained=False, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/alexnet.py#L65
    "acc@1": 56.522,
    "acc@5": 79.066,
    """
    model = models.alexnet(pretrained=pretrained)
    return model

@register_model
def vgg16_bn_imnet1k(pretrained=False, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/vgg.py#L254
    "acc@1": 73.360,
    "acc@5": 91.516,
    """
    model = models.vgg16_bn(pretrained=pretrained)
    return model

@register_model
def resnet18_imnet1k(pretrained=False, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/resnet.py#L321
    "acc@1": 69.758,
    "acc@5": 89.078,
    """
    model = models.resnet18(pretrained=pretrained)
    return model

@register_model
def resnet34_imnet1k(pretrained=False, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/resnet.py#L343
    "acc@1": 73.314,
    "acc@5": 91.420,
    """
    model = models.resnet34(pretrained=pretrained)
    return model

@register_model
def resnet50_imnet1k(pretrained=False, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/resnet.py#L383
    "acc@1": 80.858,
    "acc@5": 95.434,
    """
    model = models.resnet50(pretrained=pretrained)
    return model

@register_model
def resnext50_32x4d_imnet1k(pretrained=False, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/resnet.py#L494
    "acc@1": 77.618,
    "acc@5": 93.698,
    """
    model = models.resnext50_32x4d(pretrained=pretrained)
    return model

@register_model
def densenet121_imnet1k(pretrained=False, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/densenet.py#L275
    "acc@1": 74.434,
    "acc@5": 91.972,
    """
    model = models.densenet121(pretrained=pretrained)
    return model

@register_model
def convnext_base_imnet1k(pretrained=False, **kwargs):
    """https://github.com/pytorch/vision/blob/c9ac3a5b03731fa17d3934b552f308791314602b/torchvision/models/convnext.py#L257
    "acc@1": 84.062,
    "acc@5": 96.870,
    """
    model = models.convnext_base(pretrained=pretrained)
    return model
