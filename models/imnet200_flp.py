"""frozen p Lp-Conv in TinyImageNet"""
import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert

# AlexNet
@register_model
def flp1_alexnet_imnet200(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1, set_requires_grad=dict(log2p=False))

@register_model
def flp2_alexnet_imnet200(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=2, set_requires_grad=dict(log2p=False))

@register_model
def flp3_alexnet_imnet200(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=3, set_requires_grad=dict(log2p=False))

@register_model
def flp4_alexnet_imnet200(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4, set_requires_grad=dict(log2p=False))

# VGG
@register_model
def flp1_vgg16_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1, set_requires_grad=dict(log2p=False))

@register_model
def flp2_vgg16_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=2, set_requires_grad=dict(log2p=False))

@register_model
def flp3_vgg16_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=3, set_requires_grad=dict(log2p=False))

@register_model
def flp4_vgg16_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4, set_requires_grad=dict(log2p=False))
