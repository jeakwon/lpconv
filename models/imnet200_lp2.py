"""Lp-Conv p=16 in TinyImageNet"""
import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert

# AlexNet
@register_model
def lp2_alexnet_imnet200(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

# VGG
@register_model
def lp2_vgg11_imnet200(pretrained=False, **kwargs):
    model = models.vgg11(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_vgg13_imnet200(pretrained=False, **kwargs):
    model = models.vgg13(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_vgg16_imnet200(pretrained=False, **kwargs):
    model = models.vgg16(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_vgg19_imnet200(pretrained=False, **kwargs):
    model = models.vgg19(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

# VGG + BN
@register_model
def lp2_vgg11_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg11_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_vgg13_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg13_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_vgg16_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_vgg19_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg19_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

# ResNet
@register_model
def lp2_resnet18_imnet200(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_resnet34_imnet200(pretrained=False, **kwargs):
    model = models.resnet34(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_resnet50_imnet200(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_resnet101_imnet200(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_resnet152_imnet200(pretrained=False, **kwargs):
    model = models.resnet152(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

# DenseNet
@register_model
def lp2_densenet121_imnet200(pretrained=False, **kwargs):
    model = models.densenet121(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_densenet161_imnet200(pretrained=False, **kwargs):
    model = models.densenet161(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_densenet169_imnet200(pretrained=False, **kwargs):
    model = models.densenet169(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_densenet201_imnet200(pretrained=False, **kwargs):
    model = models.densenet201(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

# WideResNet
@register_model
def lp2_wide_resnet50_2_imnet200(pretrained=False, **kwargs):
    model = models.wide_resnet50_2(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_wide_resnet101_2_imnet200(pretrained=False, **kwargs):
    model = models.wide_resnet101_2(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

# ResNeXt
@register_model
def lp2_resnext50_32x4d_imnet200(pretrained=False, **kwargs):
    model = models.resnext50_32x4d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_resnext101_32x8d_imnet200(pretrained=False, **kwargs):
    model = models.resnext101_32x8d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_resnext101_64x4d_imnet200(pretrained=False, **kwargs):
    model = models.resnext101_64x4d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

# ConvNeXt
@register_model
def lp2_convnext_base_imnet200(pretrained=False, **kwargs):
    model = models.convnext_base(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_convnext_large_imnet200(pretrained=False, **kwargs):
    model = models.convnext_large(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_convnext_small_imnet200(pretrained=False, **kwargs):
    model = models.convnext_small(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)

@register_model
def lp2_convnext_tiny_imnet200(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=4)
