import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv2 import LpConvert

# AlexNet
@register_model
def lp2a_alexnet_imnet200(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

# VGG
@register_model
def lp2a_vgg11_imnet200(pretrained=False, **kwargs):
    model = models.vgg11(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_vgg13_imnet200(pretrained=False, **kwargs):
    model = models.vgg13(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_vgg16_imnet200(pretrained=False, **kwargs):
    model = models.vgg16(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_vgg19_imnet200(pretrained=False, **kwargs):
    model = models.vgg19(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

# VGG + BN
@register_model
def lp2a_vgg11_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg11_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_vgg13_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg13_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_vgg16_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_vgg19_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg19_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

# ResNet
@register_model
def lp2a_resnet18_imnet200(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_resnet34_imnet200(pretrained=False, **kwargs):
    model = models.resnet34(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_resnet50_imnet200(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_resnet101_imnet200(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_resnet152_imnet200(pretrained=False, **kwargs):
    model = models.resnet152(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

# DenseNet
@register_model
def lp2a_densenet121_imnet200(pretrained=False, **kwargs):
    model = models.densenet121(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_densenet161_imnet200(pretrained=False, **kwargs):
    model = models.densenet161(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_densenet169_imnet200(pretrained=False, **kwargs):
    model = models.densenet169(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_densenet201_imnet200(pretrained=False, **kwargs):
    model = models.densenet201(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

# WideResNet
@register_model
def lp2a_wide_resnet50_2_imnet200(pretrained=False, **kwargs):
    model = models.wide_resnet50_2(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_wide_resnet101_2_imnet200(pretrained=False, **kwargs):
    model = models.wide_resnet101_2(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

# ResNeXt
@register_model
def lp2a_resnext50_32x4d_imnet200(pretrained=False, **kwargs):
    model = models.resnext50_32x4d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_resnext101_32x8d_imnet200(pretrained=False, **kwargs):
    model = models.resnext101_32x8d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_resnext101_64x4d_imnet200(pretrained=False, **kwargs):
    model = models.resnext101_64x4d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

# ConvNeXt
@register_model
def lp2a_convnext_base_imnet200(pretrained=False, **kwargs):
    model = models.convnext_base(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_convnext_large_imnet200(pretrained=False, **kwargs):
    model = models.convnext_large(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_convnext_small_imnet200(pretrained=False, **kwargs):
    model = models.convnext_small(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)

@register_model
def lp2a_convnext_tiny_imnet200(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=1)
