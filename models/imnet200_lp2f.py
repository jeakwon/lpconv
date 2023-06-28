import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv2 import LpConvert

# AlexNet
@register_model
def lp2f_alexnet_imnet200(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# VGG
@register_model
def lp2f_vgg11_imnet200(pretrained=False, **kwargs):
    model = models.vgg11(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg13_imnet200(pretrained=False, **kwargs):
    model = models.vgg13(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg16_imnet200(pretrained=False, **kwargs):
    model = models.vgg16(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg19_imnet200(pretrained=False, **kwargs):
    model = models.vgg19(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# VGG + BN
@register_model
def lp2f_vgg11_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg11_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg13_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg13_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg16_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg19_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg19_bn(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# ResNet
@register_model
def lp2f_resnet18_imnet200(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnet34_imnet200(pretrained=False, **kwargs):
    model = models.resnet34(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnet50_imnet200(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnet101_imnet200(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnet152_imnet200(pretrained=False, **kwargs):
    model = models.resnet152(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# DenseNet
@register_model
def lp2f_densenet121_imnet200(pretrained=False, **kwargs):
    model = models.densenet121(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_densenet161_imnet200(pretrained=False, **kwargs):
    model = models.densenet161(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_densenet169_imnet200(pretrained=False, **kwargs):
    model = models.densenet169(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_densenet201_imnet200(pretrained=False, **kwargs):
    model = models.densenet201(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# WideResNet
@register_model
def lp2f_wide_resnet50_2_imnet200(pretrained=False, **kwargs):
    model = models.wide_resnet50_2(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_wide_resnet101_2_imnet200(pretrained=False, **kwargs):
    model = models.wide_resnet101_2(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# ResNeXt
@register_model
def lp2f_resnext50_32x4d_imnet200(pretrained=False, **kwargs):
    model = models.resnext50_32x4d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnext101_32x8d_imnet200(pretrained=False, **kwargs):
    model = models.resnext101_32x8d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnext101_64x4d_imnet200(pretrained=False, **kwargs):
    model = models.resnext101_64x4d(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# ConvNeXt
@register_model
def lp2f_convnext_base_imnet200(pretrained=False, **kwargs):
    model = models.convnext_base(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_convnext_large_imnet200(pretrained=False, **kwargs):
    model = models.convnext_large(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_convnext_small_imnet200(pretrained=False, **kwargs):
    model = models.convnext_small(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_convnext_tiny_imnet200(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, num_classes=200)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))
