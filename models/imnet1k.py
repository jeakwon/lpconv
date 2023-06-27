import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert

# AlexNet
@register_model
def alexnet_imnet1k(pretrained=True, **kwargs):
    weight = models.AlexNet_Weights.DEFAULT if pretrained else None
    model = models.alexnet(weight=weight)
    return model

# VGG
@register_model
def vgg11_imnet1k(pretrained=True, **kwargs):
    weight = models.VGG11_Weights.DEFAULT.DEFAULT if pretrained else None
    model = models.vgg11(weight=weight)
    return model

@register_model
def vgg13_imnet1k(pretrained=True, **kwargs):
    weight = models.VGG13_Weights.DEFAULT if pretrained else None
    model = models.vgg13(weight=weight)
    return model

@register_model
def vgg16_imnet1k(pretrained=True, **kwargs):
    weight = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weight=weight)
    return model

@register_model
def vgg19_imnet1k(pretrained=True, **kwargs):
    weight = models.VGG19_Weights.DEFAULT if pretrained else None
    model = models.vgg19(weight=weight)
    return model

# VGG + BN
@register_model
def vgg11_bn_imnet1k(pretrained=True, **kwargs):
    weight = models.VGG11_BN_Weights.DEFAULT if pretrained else None
    model = models.vgg11_bn(weight=weight)
    return model

@register_model
def vgg13_bn_imnet1k(pretrained=True, **kwargs):
    weight = models.VGG13_BN_Weights.DEFAULT if pretrained else None
    model = models.vgg13_bn(weight=weight)
    return model

@register_model
def vgg16_bn_imnet1k(pretrained=True, **kwargs):
    weight = models.VGG16_BN_Weights.DEFAULT if pretrained else None
    model = models.vgg16_bn(weight=weight)
    return model

@register_model
def vgg19_bn_imnet1k(pretrained=True, **kwargs):
    weight = models.VGG19_BN_Weights.DEFAULT if pretrained else None
    model = models.vgg19_bn(weight=weight)
    return model

# ResNet
@register_model
def resnet18_imnet1k(pretrained=True, **kwargs):
    weight = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weight=weight)
    return model

@register_model
def resnet34_imnet1k(pretrained=True, **kwargs):
    weight = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weight=weight)
    return model

@register_model
def resnet50_imnet1k(pretrained=True, **kwargs):
    weight = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weight=weight)
    return model

@register_model
def resnet101_imnet1k(pretrained=True, **kwargs):
    weight = models.ResNet101_Weights.DEFAULT if pretrained else None
    model = models.resnet101(weight=weight)
    return model

@register_model
def resnet152_imnet1k(pretrained=True, **kwargs):
    weight = models.ResNet152_Weights.DEFAULT if pretrained else None
    model = models.resnet152(weight=weight)
    return model

# DenseNet
@register_model
def densenet121_imnet1k(pretrained=True, **kwargs):
    weight = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weight=weight)
    return model

@register_model
def densenet161_imnet1k(pretrained=True, **kwargs):
    weight = models.DenseNet161_Weights.DEFAULT if pretrained else None
    model = models.densenet161(weight=weight)
    return model

@register_model
def densenet169_imnet1k(pretrained=True, **kwargs):
    weight = models.DenseNet169_Weights.DEFAULT if pretrained else None
    model = models.densenet169(weight=weight)
    return model

@register_model
def densenet201_imnet1k(pretrained=True, **kwargs):
    weight = models.DenseNet201_Weights.DEFAULT if pretrained else None
    model = models.densenet201(weight=weight)
    return model

# WideResNet
@register_model
def wide_resnet50_2_imnet1k(pretrained=True, **kwargs):
    weight = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
    model = models.wide_resnet50_2(weight=weight)
    return model

@register_model
def wide_resnet101_2_imnet1k(pretrained=True, **kwargs):
    weight = models.Wide_ResNet101_2_Weights.DEFAULT if pretrained else None
    model = models.wide_resnet101_2(weight=weight)
    return model

# ResNeXt
@register_model
def resnext50_32x4d_imnet1k(pretrained=True, **kwargs):
    weight = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
    model = models.resnext50_32x4d(weight=weight)
    return model

@register_model
def resnext101_32x8d_imnet1k(pretrained=True, **kwargs):
    weight = models.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
    model = models.resnext101_32x8d(weight=weight)
    return model

@register_model
def resnext101_64x4d_imnet1k(pretrained=True, **kwargs):
    weight = models.ResNeXt101_64X4D_Weights.DEFAULT if pretrained else None
    model = models.resnext101_64x4d(weight=weight)
    return model

# ConvNeXt
@register_model
def convnext_base_imnet1k(pretrained=True, **kwargs):
    weight = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
    model = models.convnext_base(weight=weight)
    return model

@register_model
def convnext_large_imnet1k(pretrained=True, **kwargs):
    weight = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
    model = models.convnext_large(weight=weight)
    return model

@register_model
def convnext_small_imnet1k(pretrained=True, **kwargs):
    weight = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
    model = models.convnext_small(weight=weight)
    return model

@register_model
def convnext_tiny_imnet1k(pretrained=True, **kwargs):
    weight = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weight=weight)
    return model
