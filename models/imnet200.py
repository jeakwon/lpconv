"""(Base) Conv in TinyImageNet"""
import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model

# AlexNet
@register_model
def alexnet_imnet200(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=200)
    return model

# VGG
@register_model
def vgg11_imnet200(pretrained=False, **kwargs):
    model = models.vgg11(pretrained=pretrained, num_classes=200)
    return model

@register_model
def vgg13_imnet200(pretrained=False, **kwargs):
    model = models.vgg13(pretrained=pretrained, num_classes=200)
    return model

@register_model
def vgg16_imnet200(pretrained=False, **kwargs):
    model = models.vgg16(pretrained=pretrained, num_classes=200)
    return model

@register_model
def vgg19_imnet200(pretrained=False, **kwargs):
    model = models.vgg19(pretrained=pretrained, num_classes=200)
    return model

# VGG + BN
@register_model
def vgg11_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg11_bn(pretrained=pretrained, num_classes=200)
    return model

@register_model
def vgg13_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg13_bn(pretrained=pretrained, num_classes=200)
    return model

@register_model
def vgg16_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=200)
    return model

@register_model
def vgg19_bn_imnet200(pretrained=False, **kwargs):
    model = models.vgg19_bn(pretrained=pretrained, num_classes=200)
    return model

# ResNet
@register_model
def resnet18_imnet200(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=200)
    return model

@register_model
def resnet34_imnet200(pretrained=False, **kwargs):
    model = models.resnet34(pretrained=pretrained, num_classes=200)
    return model

@register_model
def resnet50_imnet200(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained, num_classes=200)
    return model

@register_model
def resnet101_imnet200(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained, num_classes=200)
    return model

@register_model
def resnet152_imnet200(pretrained=False, **kwargs):
    model = models.resnet152(pretrained=pretrained, num_classes=200)
    return model

# DenseNet
@register_model
def densenet121_imnet200(pretrained=False, **kwargs):
    model = models.densenet121(pretrained=pretrained, num_classes=200)
    return model

@register_model
def densenet161_imnet200(pretrained=False, **kwargs):
    model = models.densenet161(pretrained=pretrained, num_classes=200)
    return model

@register_model
def densenet169_imnet200(pretrained=False, **kwargs):
    model = models.densenet169(pretrained=pretrained, num_classes=200)
    return model

@register_model
def densenet201_imnet200(pretrained=False, **kwargs):
    model = models.densenet201(pretrained=pretrained, num_classes=200)
    return model

# WideResNet
@register_model
def wide_resnet50_2_imnet200(pretrained=False, **kwargs):
    model = models.wide_resnet50_2(pretrained=pretrained, num_classes=200)
    return model

@register_model
def wide_resnet101_2_imnet200(pretrained=False, **kwargs):
    model = models.wide_resnet101_2(pretrained=pretrained, num_classes=200)
    return model

# ResNeXt
@register_model
def resnext50_32x4d_imnet200(pretrained=False, **kwargs):
    model = models.resnext50_32x4d(pretrained=pretrained, num_classes=200)
    return model

@register_model
def resnext101_32x8d_imnet200(pretrained=False, **kwargs):
    model = models.resnext101_32x8d(pretrained=pretrained, num_classes=200)
    return model

@register_model
def resnext101_64x4d_imnet200(pretrained=False, **kwargs):
    model = models.resnext101_64x4d(pretrained=pretrained, num_classes=200)
    return model

# ConvNeXt
@register_model
def convnext_base_imnet200(pretrained=False, **kwargs):
    model = models.convnext_base(pretrained=pretrained, num_classes=200)
    return model

@register_model
def convnext_large_imnet200(pretrained=False, **kwargs):
    model = models.convnext_large(pretrained=pretrained, num_classes=200)
    return model

@register_model
def convnext_small_imnet200(pretrained=False, **kwargs):
    model = models.convnext_small(pretrained=pretrained, num_classes=200)
    return model

@register_model
def convnext_tiny_imnet200(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, num_classes=200)
    return model

# ViT
@register_model
def vit_t_16_imnet200(pretrained=False, **kwargs):
    model = models.VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=200
    )
    return model

@register_model
def vit_b_16_imnet200(pretrained=False, **kwargs):
    model = models.vit_b_16(num_classes=200)
    return model

@register_model
def vit_b_32_imnet200(pretrained=False, **kwargs):
    model = models.vit_b_32(num_classes=200)
    return model

@register_model
def vit_l_16_imnet200(pretrained=False, **kwargs):
    model = models.vit_l_16(num_classes=200)
    return model

@register_model
def vit_l_32_imnet200(pretrained=False, **kwargs):
    model = models.vit_l_32(num_classes=200)
    return model

@register_model
def vit_h_14_imnet200(pretrained=False, **kwargs):
    model = models.vit_h_14(num_classes=200)
    return model
