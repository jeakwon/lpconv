"""frozen Lp-Conv p=256 in CIFAR-100"""
import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert

# AlexNet
@register_model
def lp2f_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# VGG
@register_model
def lp2f_vgg11_cifar100(pretrained=False, **kwargs):
    model = models.vgg11(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg13_cifar100(pretrained=False, **kwargs):
    model = models.vgg13(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg16_cifar100(pretrained=False, **kwargs):
    model = models.vgg16(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg19_cifar100(pretrained=False, **kwargs):
    model = models.vgg19(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# VGG + BN
@register_model
def lp2f_vgg11_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg11_bn(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg13_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg13_bn(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg16_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vgg19_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg19_bn(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# ResNet
@register_model
def lp2f_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnet34_cifar100(pretrained=False, **kwargs):
    model = models.resnet34(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnet50_cifar100(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnet101_cifar100(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnet152_cifar100(pretrained=False, **kwargs):
    model = models.resnet152(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# DenseNet
@register_model
def lp2f_densenet121_cifar100(pretrained=False, **kwargs):
    model = models.densenet121(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_densenet161_cifar100(pretrained=False, **kwargs):
    model = models.densenet161(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_densenet169_cifar100(pretrained=False, **kwargs):
    model = models.densenet169(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_densenet201_cifar100(pretrained=False, **kwargs):
    model = models.densenet201(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# WideResNet
@register_model
def lp2f_wide_resnet50_2_cifar100(pretrained=False, **kwargs):
    model = models.wide_resnet50_2(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_wide_resnet101_2_cifar100(pretrained=False, **kwargs):
    model = models.wide_resnet101_2(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# ResNeXt
@register_model
def lp2f_resnext50_32x4d_cifar100(pretrained=False, **kwargs):
    model = models.resnext50_32x4d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnext101_32x8d_cifar100(pretrained=False, **kwargs):
    model = models.resnext101_32x8d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_resnext101_64x4d_cifar100(pretrained=False, **kwargs):
    model = models.resnext101_64x4d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# ConvNeXt
@register_model
def lp2f_convnext_base_cifar100(pretrained=False, **kwargs):
    model = models.convnext_base(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_convnext_large_cifar100(pretrained=False, **kwargs):
    model = models.convnext_large(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_convnext_small_cifar100(pretrained=False, **kwargs):
    model = models.convnext_small(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_convnext_tiny_cifar100(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

# ViT
@register_model
def lp2f_vit_s_cifar100(pretrained=False, **kwargs):
    model = models.VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        num_classes=100
    )
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vit_b_cifar100(pretrained=False, **kwargs):
    model = models.VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=100
    )
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))

@register_model
def lp2f_vit_l_cifar100(pretrained=False, **kwargs):
    model = models.VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        num_classes=100
    )
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))