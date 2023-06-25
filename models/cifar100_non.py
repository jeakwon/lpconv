import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv2 import LpConvert

# AlexNet
@register_model
def non_alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

# VGG
@register_model
def non_vgg11_cifar100(pretrained=False, **kwargs):
    model = models.vgg11(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_vgg13_cifar100(pretrained=False, **kwargs):
    model = models.vgg13(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_vgg16_cifar100(pretrained=False, **kwargs):
    model = models.vgg16(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_vgg19_cifar100(pretrained=False, **kwargs):
    model = models.vgg19(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

# VGG + BN
@register_model
def non_vgg11_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg11_bn(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_vgg13_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg13_bn(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_vgg16_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_vgg19_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg19_bn(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

# ResNet
@register_model
def non_resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

@register_model
def non_resnet34_cifar100(pretrained=False, **kwargs):
    model = models.resnet34(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

@register_model
def non_resnet50_cifar100(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

@register_model
def non_resnet101_cifar100(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

@register_model
def non_resnet152_cifar100(pretrained=False, **kwargs):
    model = models.resnet152(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

# DenseNet
@register_model
def non_densenet121_cifar100(pretrained=False, **kwargs):
    model = models.densenet121(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

@register_model
def non_densenet161_cifar100(pretrained=False, **kwargs):
    model = models.densenet161(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

@register_model
def non_densenet169_cifar100(pretrained=False, **kwargs):
    model = models.densenet169(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

@register_model
def non_densenet201_cifar100(pretrained=False, **kwargs):
    model = models.densenet201(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

# WideResNet
@register_model
def non_wide_resnet50_2_cifar100(pretrained=False, **kwargs):
    model = models.wide_resnet50_2(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

@register_model
def non_wide_resnet101_2_cifar100(pretrained=False, **kwargs):
    model = models.wide_resnet101_2(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=None)

# ResNeXt
@register_model
def non_resnext50_32x4d_cifar100(pretrained=False, **kwargs):
    model = models.resnext50_32x4d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=None)

@register_model
def non_resnext101_32x8d_cifar100(pretrained=False, **kwargs):
    model = models.resnext101_32x8d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=None)

@register_model
def non_resnext101_64x4d_cifar100(pretrained=False, **kwargs):
    model = models.resnext101_64x4d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=None)

# ConvNeXt
@register_model
def non_convnext_base_cifar100(pretrained=False, **kwargs):
    model = models.convnext_base(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_convnext_large_cifar100(pretrained=False, **kwargs):
    model = models.convnext_large(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_convnext_small_cifar100(pretrained=False, **kwargs):
    model = models.convnext_small(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)

@register_model
def non_convnext_tiny_cifar100(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, num_classes=100)
    return LpConvert(model, log2p=None)
