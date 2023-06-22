import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert

num_classes = 10

# AlexNet
@register_model
def lp_alexnet_cifar10(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=num_classes)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

# VGG
@register_model
def lp_vgg11_cifar10(pretrained=False, **kwargs):
    model = models.vgg11(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_vgg13_cifar10(pretrained=False, **kwargs):
    model = models.vgg13(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_vgg16_cifar10(pretrained=False, **kwargs):
    model = models.vgg16(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_vgg19_cifar10(pretrained=False, **kwargs):
    model = models.vgg19(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

# VGG + BN
@register_model
def lp_vgg11_bn_cifar10(pretrained=False, **kwargs):
    model = models.vgg11_bn(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_vgg13_bn_cifar10(pretrained=False, **kwargs):
    model = models.vgg13_bn(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_vgg16_bn_cifar10(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_vgg19_bn_cifar10(pretrained=False, **kwargs):
    model = models.vgg19_bn(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

# ResNet
@register_model
def lp_resnet18_cifar10(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp_resnet34_cifar10(pretrained=False, **kwargs):
    model = models.resnet34(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp_resnet50_cifar10(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp_resnet101_cifar10(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp_resnet152_cifar10(pretrained=False, **kwargs):
    model = models.resnet152(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

# DenseNet
@register_model
def lp_densenet121_cifar10(pretrained=False, **kwargs):
    model = models.densenet121(pretrained=pretrained, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp_densenet161_cifar10(pretrained=False, **kwargs):
    model = models.densenet161(pretrained=pretrained, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp_densenet169_cifar10(pretrained=False, **kwargs):
    model = models.densenet169(pretrained=pretrained, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp_densenet201_cifar10(pretrained=False, **kwargs):
    model = models.densenet201(pretrained=pretrained, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

# WideResNet
@register_model
def lp_wide_resnet50_2_cifar10(pretrained=False, **kwargs):
    model = models.wide_resnet50_2(pretrained=pretrained, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp_wide_resnet101_2_cifar10(pretrained=False, **kwargs):
    model = models.wide_resnet101_2(pretrained=pretrained, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

# ResNeXt
@register_model
def lp_resnext50_32x4d_cifar10(pretrained=False, **kwargs):
    model = models.resnext50_32x4d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=4)

@register_model
def lp_resnext101_32x8d_cifar10(pretrained=False, **kwargs):
    model = models.resnext101_32x8d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=4)

@register_model
def lp_resnext101_64x4d_cifar10(pretrained=False, **kwargs):
    model = models.resnext101_64x4d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return LpConvert(model, log2p=4)

# ConvNeXt
@register_model
def lp_convnext_base_cifar10(pretrained=False, **kwargs):
    model = models.convnext_base(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_convnext_large_cifar10(pretrained=False, **kwargs):
    model = models.convnext_large(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_convnext_small_cifar10(pretrained=False, **kwargs):
    model = models.convnext_small(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)

@register_model
def lp_convnext_tiny_cifar10(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, num_classes=num_classes)
    return LpConvert(model, log2p=4)
