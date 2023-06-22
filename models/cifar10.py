import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert

num_classes = 10

# AlexNet
@register_model
def alexnet_cifar10():
    model = models.alexnet(pretrained=False, num_classes=num_classes)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return model

# VGG
@register_model
def vgg11_cifar10():
    model = models.vgg11(pretrained=False, num_classes=num_classes)
    return model

@register_model
def vgg13_cifar10():
    model = models.vgg13(pretrained=False, num_classes=num_classes)
    return model

@register_model
def vgg16_cifar10():
    model = models.vgg16(pretrained=False, num_classes=num_classes)
    return model

@register_model
def vgg19_cifar10():
    model = models.vgg19(pretrained=False, num_classes=num_classes)
    return model

# VGG + BN
@register_model
def vgg11_bn_cifar10():
    model = models.vgg11_bn(pretrained=False, num_classes=num_classes)
    return model

@register_model
def vgg13_bn_cifar10():
    model = models.vgg13_bn(pretrained=False, num_classes=num_classes)
    return model

@register_model
def vgg16_bn_cifar10():
    model = models.vgg16_bn(pretrained=False, num_classes=num_classes)
    return model

@register_model
def vgg19_bn_cifar10():
    model = models.vgg19_bn(pretrained=False, num_classes=num_classes)
    return model

# ResNet
@register_model
def resnet18_cifar10():
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def resnet34_cifar10():
    model = models.resnet34(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def resnet50_cifar10():
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def resnet101_cifar10():
    model = models.resnet101(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def resnet152_cifar10():
    model = models.resnet152(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

# DenseNet
@register_model
def densenet121_cifar10():
    model = models.densenet121(pretrained=False, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def densenet161_cifar10():
    model = models.densenet161(pretrained=False, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def densenet169_cifar10():
    model = models.densenet169(pretrained=False, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def densenet201_cifar10():
    model = models.densenet201(pretrained=False, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

# WideResNet
@register_model
def wide_resnet50_2_cifar10():
    model = models.wide_resnet50_2(pretrained=False, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def wide_resnet101_2_cifar10():
    model = models.wide_resnet101_2(pretrained=False, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

# ResNeXt
@register_model
def resnext50_32x4d_cifar10():
    model = models.resnext50_32x4d(pretrained=False, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return model

@register_model
def resnext101_32x8d_cifar10():
    model = models.resnext101_32x8d(pretrained=False, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return model

@register_model
def resnext101_64x4d_cifar10():
    model = models.resnext101_64x4d(pretrained=False, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return model

# ConvNeXt
@register_model
def convnext_base_cifar10():
    model = models.convnext_base(pretrained=False, num_classes=num_classes)
    return model

@register_model
def convnext_large_cifar10():
    model = models.convnext_large(pretrained=False, num_classes=num_classes)
    return model

@register_model
def convnext_small_cifar10():
    model = models.convnext_small(pretrained=False, num_classes=num_classes)
    return model

@register_model
def convnext_tiny_cifar10():
    model = models.convnext_tiny(pretrained=False, num_classes=num_classes)
    return model
