"""(Base) Conv in CIFAR-100"""
import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model

# AlexNet
@register_model
def alexnet_cifar100(pretrained=False, **kwargs):
    model = models.alexnet(pretrained=pretrained, num_classes=100)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return model

# VGG
@register_model
def vgg11_cifar100(pretrained=False, **kwargs):
    model = models.vgg11(pretrained=pretrained, num_classes=100)
    return model

@register_model
def vgg13_cifar100(pretrained=False, **kwargs):
    model = models.vgg13(pretrained=pretrained, num_classes=100)
    return model

@register_model
def vgg16_cifar100(pretrained=False, **kwargs):
    model = models.vgg16(pretrained=pretrained, num_classes=100)
    return model

@register_model
def vgg19_cifar100(pretrained=False, **kwargs):
    model = models.vgg19(pretrained=pretrained, num_classes=100)
    return model

# VGG + BN
@register_model
def vgg11_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg11_bn(pretrained=pretrained, num_classes=100)
    return model

@register_model
def vgg13_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg13_bn(pretrained=pretrained, num_classes=100)
    return model

@register_model
def vgg16_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg16_bn(pretrained=pretrained, num_classes=100)
    return model

@register_model
def vgg19_bn_cifar100(pretrained=False, **kwargs):
    model = models.vgg19_bn(pretrained=pretrained, num_classes=100)
    return model

# ResNet
@register_model
def resnet18_cifar100(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def resnet34_cifar100(pretrained=False, **kwargs):
    model = models.resnet34(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def resnet50_cifar100(pretrained=False, **kwargs):
    model = models.resnet50(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def resnet101_cifar100(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def resnet152_cifar100(pretrained=False, **kwargs):
    model = models.resnet152(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

# DenseNet
@register_model
def densenet121_cifar100(pretrained=False, **kwargs):
    model = models.densenet121(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def densenet161_cifar100(pretrained=False, **kwargs):
    model = models.densenet161(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def densenet169_cifar100(pretrained=False, **kwargs):
    model = models.densenet169(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def densenet201_cifar100(pretrained=False, **kwargs):
    model = models.densenet201(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

# WideResNet
@register_model
def wide_resnet50_2_cifar100(pretrained=False, **kwargs):
    model = models.wide_resnet50_2(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

@register_model
def wide_resnet101_2_cifar100(pretrained=False, **kwargs):
    model = models.wide_resnet101_2(pretrained=pretrained, num_classes=100)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # for cifar shape 32x32
    return model

# ResNeXt
@register_model
def resnext50_32x4d_cifar100(pretrained=False, **kwargs):
    model = models.resnext50_32x4d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return model

@register_model
def resnext101_32x8d_cifar100(pretrained=False, **kwargs):
    model = models.resnext101_32x8d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return model

@register_model
def resnext101_64x4d_cifar100(pretrained=False, **kwargs):
    model = models.resnext101_64x4d(pretrained=pretrained, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
    return model

# ConvNeXt
@register_model
def convnext_base_cifar100(pretrained=False, **kwargs):
    model = models.convnext_base(pretrained=pretrained, num_classes=100)
    return model

@register_model
def convnext_large_cifar100(pretrained=False, **kwargs):
    model = models.convnext_large(pretrained=pretrained, num_classes=100)
    return model

@register_model
def convnext_small_cifar100(pretrained=False, **kwargs):
    model = models.convnext_small(pretrained=pretrained, num_classes=100)
    return model

@register_model
def convnext_tiny_cifar100(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, num_classes=100)
    return model

# ViT
@register_model
def vit_t_16_cifar100(pretrained=False, **kwargs):
    model = models.VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=100
    )
    return model

@register_model
def vit_b_16_cifar100(pretrained=False, **kwargs):
    model = models.vit_b_16(num_classes=100)
    return model

@register_model
def vit_b_32_cifar100(pretrained=False, **kwargs):
    model = models.vit_b_32(num_classes=100)
    return model

@register_model
def vit_l_16_cifar100(pretrained=False, **kwargs):
    model = models.vit_l_16(num_classes=100)
    return model

@register_model
def vit_l_32_cifar100(pretrained=False, **kwargs):
    model = models.vit_l_32(num_classes=100)
    return model

@register_model
def vit_h_14_cifar100(pretrained=False, **kwargs):
    model = models.vit_h_14(num_classes=100)
    return model
