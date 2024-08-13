"""Lp-Conv additional exp in CIFAR-100 with DeepAlexNet"""
import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from .lpconv import LpConvert

import torch
import torch.nn as nn
import torch
import torch.nn as nn

class DeepAlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, num_new_layers: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        for _ in range(num_new_layers):
            self.features.append( nn.Conv2d(256, 256, kernel_size=3, padding=1) )
            self.features.append( nn.ReLU(inplace=True) )

        self.features.append( nn.MaxPool2d(kernel_size=3, stride=2) )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# AlexNet
@register_model
def lp2a_alexnet13_cifar100(pretrained=False, **kwargs):
    model = DeepAlexNet(num_classes=100, num_new_layers=5)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=1)

@register_model
def lp2a_alexnet18_cifar100(pretrained=False, **kwargs):
    model = DeepAlexNet(num_classes=100, num_new_layers=10)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=1)


# AlexNet
@register_model
def lp2b_alexnet13_cifar100(pretrained=False, **kwargs):
    model = DeepAlexNet(num_classes=100, num_new_layers=5)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=2)

@register_model
def lp2b_alexnet18_cifar100(pretrained=False, **kwargs):
    model = DeepAlexNet(num_classes=100, num_new_layers=10)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=2)


# AlexNet
@register_model
def lp2c_alexnet13_cifar100(pretrained=False, **kwargs):
    model = DeepAlexNet(num_classes=100, num_new_layers=5)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=3)

@register_model
def lp2c_alexnet18_cifar100(pretrained=False, **kwargs):
    model = DeepAlexNet(num_classes=100, num_new_layers=10)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=3)


# AlexNet
@register_model
def lp2_alexnet13_cifar100(pretrained=False, **kwargs):
    model = DeepAlexNet(num_classes=100, num_new_layers=5)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

@register_model
def lp2_alexnet18_cifar100(pretrained=False, **kwargs):
    model = DeepAlexNet(num_classes=100, num_new_layers=10)
    model.features[0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # for cifar shape 32x32
    return LpConvert(model, log2p=4)

