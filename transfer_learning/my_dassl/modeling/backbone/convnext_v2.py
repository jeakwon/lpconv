import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model
from .lpconv import LpConvert

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(Backbone):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1., drop_rate=None
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def convnextv2_tiny(pretrained=True, pretrained_path="/data1/lsj9862/lpconv/transfer_learning/convnextv2_tiny_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['model'])
        print("Successfully load pre-trained ConvNextV2")
    return model

@BACKBONE_REGISTRY.register()
def large_convnextv2_tiny(pretrained=True, pretrained_path="/data1/lsj9862/lpconv/transfer_learning/convnextv2_tiny_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['model'])
    return LpConvert(model, log2p=None)

@BACKBONE_REGISTRY.register()
def lp2a_convnextv2_tiny(pretrained=True, pretrained_path="/data1/lsj9862/lpconv/transfer_learning/convnextv2_tiny_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['model'])
    return LpConvert(model, log2p=0)

@BACKBONE_REGISTRY.register()
def lp2b_convnextv2_tiny(pretrained=True, pretrained_path="/data1/lsj9862/lpconv/transfer_learning/convnextv2_tiny_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['model'])
    return LpConvert(model, log2p=1)

@BACKBONE_REGISTRY.register()
def lp2c_convnextv2_tiny(pretrained=True, pretrained_path="/data1/lsj9862/lpconv/transfer_learning/convnextv2_tiny_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['model'])
    return LpConvert(model, log2p=2)

@BACKBONE_REGISTRY.register()
def lp2d_convnextv2_tiny(pretrained=True, pretrained_path="/data1/lsj9862/lpconv/transfer_learning/convnextv2_tiny_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['model'])
    return LpConvert(model, log2p=3)

@BACKBONE_REGISTRY.register()
def lp2e_convnextv2_tiny(pretrained=True, pretrained_path="/data1/lsj9862/lpconv/transfer_learning/convnextv2_tiny_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['model'])
    return LpConvert(model, log2p=4)

@BACKBONE_REGISTRY.register()
def lp2f_convnextv2_tiny(pretrained=True, pretrained_path="/data1/lsj9862/lpconv/transfer_learning/convnextv2_tiny_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict['model'])
    return LpConvert(model, log2p=8, set_requires_grad=dict(log2p=False, C=False))