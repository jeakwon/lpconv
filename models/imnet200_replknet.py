
# https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/replknet.py
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from timm.models.registry import register_model

import sys
import os


class LpConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, sigma, log2p, kernel_size,
                 stride=1, padding=0, bias=True, *args, **kwargs):
        super(LpConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias, *args, **kwargs)

        sigma = _pair(sigma)
        C_00 = 1 / (sigma[0] + 1e-4)
        C_11 = 1 / (sigma[1] + 1e-4)
        if log2p is None:
            self.register_buffer('log2p', None)
            self.register_buffer('C', None)
        else:
            self.log2p = nn.Parameter( torch.Tensor([log2p]).repeat(out_channels) )
            self.C = nn.Parameter( torch.Tensor( [[C_00, 0], [0, C_11]] ).repeat(out_channels, 1, 1) )

    def forward(self, input):
        return lp_convolution(input, self.out_channels, self.weight, self.bias, self.C, self.log2p,
        self.kernel_size, self.stride, self.padding, self.dilation, self.groups)

    def set_requires_grad(self, **params):
        for param, requires_grad in params.items():
            p = getattr(self, param)
            p.requires_grad = requires_grad
            print(f'{param}.requires_grad = {requires_grad}')

    @classmethod
    def convert(cls, conv2d, log2p, transfer_params=False, set_requires_grad={}, scale=2):
        in_channels=conv2d.in_channels
        out_channels=conv2d.out_channels
        kernel_size=(int(conv2d.kernel_size[0]*scale) + int(conv2d.kernel_size[0]%scale),
                     int(conv2d.kernel_size[1]*scale) + int(conv2d.kernel_size[1]%scale))
        stride=conv2d.stride
        padding=(conv2d.padding[0] + (kernel_size[0] - conv2d.kernel_size[0]) //2 ,
                 conv2d.padding[1] + (kernel_size[1] - conv2d.kernel_size[1]) //2 )

        dilation=conv2d.dilation
        groups=conv2d.groups
        padding_mode=conv2d.padding_mode
        bias=conv2d.bias is not None
        sigma=(conv2d.kernel_size[0] * 0.5, conv2d.kernel_size[1] * 0.5)

        new_conv2d = cls(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, padding_mode=padding_mode, bias=bias, sigma=sigma, log2p=log2p)

        if transfer_params:
            k0 = conv2d.kernel_size[0]//2+conv2d.kernel_size[0]%2
            k1 = conv2d.kernel_size[1]//2+conv2d.kernel_size[1]%2

            new_conv2d.weight.data = torch.zeros_like(new_conv2d.weight.data)
            new_conv2d.weight.data[:, :, k0:-k0, k1:-k1] = conv2d.weight.data
            if bias:
                new_conv2d.bias.data = conv2d.bias

        new_conv2d.set_requires_grad(**set_requires_grad)

        return new_conv2d

def lp_convolution(input, out_channels, weight, bias, C, log2p, kernel_size, stride, padding, dilation, groups, constraints=False):
    if log2p is None:
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    # offsets from kernel center
    x = torch.arange( kernel_size[0] ).to(input.device)
    y = torch.arange( kernel_size[1] ).to(input.device)
    xx, yy = torch.meshgrid(x, y)
    x0 = (x.max()-x.min())/2
    y0 = (y.max()-y.min())/2
    offset = torch.stack( [xx - x0, yy - y0] )

    # set bounds and constraints to keep C symmetric and positive definite
    if constraints: # was not applied in the paper
        C_00 = torch.clamp(C[:, 0, 0], min=1e-4)
        C_11 = torch.clamp(C[:, 1, 1], min=1e-4)
        C_01 = torch.max(C[:, 0, 1], torch.sqrt( C_00 * C_11 ))
        C_10 = torch.max(C[:, 1, 0], torch.sqrt( C_00 * C_11 ))
        C[:, 0, 0].data.fill_(C_00)
        C[:, 1, 1].data.fill_(C_11)
        C[:, 0, 1].data.fill_(C_01)
        C[:, 1, 0].data.fill_(C_10)

    Z = torch.einsum('cij, jmn -> cimn', C, offset).abs()
    mask = torch.exp( - Z.pow( 2**log2p[:, None, None, None] ).sum(dim=1, keepdim=True) )

    return F.conv2d(input, weight * mask, bias, stride, padding, dilation, groups)

def LpConvert(module: nn.Module, log2p: float, transfer_params=False, set_requires_grad={}, scale=2) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LpConv2d.convert(child, log2p=log2p, transfer_params=transfer_params, set_requires_grad=set_requires_grad, scale=scale))
        else:
            LpConvert(child, log2p, transfer_params=transfer_params, set_requires_grad=set_requires_grad, scale=scale)
    return module

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = get_bn(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKNetStage(nn.Module):

    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 small_kernel, dw_ratio=1, ffn_ratio=4,
                 use_checkpoint=False,      # train with torch.utils.checkpoint to save memory
                 small_kernel_merged=False,
                 norm_intermediate_features=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)
        if norm_intermediate_features:
            self.norm = get_bn(channels)    #   Only use this with RepLKNet-XL on downstream tasks
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)   # Save training memory
            else:
                x = blk(x)
        return x

class RepLKNet(nn.Module):

    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 use_checkpoint=False,
                 small_kernel_merged=False,
                 use_sync_bn=True,
                 norm_intermediate_features=False       # for RepLKNet-XL on COCO and ADE20K, use an extra BN to normalize the intermediate feature maps then feed them into the heads
                 ):
        super().__init__()

        if num_classes is None and out_indices is None:
            raise ValueError('must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and out_indices is not None:
            raise ValueError('cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError('for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices
        if use_sync_bn:
            enable_sync_bn()

        base_width = channels[0]
        self.use_checkpoint = use_checkpoint
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)
        self.stem = nn.ModuleList([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)

        if num_classes is not None:
            self.norm = get_bn(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels[-1], num_classes)



    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(stem_layer, x)     # save memory
            else:
                x = stem_layer(x)

        if self.out_indices is None:
            #   Just need the final output
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return x
        else:
            #   Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx in self.out_indices:
                    outs.append(self.stages[stage_idx].norm(x))     # For RepLKNet-XL normalize the features before feeding them into the heads
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.out_indices:
            return x
        else:
            x = self.norm(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = get_conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()

def create_RepLKNet31T(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,6,2], channels=[64,128,256,512],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[128,256,512,1024],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_RepLKNet31L(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[192,384,768,1536],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_RepLKNetXL(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[27,27,27,13], layers=[2,2,18,2], channels=[256,512,1024,2048],
                    drop_path_rate=drop_path_rate, small_kernel=None, dw_ratio=1.5,
                    num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)


@register_model
def replknet31t_imnet200(pretrained=False, **kwargs):
    model = create_RepLKNet31T(num_classes=200)
    return model

@register_model
def replknet31b_imnet200(pretrained=False, **kwargs):
    model = create_RepLKNet31B(num_classes=200)
    return model

@register_model
def replknet31l_imnet200(pretrained=False, **kwargs):
    model = create_RepLKNet31L(num_classes=200)
    return model

@register_model
def replknet31xl_imnet200(pretrained=False, **kwargs):
    model = create_RepLKNetXL(num_classes=200)
    return model

@register_model
def lp2a_replknet31t_imnet200(pretrained=False, **kwargs):
    model = create_RepLKNet31T(num_classes=200)
    return LpConvert(model, log2p=1, scale=1)
