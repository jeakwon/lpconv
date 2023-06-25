import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import _pair

class LpConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, sigma, log2p, kernel_size,
                 stride=1, padding=0, bias=True, *args, **kwargs):
        super(LpConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias, *args, **kwargs)

        sigma = _pair(sigma)
        C_00 = 1 / (sigma[0] + 1e-4)
        C_11 = 1 / (sigma[1] + 1e-4)
        self.C = nn.Parameter( torch.Tensor( [[C_00, 0], [0, C_11]] ).repeat(out_channels, 1, 1) )
        if log2p is None:
            self.register_buffer('log2p', None)
        else:
            self.log2p = nn.Parameter( torch.Tensor([log2p]).repeat(out_channels) )

    def forward(self, input):
        return lp_convolution(input, self.out_channels, self.weight, self.bias, self.C, self.log2p, 
        self.kernel_size, self.stride, self.padding, self.dilation, self.groups)

    def set_requires_grad(self, **params):
        for param, requires_grad in params.items():
            p = getattr(self, param)
            p.requires_grad = requires_grad
            print(f'{param}.requires_grad = {requires_grad}')

    @classmethod
    def convert(cls, conv2d, log2p, transfer_params=False, set_requires_grad={}):
        in_channels=conv2d.in_channels
        out_channels=conv2d.out_channels
        kernel_size=(conv2d.kernel_size[0]*2 + conv2d.kernel_size[0]%2,
                     conv2d.kernel_size[1]*2 + conv2d.kernel_size[1]%2)
        stride=conv2d.stride
        padding=(conv2d.padding[0] + conv2d.kernel_size[0]//2+conv2d.kernel_size[0]%2,
                 conv2d.padding[1] + conv2d.kernel_size[1]//2+conv2d.kernel_size[1]%2)
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

            new_conv2d.weight.data[:, :, k0:-k0, k1:-k1] = conv2d.weight.data
            if bias:
                new_conv2d.bias.data = conv2d.bias

        new_conv2d.set_requires_grad(**set_requires_grad)

        return new_conv2d

def lp_convolution(input, out_channels, weight, bias, C, log2p, kernel_size, stride, padding, dilation, groups):
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
    C[:, 0, 0].data = C_00 = torch.clamp(C[:, 0, 0], min=1e-4)
    C[:, 1, 1].data = C_11 = torch.clamp(C[:, 1, 1], min=1e-4)
    C[:, 0, 1].data = C_01 = torch.max(C[:, 1, 1], torch.sqrt( C_00 * C_11 ))
    C[:, 1, 0].data = C_10 = torch.max(C[:, 1, 1], torch.sqrt( C_00 * C_11 ))

    Z = torch.einsum('cij, jmn -> cimn', C, offset).abs()
    mask = torch.exp( - Z.pow( 2**log2p[:, None, None, None] ).sum(dim=1, keepdim=True) )

    return F.conv2d(input, weight * mask, bias, stride, padding, dilation, groups)

def LpConvert(module: nn.Module, log2p: float, transfer_params=True, set_requires_grad={}) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LpConv2d.convert(child, log2p=log2p, transfer_params=transfer_params, set_requires_grad=set_requires_grad))
        else:
            LpConvert(child, log2p, transfer_params=transfer_params, set_requires_grad=set_requires_grad)
    return module
