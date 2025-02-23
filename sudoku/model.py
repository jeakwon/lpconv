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
    if constraints:
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

def LpConvert(module: nn.Module, log2p: float, transfer_params=True, set_requires_grad={}) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LpConv2d.convert(child, log2p=log2p, transfer_params=transfer_params, set_requires_grad=set_requires_grad))
        else:
            LpConvert(child, log2p, transfer_params=transfer_params, set_requires_grad=set_requires_grad)
    return module

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        """It only support square kernels and stride=1, dilation=1, groups=1."""
        super(Conv2dSame, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.net(x)

class SudokuCNN(nn.Module):
    def __init__(self, num_hidden=256, num_layers=15):
        super(SudokuCNN, self).__init__()
        self.conv_layers = nn.Sequential()
        N_in, N_out = 1, num_hidden
        for i in range(num_layers):
            self.conv_layers.append( Conv2dSame(N_in, N_out, 3) )
            N_in, N_out = num_hidden, num_hidden
        self.last_conv = nn.Conv2d(N_out, 9, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.last_conv(x)
        return x

def sudoku_lpconv(num_hidden=512, num_layers=15, log2p=None, lpconvert=True, learnable=True):
    model = SudokuCNN(num_hidden=num_hidden, num_layers=num_layers)
    if lpconvert:
        if log2p is None:
            return LpConvert(model, log2p=log2p)
        else:
            return LpConvert(model, log2p=log2p, set_requires_grad=dict(log2p=learnable, C=learnable))
    return model