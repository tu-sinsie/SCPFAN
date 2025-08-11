import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_helper import SimAM, CBAM, ECA


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0*g:1*g, 1, 2] = 1.0
        mask[:, 1*g:2*g, 1, 0] = 1.0
        mask[:, 2*g:3*g, 2, 1] = 1.0
        mask[:, 3*g:4*g, 0, 1] = 1.0
        mask[:, 4*g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0*g:1*g, 0, 1, 2] = 1.0  # left
        self.weight[1*g:2*g, 0, 1, 0] = 1.0  # right
        self.weight[2*g:3*g, 0, 2, 1] = 1.0  # up
        self.weight[3*g:4*g, 0, 0, 1] = 1.0  # down
        self.weight[4*g:, 0, 1, 1] = 1.0  # identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(LFE, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels*exp_ratio)
        self.conv1 = ShiftConv2d(out_channels*exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        y = x + y
        return y


class SCPFARB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=[4, 8, 12], shared_depth=1, attention_type="SimAM"):
        super(SCPFARB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth
        self.attention_type = attention_type

        modules_lfe = {}
        modules_atten = {}
        modules_lfe['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        if self.attention_type == "CBAM":
            modules_atten['atten_0'] = CBAM(in_planes=out_channels)
        elif self.attention_type == "SimAM":
            modules_atten['atten_0'] = SimAM()
        elif self.attention_type == "ECA":
            modules_atten['atten_0'] = ECA(in_planes=out_channels)

        for i in range(shared_depth):
            modules_lfe['lfe_{}'.format(i+1)] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
            if self.attention_type == "CBAM":
                modules_atten['atten_{}'.format(i+1)] = CBAM(in_planes=out_channels)
            elif self.attention_type == "SimAM":
                modules_atten['atten_{}'.format(i+1)] = SimAM()
            elif self.attention_type == "ECA":
                modules_atten['atten_{}'.format(i+1)] = ECA(in_planes=out_channels)
            # else:
            #     print("not add attention module !!!")

        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_atten = nn.ModuleDict(modules_atten)

    def forward(self, x):
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:  # only calculate attention for the 1-st module
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                
                if not len(self.modules_atten):
                    y = x
                else:
                    y = self.modules_atten['atten_{}'.format(i)](x)
                # x = y + x
                x = y
            else:
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                if not len(self.modules_atten):
                    y = x
                else:
                    y = self.modules_atten['atten_{}'.format(i)](x)
                # x = y + x
                x = y
        return x
