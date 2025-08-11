import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.scpfa_block import SCPFARB, MeanShift


def create_model(args):
    return SCPFA(args)


class SCPFA(nn.Module):
    def __init__(self, args):
        super(SCPFA, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.m_scpfa = args.m_scpfa
        self.c_scpfa = args.c_scpfa
        self.n_share = args.n_share
        self.r_expand = args.r_expand
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)
        self.attention_type = args.attention_type

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_scpfa, kernel_size=1, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_scpfa // (1+self.n_share)):
            if (i+1) % 2 == 1:
                m_body.append(
                    SCPFARB(
                        self.c_scpfa, self.c_scpfa, self.r_expand, 0,
                        self.window_sizes, shared_depth=self.n_share,
                        attention_type=self.attention_type
                    )
                )
            else:
                m_body.append(
                    SCPFARB(
                        self.c_scpfa, self.c_scpfa, self.r_expand, 1,
                        self.window_sizes, shared_depth=self.n_share,
                        attention_type=self.attention_type
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_scpfa, self.colors*self.scale * self.scale, kernel_size=1, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        # print(H, W)
        x1 = self.check_image_size(x)
        # print("x1.shape:", x1.shape)
        x2 = self.sub_mean(x1)
        # print("x2.shape:", x2.shape)
        x3 = self.head(x2)
        # print("x3.shape:", x3.shape)
        res = self.body(x3)
        res = res + x3
        x4 = self.tail(res)
        x5 = self.add_mean(x4)
        out = x5[:, :, 0:H*self.scale, 0:W*self.scale]

        # base = F.interpolate(x, scale_factor=self.scale,
        #                      mode='bilinear', align_corners=False)
        # out += base
        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
     
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                print(name)
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    pass
