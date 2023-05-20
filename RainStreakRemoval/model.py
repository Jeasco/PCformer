import os
import math
import torch
import torch.nn as nn
import numpy as np
import functools
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 8

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

#Self-Attention Gated Module
class SGM(nn.Module):
    def __init__(self, channels=32, kernel_size=3, stride=1, bias=False):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
        self.k = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
        self.v = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
        self.attend = nn.Softmax(dim=-1)
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.flatten(2).contiguous()
        k = k.flatten(2).transpose(1, 2).contiguous()
        v = v.flatten(2).contiguous()

        dot = torch.matmul(q, k)
        attn = self.attend(dot)
        out = torch.matmul(attn, v)
        out = out.contiguous().view(B, C, H, W)

        out += x
        return out

#Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = y_avg + y_max
        y = self.conv(y)
        return x * y + x

#Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, channel,  bias=False):
        super(SALayer, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel//4, 3, padding=1, bias=bias),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channel//4, 1, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y + x

#Encoder
class convdownblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, down = True):
        super(convdownblock, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size // 2), bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      padding=(kernel_size // 2), bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      padding=(kernel_size // 2), bias=False),

            nn.LeakyReLU(0.2, inplace=True)
        )

        self.s_att = SALayer(out_channels)
        self.c_att = CALayer(out_channels)
        self.out = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size,
                      padding=(kernel_size // 2), bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.down = down
        if self.down:
            self.down_sample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)


    def forward(self, x):
        conv_pre = self.conv(x)
        conv = self.conv1(conv_pre)

        c_att = self.c_att(conv)
        s_att = self.s_att(conv)
        out = torch.cat([s_att,c_att], 1)
        conv = conv_pre + self.out(out)


        if self.down:
            down = self.down_sample(conv)
            return conv, down
        else:
            return conv

#Decoder
class convupblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(convupblock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv = nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size // 2), bias=False)
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size // 2), bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size,
                          padding=(kernel_size // 2), bias=False),
                nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, skip):
        x = self.up_sample(x)
        x = self.dconv(x)
        conv = self.conv(torch.cat([x, skip], 1))

        return conv

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, h, w):
        return self.fn(self.norm(x), h, w)


class C_FF(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False, stride=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0, bias=False, stride=1, groups=1),
            nn.GELU(),
        )
        self.out_linear = nn.Linear(hidden_dim, dim)
    def forward(self, x, h, w):
        b = x.shape[0]
        x = self.net(x)
        x = x.transpose(1, 2).contiguous().view(b, self.hidden_dim, h, w)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.out_linear(x)
        return x

class C_MSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        self.inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False)
        self.conv_q = nn.Conv2d(self.inner_dim, self.inner_dim, 3, padding=1, bias=False, stride=1, groups=heads)
        self.conv_k = nn.Conv2d(self.inner_dim, self.inner_dim, 3, padding=1, bias=False, stride=1, groups=heads)
        self.conv_v = nn.Conv2d(self.inner_dim, self.inner_dim, 3, padding=1, bias=False, stride=1, groups=heads)

        self.to_out = nn.Linear(self.inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x, h, w):
        b = x.shape[0]
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q = q.transpose(1, 2).contiguous().view(b, self.inner_dim, h, w)
        q = self.conv_q(q)
        q = q.flatten(2).transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous().view(b, self.inner_dim, h, w)
        k = self.conv_k(k)
        k = k.flatten(2).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous().view(b, self.inner_dim, h, w)
        v = self.conv_v(v)
        v = v.flatten(2).transpose(1, 2).contiguous()

        qkv = (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CTB(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, C_MSA(dim, heads = heads, dim_head = dim_head)),
                PreNorm(dim, C_FF(dim, mlp_dim))
            ]))
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        for attn, ff in self.layers:
            x = attn(x,H,W) + x
            x = ff(x,H,W) + x
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return x

# Convlutional Transformer
class CFormer(nn.Module):
    def __init__(self, first=True, channel=32, num_head=8, dim_head=64, mlp_dim=512, ctb_depth=2):
        super(CFormer, self).__init__()

        self.first_channel = 3
        self.first = first
        if not self.first:
            self.transmission = conv(3, channel, 3)
            self.gate = SGM(channels=channel)
            self.first_channel = channel*2

        # ===================downsample====================
        self.down_1 = convdownblock(in_channels=self.first_channel, out_channels=channel)
        self.down_2 = convdownblock(in_channels=channel, out_channels=channel*2)
        self.down_3 = convdownblock(in_channels=channel*2, out_channels=channel*4)
        # ===================bottleneck====================
        self.c = conv(channel*4, channel*8, 3)
        self.former = CTB(channel*8, heads=num_head, dim_head=dim_head, mlp_dim=mlp_dim, depth=ctb_depth)
        # ====================upsample=====================
        self.up_4 = convupblock(in_channels=channel*8, out_channels=channel*4)
        self.up_5 = convupblock(in_channels=channel*4, out_channels=channel*2)
        self.up_6 = convupblock(in_channels=channel*2, out_channels=channel)
        self.up_7 = conv(channel, 3, 3)
        self.leak = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, input, pre_f=None):
        N_input = input

        if not self.first and pre_f != None:
            pre_f = self.gate(pre_f)
            conv = self.transmission(input)
            N_input = torch.cat([conv, pre_f], 1)

        conv1, down1 = self.down_1(N_input)
        conv2, down2 = self.down_2(down1)
        conv3, down3 = self.down_3(down2)

        con = self.leak(self.c(down3))
        conv4 = self.former(con)

        conv5 = self.up_4(conv4, conv3)
        conv6 = self.up_5(conv5, conv2)
        conv7 = self.up_6(conv6, conv1)
        rain = self.up_7(conv7)
        output = input - rain

        return output, conv7


#Progressive Convlutional Transformer
class PCFormer(nn.Module):
    def __init__(self, channel=32, num_head=8, dim_head=64, mlp_dim=512, ctb_depth=2):
        super(PCFormer, self).__init__()

        self.cformer_1 = CFormer(first=True,  channel=channel, num_head=num_head, dim_head=dim_head, mlp_dim=mlp_dim, ctb_depth=ctb_depth)
        self.cformer_2 = CFormer(first=False, channel=channel, num_head=num_head, dim_head=dim_head, mlp_dim=mlp_dim, ctb_depth=ctb_depth)
        self.cformer_3 = CFormer(first=False, channel=channel, num_head=num_head, dim_head=dim_head, mlp_dim=mlp_dim, ctb_depth=ctb_depth)
        self.cformer_4 = CFormer(first=False, channel=channel, num_head=num_head, dim_head=dim_head, mlp_dim=mlp_dim, ctb_depth=ctb_depth)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)

    def forward(self, input):
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)

        output_1, transmission = self.cformer_1(input)
        output_2, transmission = self.cformer_2(output_1, transmission)
        output_3, transmission = self.cformer_3(output_2, transmission)
        output_4, _ = self.cformer_4(output_3, transmission)


        output_1 = pad_tensor_back(output_1, pad_left, pad_right, pad_top, pad_bottom)
        output_2 = pad_tensor_back(output_2, pad_left, pad_right, pad_top, pad_bottom)
        output_3 = pad_tensor_back(output_3, pad_left, pad_right, pad_top, pad_bottom)
        output_4 = pad_tensor_back(output_4, pad_left, pad_right, pad_top, pad_bottom)


        return output_1, output_2, output_3, output_4


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    input_size = 256
    model = PCFormer(channel=32).cuda()

    macs, params = get_model_complexity_info(model, (3, input_size, input_size), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    '''hw * hw * num_heads * head_dims * 2 * num_ctb / 1e9  +
       channel * channel * h * w * 2 * num_sgm / 1e9'''
    att_flops = 32*32*32*32*8*64*2*8/1e9 + 32*32*256*256*6/1e9

    macs = float(macs.split(' ')[0]) + att_flops
    print('{:<30}  {:<5} G'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

