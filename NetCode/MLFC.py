import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Sequence
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule, constant_init
from ultralytics.nn.modules.conv import Conv,autopad
from ultralytics.nn.modules.conv import Conv, DWConv

__all__ = ['MLFC']


class CRM_1(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.gai22 = CGA(c1)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.gai22(x) + x + id_out

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)

        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)

        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 1,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class CRM_mainbranch(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = CRM_1(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.SCConv = ScConv(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.SCConv(self.cv1(x))) if self.add else self.cv2(self.cv1(x))


class CRM(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(CRM_mainbranch(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return x + self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


def autopad(k, p=None, d=1):

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class CGA(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 1, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = in_channels

        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=1, norm_cfg=norm_cfg, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                  autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
                                  groups=1, norm_cfg=norm_cfg, act_cfg=None)
        self.h = 8
        self.w = 8

        self.channel2x1 = nn.Sequential(
            *[nn.Conv2d(hidden_channels, hidden_channels,1, groups=1)],
        )
        self.silu = nn.SiLU()
        
        self.post_conv2 = ConvModule(4 *hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.cv1 = nn.Conv2d(2, 1, 7, padding=3, bias=True)
        self.cv2 = Conv(hidden_channels, hidden_channels, 1, 1)
 
    def forward(self, x):
        y = self.silu(self.dw_conv(x) + self.dw_conv3(x))
        z = self.cv2(self.dw_conv1(x)) + self.cv2(self.dw_conv2(x))
        yz = y + z
        y1 = z * torch.sigmoid(self.cv1(torch.cat([torch.mean(y, 1, keepdim=True), torch.max(y, 1, keepdim=True)[0]], 1)))
        n, c, _, _ = z.size()

        amaxp = F.adaptive_max_pool2d(z, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(z, output_size=(self.h, self.w))

        amaxp = torch.sum(self.silu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.silu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel2x1(amaxp) + self.channel2x1(aavgp)
        z1 = y * torch.sigmoid(channel)

        return self.post_conv2(torch.cat((x, yz, y1, z1), 1))


class MLFC(nn.Module):
    def __init__(self, c1, c2, c5=1):
        super().__init__()
        c3 = c2
        self.c4 = int(c3 / 2)
        self.c6 = int(self.c4 / 2)
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv5 = nn.Sequential(CRM(self.c4 // 2, self.c6, c5), Conv(self.c6, self.c6, 3, 1))
        self.cv2 = nn.Sequential(CRM(self.c4, self.c4, c5), Conv(self.c4, self.c4, 3, 1))
        self.cv3 = nn.Sequential(CRM(self.c4, self.c4, c5), Conv(self.c4, self.c4, 3, 1))
        self.cv4 = Conv(self.c6 + self.c6 + self.c6 + self.c6 + self.c4 + (2 * self.c4), c2, 1, 1)
        self.cv6 = CGA(self.c6)
        self.cv7 = Conv(self.c4, self.c4, 3, 1, g=2)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c6, self.c6, self.c4), 1))
        processed6 = self.cv6(y[0])
        y.append(processed6)
        processed5 = self.cv5(y[1])
        y.append(processed5)
        processed7 = self.cv7(torch.cat((processed6, processed5), 1))
        processed2 = self.cv2(y[2])
        processed3 = self.cv3(processed2 + processed7)
        y.append(processed2)
        y.append(processed3)
        return self.cv4(torch.cat(y, 1))

if __name__ == "__main__":

    image_size = (1, 24, 224, 224)
    image = torch.rand(*image_size)

    mobilenet_v1 = MLFC(24, 24)

    out = mobilenet_v1(image)
    print(out.size())