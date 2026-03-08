import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors
import math
import torch.nn.functional as F
from typing import Optional, Union, Sequence
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule, constant_init
from ultralytics.nn.modules.conv import Conv,autopad

__all__ = ['CGAHead']

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

class DFL(nn.Module):

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class ASFFV5(nn.Module):
    def __init__(self, level, ch, multiplier=1, rfb=False, vis=False, act_cfg=True):
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [int(ch[3] * multiplier), int(ch[2] * multiplier), int(ch[1] * multiplier),
                    int(ch[0] * multiplier)]

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(ch[2] * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)

            self.stride_level_3 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                ch[3] * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(ch[3] * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(ch[1] * multiplier), self.inter_dim, 3, 2)
            self.stride_level_3 = Conv(
                int(ch[0] * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(ch[2] * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(ch[3] * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(ch[2] * multiplier), self.inter_dim, 1, 1)
            self.stride_level_3 = Conv(
                int(ch[0] * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(ch[1] * multiplier), 3, 1)
        elif level == 3:
            self.compress_level_0 = Conv(
                int(ch[3] * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(ch[2] * multiplier), self.inter_dim, 1, 1)
            self.compress_level_2 = Conv(
                int(ch[1] * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(ch[0] * multiplier), 3, 1)

        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 4, 4, 1, 1)
        self.vis = vis

    def forward(self, x):
        x_level_0 = x[3]
        x_level_1 = x[2]
        x_level_2 = x[1]
        x_level_3 = x[0]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
            level_3_downsampled_inter1 = F.max_pool2d(x_level_3, 3, stride=2, padding=1)
            level_3_downsampled_inter2 = F.max_pool2d(level_3_downsampled_inter1, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter2)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_downsampled_inter = F.max_pool2d(x_level_3, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
            level_3_resized = self.stride_level_3(x_level_3)
        elif self.level == 3:  # 新增level=3的resize逻辑（上采样3路到最小尺度）
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=8, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=4, mode='nearest')
            level_2_compressed = self.compress_level_2(x_level_2)
            level_2_resized = F.interpolate(level_2_compressed, scale_factor=2, mode='nearest')
            level_3_resized = x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:3, :, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
    
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

class CGAHead(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=(), multiplier=1, rfb=False):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.l0_fusion = ASFFV5(level=0, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l1_fusion = ASFFV5(level=1, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l2_fusion = ASFFV5(level=2, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l3_fusion = ASFFV5(level=3, ch=ch, multiplier=multiplier, rfb=rfb)
        self.gai25_list = nn.ModuleList([
            CGA(
                in_channels=ch[i],
            ) for i in range(self.nl)  
        ])

    def forward(self, x):
        x = [self.gai25_list[i](xi) for i, xi in enumerate(x)]
        x1 = self.l0_fusion(x)
        x2 = self.l1_fusion(x)
        x3 = self.l2_fusion(x)
        x4 = self.l3_fusion(x)
        x = [x4, x3, x2, x1]
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


if __name__ == "__main__":

    image1 = (1, 64, 32, 32)
    image2 = (1, 128, 16, 16)
    image3 = (1, 256, 8, 8)

    image1 = torch.rand(image1)
    image2 = torch.rand(image2)
    image3 = torch.rand(image3)
    image = [image1, image2, image3]
    channel = (64, 128, 256)

    mobilenet_v1 = CGAHead(nc=80, ch=channel)

    out = mobilenet_v1(image)
    print(out)