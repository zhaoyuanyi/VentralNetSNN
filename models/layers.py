"""Multi-step convolution, attention and gating layers shared across the network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer

from .neurons import create_lif


# 多步张量插值工具：对 [T, N, C, H, W] 做空间尺度变换
def ms_interpolate(x, size, mode='bilinear'):
    """对多步张量进行插值 [T, N, C, H, W]"""
    T, N, C, H, W = x.shape
    x = x.flatten(0, 1)
    x = F.interpolate(x, size, mode=mode, align_corners=False if mode == 'bilinear' else None)
    x = x.view(T, N, C, size[0], size[1])
    return x


# 基础时序卷积块：Conv2d + BatchNorm2d（均为 step_mode='m'）
class ConvBN(nn.Module):
    """Conv + BN，支持多步模式"""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, g=1):
        super().__init__()
        self.conv = layer.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=g, bias=False, step_mode='m')
        self.bn = layer.BatchNorm2d(out_ch, step_mode='m')

    def forward(self, x):
        return self.bn(self.conv(x))


# 带脉冲激活的卷积块：Conv + BN + LIF
class ConvBNLIF(nn.Module):
    """Conv + BN + LIF，支持多步模式和可学习阈值"""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, g=1, tau=2.0, v_threshold=1.0, learnable_threshold=True):
        super().__init__()
        self.conv = layer.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=g, bias=False, step_mode='m')
        self.bn = layer.BatchNorm2d(out_ch, step_mode='m')
        self.lif = create_lif(tau, v_threshold, learnable_threshold)

    def forward(self, x):
        return self.lif(self.bn(self.conv(x)))


# 通道注意力（多步时序版）
class ChannelAttentionMS(nn.Module):
    """通道注意力，支持多步模式"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        T, N, C, H, W = x.shape
        x_flat = x.flatten(0, 1)
        avg_out = self.fc(self.avg_pool(x_flat))
        max_out = self.fc(self.max_pool(x_flat))
        attn = self.sigmoid(avg_out + max_out)
        attn = attn.view(T, N, C, 1, 1)
        return x * attn


# 空间注意力（多步时序版）
class SpatialAttentionMS(nn.Module):
    """空间注意力，支持多步模式"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        T, N, C, H, W = x.shape
        x_flat = x.flatten(0, 1)
        avg_out = torch.mean(x_flat, dim=1, keepdim=True)
        max_out, _ = torch.max(x_flat, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        attn = attn.view(T, N, 1, H, W)
        return x * attn


# 组合通道 + 空间注意力
class CBAM_MS(nn.Module):
    """CBAM，支持多步模式"""

    def __init__(self, channels, reduction=4, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttentionMS(channels, reduction)
        self.sa = SpatialAttentionMS(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


# 门控分支：生成 [0,1] 的逐位置权重
class GateMS(nn.Module):
    """门控模块"""

    def __init__(self, channels):
        super().__init__()
        self.conv = layer.Conv2d(channels, channels, 1, bias=False, step_mode='m')
        self.bn = layer.BatchNorm2d(channels, step_mode='m')

    def forward(self, x):
        return torch.sigmoid(self.bn(self.conv(x)))
