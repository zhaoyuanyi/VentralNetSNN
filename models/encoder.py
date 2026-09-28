"""Dual-stream V1/V2 encoders and the V4 / IT modules."""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from .layers import CBAM_MS, ChannelAttentionMS, ConvBN, ConvBNLIF, GateMS
from .neurons import create_lif


# 增强编码块：在基础块上加入残差门控
class EncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, use_dilation=False, tau=2.0, v_threshold=1.0, learnable_threshold=True):
        super().__init__()
        d = 2 if use_dilation else 1
        p = 2 if use_dilation else 1

        self.conv1 = ConvBNLIF(in_ch, out_ch, 3, 1, 1, tau=tau, v_threshold=v_threshold,
                               learnable_threshold=learnable_threshold)
        self.conv2 = ConvBNLIF(out_ch, out_ch, 3, 1, p, d=d, tau=tau, v_threshold=v_threshold,
                               learnable_threshold=learnable_threshold)
        self.conv3 = ConvBN(out_ch, out_ch, 3, 1, 1)
        self.res = ConvBN(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else None

        # 门控残差
        self.res_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1, bias=False),
            nn.Sigmoid()
        )

        self.ca = ChannelAttentionMS(out_ch, reduction=4)
        self.lif = create_lif(tau, v_threshold, learnable_threshold)
        self.down = nn.Sequential(
            layer.Conv2d(out_ch, out_ch, 3, 2, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_ch, step_mode='m'),
            create_lif(tau, v_threshold, learnable_threshold)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        membrane = self.conv3(out)
        res = x if self.res is None else self.res(x)

        # 门控残差连接
        T, N, C, H, W = membrane.shape
        gate = self.res_gate(membrane.flatten(0, 1))
        gate = gate.view(T, N, C, 1, 1)
        membrane = membrane + res * gate

        membrane = self.ca(membrane)
        spike = self.lif(membrane)
        down = self.down(spike)
        return down, membrane


# 双流编码器：并行处理 M/P 分支并交互融合
class DualStreamEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, tau=2.0, v_threshold=1.0, learnable_threshold=True):
        super().__init__()
        self.m_block = EncoderBlock(in_ch, out_ch, use_dilation=True, tau=tau, v_threshold=v_threshold,
                                        learnable_threshold=learnable_threshold)
        self.p_block = EncoderBlock(in_ch, out_ch, use_dilation=False, tau=tau, v_threshold=v_threshold,
                                        learnable_threshold=learnable_threshold)

        self.gate_m = GateMS(out_ch)
        self.gate_p = GateMS(out_ch)

        self.skip_fuse = nn.Sequential(
            layer.Conv2d(out_ch * 2, out_ch, 3, 1, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_ch, step_mode='m')
        )
        self.skip_attn = CBAM_MS(out_ch, reduction=4)

    def forward(self, M_in, P_in):
        M_down, M_membrane = self.m_block(M_in)
        P_down, P_membrane = self.p_block(P_in)

        gate_m = self.gate_m(P_membrane)
        gate_p = self.gate_p(M_membrane)

        M_membrane_gated = M_membrane * (0.6 + 0.4 * gate_m)
        P_membrane_gated = P_membrane * (0.6 + 0.4 * gate_p)

        skip = self.skip_fuse(torch.cat([M_membrane_gated, P_membrane_gated], dim=2))
        skip = self.skip_attn(skip)

        return M_down, P_down, skip


# V4 模块：融合双流特征并继续下采样
class V4Module(nn.Module):
    def __init__(self, in_ch, out_ch, tau=2.0, v_threshold=1.0, learnable_threshold=True):
        super().__init__()
        self.fuse_conv = ConvBNLIF(in_ch * 2, out_ch, 3, 1, 1, tau=tau, v_threshold=v_threshold,
                                   learnable_threshold=learnable_threshold)
        self.branch1 = ConvBN(out_ch, out_ch // 2, 1, 1, 0)
        self.branch3 = ConvBN(out_ch, out_ch // 2, 3, 1, 1)
        self.conv = ConvBN(out_ch, out_ch, 3, 1, 1)
        self.res = ConvBN(in_ch * 2, out_ch, 1, 1, 0)
        self.attn = CBAM_MS(out_ch, reduction=4)
        self.lif = create_lif(tau, v_threshold, learnable_threshold)
        self.down = nn.Sequential(
            layer.Conv2d(out_ch, out_ch, 3, 2, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_ch, step_mode='m'),
            create_lif(tau, v_threshold, learnable_threshold)
        )

    def forward(self, M, P):
        x = torch.cat([M, P], dim=2)
        out = self.fuse_conv(x)
        b1 = self.branch1(out)
        b3 = self.branch3(out)
        out = torch.cat([b1, b3], dim=2)
        membrane = self.conv(out) + self.res(x)
        membrane = self.attn(membrane)
        spike = self.lif(membrane)
        down = self.down(spike)
        return spike, down, membrane


# IT 模块：多尺度深度可分卷积 + 全局上下文调制
class ITModule(nn.Module):
    def __init__(self, ch, tau=2.0, v_threshold=1.0, learnable_threshold=True):
        super().__init__()
        self.large_kernel = nn.Sequential(
            layer.Conv2d(ch, ch, (7, 1), 1, (3, 0), groups=ch, bias=False, step_mode='m'),
            layer.BatchNorm2d(ch, step_mode='m'),
            layer.Conv2d(ch, ch, (1, 7), 1, (0, 3), groups=ch, bias=False, step_mode='m'),
            layer.BatchNorm2d(ch, step_mode='m'),
        )
        self.medium_kernel = nn.Sequential(
            layer.Conv2d(ch, ch, (5, 1), 1, (2, 0), groups=ch, bias=False, step_mode='m'),
            layer.BatchNorm2d(ch, step_mode='m'),
            layer.Conv2d(ch, ch, (1, 5), 1, (0, 2), groups=ch, bias=False, step_mode='m'),
            layer.BatchNorm2d(ch, step_mode='m'),
        )
        self.small_kernel = nn.Sequential(
            layer.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False, step_mode='m'),
            layer.BatchNorm2d(ch, step_mode='m')
        )
        self.fuse = nn.Sequential(
            layer.Conv2d(ch * 3, ch, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(ch, step_mode='m')
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, ch, 1, bias=False),
            nn.Sigmoid()
        )
        self.lif = create_lif(tau, v_threshold, learnable_threshold)

    def forward(self, x):
        large = self.large_kernel(x)
        medium = self.medium_kernel(x)
        small = self.small_kernel(x)
        multi_scale = self.fuse(torch.cat([large, medium, small], dim=2))
        membrane = multi_scale + x

        T, N, C, H, W = membrane.shape
        mem_flat = membrane.flatten(0, 1)
        ctx = self.global_fc(self.global_pool(mem_flat))
        ctx = ctx.view(T, N, C, 1, 1)
        membrane = membrane * ctx

        return self.lif(membrane)
