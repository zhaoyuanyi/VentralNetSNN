"""Multi-scale spiking decoder."""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from .layers import CBAM_MS, ConvBN, ConvBNLIF, ms_interpolate
from .neurons import create_lif


# 解码块：上采样 + skip 融合 + 多尺度细化
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, tau=2.0, v_threshold=1.0, learnable_threshold=True):
        super().__init__()
        self.up_proj = ConvBN(in_ch, out_ch, 1, 1, 0)
        self.skip_proj = ConvBN(skip_ch, out_ch, 1, 1, 0)
        self.fuse = nn.Sequential(
            layer.Conv2d(out_ch * 2, out_ch, 3, 1, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_ch, step_mode='m')
        )
        self.branch3 = nn.Sequential(
            layer.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_ch, step_mode='m')
        )
        self.branch5 = nn.Sequential(
            layer.Conv2d(out_ch, out_ch, 3, 1, 2, dilation=2, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_ch, step_mode='m')
        )
        self.scale_fuse = nn.Sequential(
            layer.Conv2d(out_ch * 2, out_ch, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_ch, step_mode='m')
        )
        self.refine1 = ConvBNLIF(out_ch, out_ch, 3, 1, 1, tau=tau, v_threshold=v_threshold,
                                 learnable_threshold=learnable_threshold)
        self.refine2 = ConvBN(out_ch, out_ch, 3, 1, 1)
        self.attn = CBAM_MS(out_ch, reduction=4, kernel_size=5)
        self.lif = create_lif(tau, v_threshold, learnable_threshold)

    def forward(self, x, skip, target_size):
        x = ms_interpolate(x, target_size)
        if skip.shape[3:] != tuple(target_size):
            skip = ms_interpolate(skip, target_size)

        up_proj = self.up_proj(x)
        skip_proj = self.skip_proj(skip)
        fused = self.fuse(torch.cat([up_proj, skip_proj], dim=2))

        b3 = self.branch3(fused)
        b5 = self.branch5(fused)
        ms = self.scale_fuse(torch.cat([b3, b5], dim=2))
        ms = ms + fused

        out = self.refine1(ms)
        out = self.refine2(out) + ms
        out = self.attn(out)

        return self.lif(out)


# 最终解码头：融合 LGN skip 并输出 RGB 残差序列
class FinalDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, lgn_ch, tau=2.0, v_threshold=1.0, learnable_threshold=True):
        super().__init__()
        mid = max(in_ch, 48)
        self.up_proj = ConvBN(in_ch, mid, 1, 1, 0)
        self.skip_proj = ConvBN(skip_ch, mid, 1, 1, 0)
        self.lgn_proj = ConvBN(lgn_ch, mid, 1, 1, 0)
        self.fuse = nn.Sequential(
            layer.Conv2d(mid * 3, mid, 3, 1, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(mid, step_mode='m')
        )
        self.branch3 = ConvBN(mid, mid, 3, 1, 1)
        self.branch5 = ConvBN(mid, mid, 3, 1, 2, d=2)
        self.scale_fuse = nn.Sequential(
            layer.Conv2d(mid * 2, mid, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(mid, step_mode='m')
        )
        self.refine1 = ConvBNLIF(mid, mid, 3, 1, 1, tau=tau, v_threshold=v_threshold,
                                 learnable_threshold=learnable_threshold)
        self.refine2 = ConvBN(mid, mid, 3, 1, 1)
        self.attn = CBAM_MS(mid, reduction=4, kernel_size=5)

        self.to_rgb = nn.Sequential(
            layer.Conv2d(mid, mid // 2, 3, 1, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(mid // 2, step_mode='m'),
        )
        self.to_rgb_act = nn.ReLU(inplace=False)
        self.to_rgb2 = nn.Sequential(
            layer.Conv2d(mid // 2, mid // 4, 3, 1, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(mid // 4, step_mode='m'),
        )
        self.to_rgb_act2 = nn.ReLU(inplace=False)
        self.to_rgb_final = layer.Conv2d(mid // 4, 3, 3, 1, 1, step_mode='m')

    def forward(self, x, skip, lgn_mem, target_size):
        x = ms_interpolate(x, target_size)
        if skip.shape[3:] != tuple(target_size):
            skip = ms_interpolate(skip, target_size)
        if lgn_mem.shape[3:] != tuple(target_size):
            lgn_mem = ms_interpolate(lgn_mem, target_size)

        up_feat = self.up_proj(x)
        skip_feat = self.skip_proj(skip)
        lgn_feat = self.lgn_proj(lgn_mem)
        fused = self.fuse(torch.cat([up_feat, skip_feat, lgn_feat], dim=2))

        b3 = self.branch3(fused)
        b5 = self.branch5(fused)
        ms = self.scale_fuse(torch.cat([b3, b5], dim=2))
        ms = ms + fused

        out = self.refine1(ms)
        out = self.refine2(out) + ms
        out = self.attn(out)

        out = self.to_rgb(out)
        T, N, C, H, W = out.shape
        out = out.flatten(0, 1)
        out = self.to_rgb_act(out)
        out = out.view(T, N, -1, H, W)

        out = self.to_rgb2(out)
        T, N, C, H, W = out.shape
        out = out.flatten(0, 1)
        out = self.to_rgb_act2(out)
        out = out.view(T, N, -1, H, W)

        out = self.to_rgb_final(out)
        return out
