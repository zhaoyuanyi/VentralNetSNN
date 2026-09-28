"""LGN front-end: difference-of-Gaussians centre-surround filtering and magno/parvo streams."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer

from .layers import ConvBN, ConvBNLIF
from .neurons import create_lif


# 差分高斯（DoG）增强层，用于模拟中心-周围响应
class DoGLayerStable(nn.Module):

    def __init__(self, channels, sigma_center=0.8, sigma_surround=2.4,
                 amplitude_center=1.0, amplitude_surround=0.7, init_mix=0.1):
        super().__init__()
        self.channels = channels
        self.sigma_c = sigma_center
        self.sigma_s = sigma_surround
        self.amp_c = amplitude_center
        self.amp_s = amplitude_surround

        self.k_center = max(int(sigma_center * 4) | 1, 3)
        self.k_surround = max(int(sigma_surround * 4) | 1, 5)
        self.pad_c = self.k_center // 2
        self.pad_s = self.k_surround // 2

        init_logit = torch.log(torch.tensor(init_mix / (1 - init_mix + 1e-6)))
        self.mix_logit = nn.Parameter(init_logit)
        self._build_kernels()

    def _build_kernels(self):
        coords_c = torch.arange(self.k_center, dtype=torch.float32) - self.k_center // 2
        g_c = torch.exp(-(coords_c ** 2) / (2 * self.sigma_c ** 2))
        g_c = g_c / g_c.sum()
        kernel_c = (g_c.view(-1, 1) @ g_c.view(1, -1)).view(1, 1, self.k_center, self.k_center)
        self.register_buffer('kernel_c', kernel_c)

        coords_s = torch.arange(self.k_surround, dtype=torch.float32) - self.k_surround // 2
        g_s = torch.exp(-(coords_s ** 2) / (2 * self.sigma_s ** 2))
        g_s = g_s / g_s.sum()
        kernel_s = (g_s.view(-1, 1) @ g_s.view(1, -1)).view(1, 1, self.k_surround, self.k_surround)
        self.register_buffer('kernel_s', kernel_s)

    def forward(self, x):
        T, N, C, H, W = x.shape
        x_flat = x.flatten(0, 1)
        kernel_c = self.kernel_c.repeat(C, 1, 1, 1)
        kernel_s = self.kernel_s.repeat(C, 1, 1, 1)
        center = F.conv2d(x_flat, kernel_c, padding=self.pad_c, groups=C)
        surround = F.conv2d(x_flat, kernel_s, padding=self.pad_s, groups=C)
        dog = self.amp_c * center - self.amp_s * surround
        dog = dog.view(T, N, C, H, W)
        mix = torch.sigmoid(self.mix_logit)
        return x * (1 - mix) + dog * mix


# 可混合的高斯平滑层
class GaussianBlurStable(nn.Module):

    def __init__(self, channels, kernel_size=5, sigma=1.5, init_mix=0.3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = (g.view(-1, 1) @ g.view(1, -1)).view(1, 1, kernel_size, kernel_size)
        self.register_buffer('kernel', kernel)

        init_logit = torch.log(torch.tensor(init_mix / (1 - init_mix + 1e-6)))
        self.mix_logit = nn.Parameter(init_logit)

    def forward(self, x):
        T, N, C, H, W = x.shape
        x_flat = x.flatten(0, 1)
        kernel = self.kernel.repeat(C, 1, 1, 1)
        blurred = F.conv2d(x_flat, kernel, padding=self.padding, groups=C)
        blurred = blurred.view(T, N, C, H, W)
        mix = torch.sigmoid(self.mix_logit)
        return x * (1 - mix) + blurred * mix


# LGN 模块：构建 M/P 两条早期视觉通路并融合出 skip 特征
class LGNModuleStable(nn.Module):
    """稳定版LGN模块"""

    def __init__(self, in_ch=3, out_ch=32, tau=2.0, v_threshold=1.0, use_luminance_for_m=True,
                 learnable_threshold=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_luminance_for_m = use_luminance_for_m
        self.learnable_threshold = learnable_threshold

        tau_m = max(float(tau) * 0.8, 1.2)
        tau_p = max(float(tau) * 1.5, 1.5)

        if use_luminance_for_m:
            self.register_buffer('rgb_to_y', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
            m_in_ch = 1
        else:
            m_in_ch = in_ch

        self.m_blur = GaussianBlurStable(m_in_ch, kernel_size=5, sigma=1.5, init_mix=0.2)
        self.m_dog = DoGLayerStable(m_in_ch, sigma_center=1.2, sigma_surround=3.5,
                                    amplitude_center=1.0, amplitude_surround=0.5, init_mix=0.1)

        self.p_dog = DoGLayerStable(in_ch, sigma_center=0.5, sigma_surround=1.5,
                                    amplitude_center=1.0, amplitude_surround=0.6, init_mix=0.1)

        self.m_conv1 = ConvBNLIF(m_in_ch, out_ch, k=5, s=1, p=2, tau=tau_m, v_threshold=v_threshold,
                                 learnable_threshold=learnable_threshold)
        self.m_conv2 = ConvBN(out_ch, out_ch, k=3, s=1, p=1)
        self.m_res = ConvBN(m_in_ch, out_ch, k=1, s=1, p=0)
        self.m_lif = create_lif(tau_m, v_threshold, learnable_threshold)

        self.p_conv1 = ConvBNLIF(in_ch, out_ch, k=3, s=1, p=1, tau=tau_p, v_threshold=v_threshold,
                                 learnable_threshold=learnable_threshold)
        self.p_conv2 = ConvBN(out_ch, out_ch, k=3, s=1, p=1)
        self.p_res = ConvBN(in_ch, out_ch, k=1, s=1, p=0)
        self.p_lif = create_lif(tau_p, v_threshold, learnable_threshold)

        self.gate_strength = nn.Parameter(torch.tensor(-3.0))

        self.skip_fuse = nn.Sequential(
            layer.Conv2d(out_ch * 2, out_ch, 3, 1, 1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_ch, step_mode='m')
        )

        self.direct_proj = ConvBN(in_ch, out_ch, k=1, s=1, p=0)
        self.skip_weight = nn.Parameter(torch.tensor(0.85))

        self.global_pool_lgn = nn.AdaptiveAvgPool2d(1)
        reduction = max(4, out_ch // 32)
        self.global_fc_lgn = nn.Sequential(
            nn.Conv2d(out_ch, reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, out_ch, 1, bias=False),
            nn.Sigmoid()
        )

    def _rgb_to_luminance(self, x):
        T, N, C, H, W = x.shape
        x_flat = x.flatten(0, 1)
        y = (x_flat * self.rgb_to_y).sum(dim=1, keepdim=True)
        return y.view(T, N, 1, H, W)

    def forward(self, x):
        direct = self.direct_proj(x)

        if self.use_luminance_for_m:
            x_m = self._rgb_to_luminance(x)
        else:
            x_m = x

        x_m_processed = self.m_blur(x_m)
        x_m_processed = self.m_dog(x_m_processed)

        m = self.m_conv1(x_m_processed)
        m_membrane = self.m_conv2(m) + self.m_res(x_m)

        x_p_processed = self.p_dog(x)

        p = self.p_conv1(x_p_processed)
        p_membrane = self.p_conv2(p) + self.p_res(x)

        # 通过可学习门控做 M/P 轻度信息交换（detach 防止强耦合梯度）
        gate_s = torch.sigmoid(self.gate_strength)

        m_enhanced = m_membrane * (1.0 - 0.2 * gate_s) + p_membrane.detach() * (0.2 * gate_s)
        p_enhanced = p_membrane * (1.0 - 0.2 * gate_s) + m_membrane.detach() * (0.2 * gate_s)

        fused = self.skip_fuse(torch.cat([m_enhanced, p_enhanced], dim=2))

        # skip 兼顾原始直连信息与融合信息
        skip_w = torch.sigmoid(self.skip_weight)
        skip = fused * (1 - skip_w) + direct * skip_w

        T, N, C, H, W = skip.shape
        skip_flat = skip.flatten(0, 1)
        ctx = self.global_fc_lgn(self.global_pool_lgn(skip_flat))
        ctx = ctx.view(T, N, C, 1, 1)
        skip = skip * ctx

        F_M = self.m_lif(m_enhanced)
        F_P = self.p_lif(p_enhanced)

        return F_M, F_P, skip
