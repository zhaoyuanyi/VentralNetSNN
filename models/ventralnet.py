"""VentralNetSNN: the full ventral-pathway-inspired restoration network."""

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .decoder import DecoderBlock, FinalDecoder
from .encoder import DualStreamEncoder, ITModule, V4Module
from .lgn import LGNModuleStable
from .neurons import LearnableThresholdLIFNode


# 整体网络：LGN -> 双流编码 -> V4/IT -> 逐级解码
class VentralNetSNN(nn.Module):
    """SNN去雨网络（支持可学习阈值）"""

    def __init__(self, lgn_ch=16, v1_ch=32, v2_ch=64, v4_ch=128, T=4, tau=2.0, v_threshold=1.0,
                 learnable_threshold=True):
        super().__init__()
        self.T = T
        self.learnable_threshold = learnable_threshold

        self.lgn = LGNModuleStable(3, lgn_ch, tau, v_threshold, learnable_threshold=learnable_threshold)
        self.v1 = DualStreamEncoder(lgn_ch, v1_ch, tau, v_threshold, learnable_threshold=learnable_threshold)
        self.v2 = DualStreamEncoder(v1_ch, v2_ch, tau, v_threshold, learnable_threshold=learnable_threshold)
        self.v4 = V4Module(v2_ch, v4_ch, tau, v_threshold, learnable_threshold=learnable_threshold)
        self.it = ITModule(v4_ch, tau, v_threshold, learnable_threshold=learnable_threshold)
        self.dec3 = DecoderBlock(v4_ch, v4_ch, v2_ch, tau, v_threshold, learnable_threshold=learnable_threshold)
        self.dec2 = DecoderBlock(v2_ch, v2_ch, v1_ch, tau, v_threshold, learnable_threshold=learnable_threshold)
        self.dec1 = FinalDecoder(v1_ch, v1_ch, lgn_ch, tau, v_threshold, learnable_threshold=learnable_threshold)

        self.perturb_strength = 0.01

    def _build_input_sequence(self, x):
        T = self.T
        # 将静态输入复制为 T 个时间步输入
        x_seq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)

        if self.perturb_strength <= 0 or not self.training:
            return x_seq

        device = x.device
        # 训练时加入轻微强度扰动，提升时序鲁棒性
        noise = (torch.rand(T, 1, 1, 1, 1, device=device) - 0.5) * 2.0 * self.perturb_strength
        x_seq = x_seq * (1.0 + noise)
        return torch.clamp(x_seq, 0.0, 1.0)

    def forward(self, x):
        # 每次前向前重置 SNN 神经元状态，避免跨 batch 泄漏
        functional.reset_net(self)

        N, C, H, W = x.shape
        x_seq = self._build_input_sequence(x)

        # 编码阶段：LGN + 双流编码 + 高阶视觉模块
        F_M, F_P, lgn_mem = self.lgn(x_seq)
        M1, P1, skip1 = self.v1(F_M, F_P)
        M2, P2, skip2 = self.v2(M1, P1)
        v4_out, v4_down, v4_mem = self.v4(M2, P2)
        it_out = self.it(v4_down)

        # 解码阶段：逐级上采样恢复分辨率
        d3 = self.dec3(it_out, v4_mem, (H // 4, W // 4))
        d2 = self.dec2(d3, skip2, (H // 2, W // 2))
        rgb_seq = self.dec1(d2, skip1, lgn_mem, (H, W))

        # 网络输出的是残差，和输入序列相加后做时间平均
        output_seq = x_seq + rgb_seq
        output = output_seq.mean(dim=0)

        return torch.clamp(output, 0, 1)

    def get_threshold_stats(self):
        """获取所有可学习阈值的统计信息"""
        thresholds = []
        taus = []
        for name, module in self.named_modules():
            if isinstance(module, LearnableThresholdLIFNode):
                thresholds.append(module.v_threshold.item())
                tau_val = 1. / torch.sigmoid(module.w).item()
                taus.append(tau_val)
        return thresholds, taus

    def print_neuron_stats(self):
        """打印神经元参数统计"""
        thresholds, taus = self.get_threshold_stats()
        if thresholds:
            print(f"    Thresholds: avg={np.mean(thresholds):.3f}, "
                  f"min={np.min(thresholds):.3f}, max={np.max(thresholds):.3f}")
            print(f"    Taus: avg={np.mean(taus):.3f}, "
                  f"min={np.min(taus):.3f}, max={np.max(taus):.3f}")


# Backward-compatible alias used in the original training scripts.
VentralDerainNetSNN = VentralNetSNN
