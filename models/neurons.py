"""Leaky integrate-and-fire neurons with learnable firing thresholds."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate


class LearnableThresholdLIFNode(neuron.ParametricLIFNode):
    """
    可学习阈值的参数化LIF神经元

    继承自 ParametricLIFNode，在其基础上增加可学习的阈值参数
    - tau: 可学习（继承自 ParametricLIFNode，通过 self.w 参数控制）
    - v_threshold: 可学习（本类新增，通过 self.v_threshold_raw 参数控制）

    使用 softplus 变换确保阈值始终为正值
    """

    def __init__(self, init_tau: float = 2.0, init_v_threshold: float = 1.0,
                 decay_input: bool = True, v_reset: float = 0.,
                 surrogate_function=surrogate.ATan(),
                 detach_reset: bool = True, step_mode='m', backend='torch',
                 store_v_seq: bool = False,
                 v_threshold_min: float = 0.1):
        """
        Args:
            init_tau: 初始时间常数
            init_v_threshold: 初始阈值
            decay_input: 输入是否参与衰减
            v_reset: 重置电压
            surrogate_function: 代理梯度函数
            detach_reset: 是否在reset时detach
            step_mode: 's'单步, 'm'多步
            backend: 后端
            store_v_seq: 是否存储电压序列
            v_threshold_min: 阈值最小值，确保稳定性
        """
        # 调用父类初始化（ParametricLIFNode）
        super().__init__(
            init_tau=init_tau,
            decay_input=decay_input,
            v_threshold=init_v_threshold,  # 父类会存储这个值，但我们会用可学习参数覆盖
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq
        )

        self.v_threshold_min = v_threshold_min
        self.init_v_threshold = init_v_threshold

        # 创建可学习的阈值参数
        # 使用 softplus 的逆函数初始化，使得 softplus(v_threshold_raw) + min ≈ init_v_threshold
        init_raw = self._inverse_softplus(max(init_v_threshold - v_threshold_min, 0.01))
        self.v_threshold_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @staticmethod
    def _inverse_softplus(x, beta=1.0):
        """softplus的逆函数: y = log(exp(x) - 1) / beta"""
        if x * beta > 20:
            return x
        return math.log(math.expm1(beta * x)) / beta

    @property
    def v_threshold(self):
        """获取可学习的阈值（始终为正，最小为 v_threshold_min）"""
        return F.softplus(self.v_threshold_raw) + self.v_threshold_min

    @v_threshold.setter
    def v_threshold(self, value):
        # 忽略父类的赋值，我们使用可学习参数
        pass

    def neuronal_fire(self):
        """重写fire方法，使用可学习阈值"""
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        """重写reset方法，使用可学习阈值"""
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d * self.v_threshold
        else:
            # hard reset
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / torch.sigmoid(self.w)
            v_th = self.v_threshold.item() if isinstance(self.v_threshold, torch.Tensor) else self.v_threshold
        return (f'tau={tau.item():.3f}, v_threshold={v_th:.3f}, '
                f'v_reset={self.v_reset}, detach_reset={self.detach_reset}, '
                f'step_mode={self.step_mode}, backend={self.backend}')


# 工厂函数：按开关创建“可学习阈值”或“固定阈值”的 LIF 神经元
def create_lif(tau=2.0, v_threshold=1.0, learnable_threshold=True):
    """
    创建LIF神经元

    Args:
        tau: 时间常数（始终可学习，通过ParametricLIFNode）
        v_threshold: 阈值
        learnable_threshold: 是否使用可学习阈值

    Returns:
        LIF神经元模块
    """
    if learnable_threshold:
        return LearnableThresholdLIFNode(
            init_tau=tau,
            init_v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode='m',
            backend='torch'
        )
    else:
        return neuron.ParametricLIFNode(
            init_tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode='m',
            backend='torch'
        )
