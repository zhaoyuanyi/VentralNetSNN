"""Training losses: Charbonnier, and a VGG16 perceptual loss (first 16 layers)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


# 感知损失：使用 VGG16 前层特征约束输出与 GT 的语义一致性
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        self.features = nn.Sequential(*list(vgg.children()))
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.mse_loss(self.features(pred), self.features(target))


# 平滑 L1（Charbonnier）损失，鲁棒于局部异常值
def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))
