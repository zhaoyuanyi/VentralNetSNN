"""Evaluation metrics (PSNR / SSIM). ``ssim_value`` is also used by the SSIM loss."""
import torch
import torch.nn.functional as F


# 计算 SSIM 值（输入形状为 [N, C, H, W]）
def ssim_value(img1, img2, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g.view(-1, 1) @ g.view(1, -1)).view(1, 1, window_size, window_size)
    window = window.repeat(img1.size(1), 1, 1, 1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=pad, groups=img2.size(1))
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=img1.size(1)) - mu1 ** 2
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=img2.size(1)) - mu2 ** 2
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=img1.size(1)) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# 常用 PSNR 指标
@torch.no_grad()
def psnr(pred, target):
    mse = F.mse_loss(pred, target).clamp_min(1e-10)
    return float(10.0 * torch.log10(1.0 / mse))


# 封装 SSIM 指标为 float，便于日志打印
@torch.no_grad()
def ssim_metric(img1, img2):
    return float(ssim_value(img1, img2).item())
