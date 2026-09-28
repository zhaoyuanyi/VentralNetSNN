"""One training epoch and evaluation loop."""
import torch
import torch.nn.functional as F

from .losses import charbonnier_loss
from .metrics import psnr, ssim_metric, ssim_value


def train_one_epoch(model, loader, optimizer, device, perceptual_loss=None,
                    ssim_weight=0.2, perceptual_weight=0.1, grad_clip=1.0):
    # 单轮训练：Charbonnier + SSIM + 感知损失
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = charbonnier_loss(pred, y)
        if ssim_weight > 0:
            # 以 (1 - SSIM) 形式加入结构一致性约束
            loss = loss + ssim_weight * (1.0 - ssim_value(pred, y))
        if perceptual_weight > 0 and perceptual_loss is not None:
            loss = loss + perceptual_weight * perceptual_loss(pred, y)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, pad_to=None, per_image=False, on_output=None):
    """Average L1 / PSNR / SSIM over ``loader``.

    pad_to:     reflect-pad inputs to a multiple of this value and crop the output back
                (used for native-resolution evaluation).
    per_image:  additionally return a list of (index, l1, psnr, ssim) per batch.
    on_output:  optional callback ``on_output(index, prediction)``, e.g. to save images.
    """
    model.eval()
    l1, ps, ss = 0.0, 0.0, 0.0
    rows = []
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if pad_to:
            H, W = x.shape[-2:]
            ph = (pad_to - H % pad_to) % pad_to
            pw = (pad_to - W % pad_to) % pad_to
            pred = model(F.pad(x, (0, pw, 0, ph), mode="reflect"))[..., :H, :W] if (ph or pw) else model(x)
        else:
            pred = model(x)
        b_l1 = F.l1_loss(pred, y).item()
        b_ps = psnr(pred, y)
        b_ss = ssim_metric(pred, y)
        l1 += b_l1
        ps += b_ps
        ss += b_ss
        if per_image:
            rows.append((idx, b_l1, b_ps, b_ss))
        if on_output is not None:
            on_output(idx, pred)
    n = len(loader)
    if per_image:
        return l1 / n, ps / n, ss / n, rows
    return l1 / n, ps / n, ss / n
