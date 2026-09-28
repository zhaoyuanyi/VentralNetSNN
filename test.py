"""Evaluate a trained VentralNetSNN checkpoint.

Examples:
    python test.py --config configs/its.yaml --checkpoint results/its/seed_42/best.pth --measure_latency
    python test.py --config configs/its.yaml --checkpoint results/its/seed_42/best.pth --eval_resolution native
"""
import argparse
import os
import sys
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import DATASET_CONFIGS, PairedImageDataset, build_dataset_pairs, print_expected_structure
from models import LearnableThresholdLIFNode, VentralNetSNN
from utils.common import environment_info, parse_args_with_config, save_json, set_seed, sha256_file, write_per_image_csv
from utils.engine import evaluate


def get_parser():
    p = argparse.ArgumentParser(description="Evaluate VentralNetSNN")
    p.add_argument("--config", type=str, default=None, help="YAML file whose values become argument defaults")
    p.add_argument("--dataset", type=str, default=None, choices=sorted(DATASET_CONFIGS.keys()))
    p.add_argument("--data_root", type=str, default="./data/RESIDE-IN")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--save_dir", type=str, default=None, help="default: <checkpoint dir>/eval_<resolution>")
    p.add_argument("--eval_resolution", type=str, default="fixed", choices=["fixed", "native"],
                   help="fixed: resize test images to img_size (training protocol); native: original resolution")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--lgn_ch", type=int, default=16)
    p.add_argument("--v1_ch", type=int, default=32)
    p.add_argument("--v2_ch", type=int, default=64)
    p.add_argument("--v4_ch", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--measure_latency", action="store_true", help="measure inference latency (batch size 1)")
    p.add_argument("--latency_warmup", type=int, default=20)
    p.add_argument("--latency_iters", type=int, default=100)
    p.add_argument("--save_images", action="store_true", help="save restored images to <save_dir>/images")
    p.add_argument("--num_workers", type=int, default=2)
    return p


def main():
    # Training-only keys in the config (epochs, lr, ...) are ignored here.
    args = parse_args_with_config(get_parser(), strict=False)
    if args.dataset is None:
        sys.exit("error: --dataset is required (on the command line or in --config)")
    set_seed(args.seed)
    args.dataset = args.dataset.lower()
    display = DATASET_CONFIGS[args.dataset]["display_name"]
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), f"eval_{args.eval_resolution}")
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_pairs = build_dataset_pairs(args.dataset, args.data_root)
    if len(test_pairs) == 0:
        print("ERROR: no test pairs found.")
        print_expected_structure(args.dataset, args.data_root)
        sys.exit(1)
    names = [os.path.basename(a) for a, _ in test_pairs]
    resize = None if args.eval_resolution == "native" else (args.img_size, args.img_size)
    test_loader = DataLoader(PairedImageDataset(test_pairs, resize, augment=False),
                             batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = VentralNetSNN(args.lgn_ch, args.v1_ch, args.v2_ch, args.v4_ch, args.T).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{display} | {len(test_pairs)} test images | resolution={args.eval_resolution} | "
          f"params={total_params:,} | T={args.T} | device={device}")

    # Spike firing rate: fraction of active (spiking) outputs over all LIF layers, time steps and pixels.
    counts = {}

    def hook(name):
        def fn(module, inputs, output):
            s, c = counts.get(name, (0.0, 0))
            counts[name] = (s + output.detach().float().sum().item(), c + output.numel())
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
               if isinstance(m, LearnableThresholdLIFNode)]

    on_output = None
    if args.save_images:
        img_dir = os.path.join(args.save_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        def on_output(idx, pred):
            arr = pred[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(arr).save(os.path.join(img_dir, os.path.splitext(names[idx])[0] + ".png"))

    pad_to = 8 if args.eval_resolution == "native" else None
    l1, ps, ss, rows = evaluate(model, test_loader, device, pad_to=pad_to, per_image=True, on_output=on_output)
    for h in handles:
        h.remove()
    layer_rates = {n: s / c for n, (s, c) in counts.items()}
    firing = {"global": sum(s for s, _ in counts.values()) / sum(c for _, c in counts.values()),
              "per_layer": layer_rates}

    print("=" * 60)
    print(f"L1: {l1:.4f}  PSNR: {ps:.2f}  SSIM: {ss:.4f}  firing rate: {firing['global']:.4f}")
    model.print_neuron_stats()

    latency = None
    if args.measure_latency:
        x, _ = next(iter(test_loader))
        x = x.to(device)
        if pad_to:
            H, W = x.shape[-2:]
            x = torch.nn.functional.pad(x, (0, (pad_to - W % pad_to) % pad_to, 0, (pad_to - H % pad_to) % pad_to),
                                        mode="reflect")
        with torch.no_grad():
            for _ in range(args.latency_warmup):
                model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.latency_iters):
                model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / args.latency_iters
        latency = {"ms_per_image": ms, "input_hw": list(x.shape[-2:]),
                   "warmup": args.latency_warmup, "iters": args.latency_iters}
        print(f"Latency: {ms:.2f} ms/image (input {tuple(x.shape[-2:])})")
    print("=" * 60)

    write_per_image_csv(rows, names, os.path.join(args.save_dir, "per_image_metrics.csv"))
    save_json({"dataset": args.dataset, "eval_resolution": args.eval_resolution, "l1": l1, "psnr": ps, "ssim": ss,
               "num_test_images": len(test_pairs), "total_params": total_params, "T": args.T,
               "firing_rate": firing, "latency": latency, "checkpoint": os.path.abspath(args.checkpoint),
               "checkpoint_sha256": sha256_file(args.checkpoint), "environment": environment_info()},
              os.path.join(args.save_dir, "result.json"))
    print(f"Results written to {args.save_dir}")


if __name__ == "__main__":
    main()
