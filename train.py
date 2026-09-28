"""Train VentralNetSNN.

Example:
    python train.py --config configs/its.yaml --seed 42 --save_dir results/its/seed_42
"""
import argparse
import os
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader

from datasets import (DATASET_CONFIGS, PairedImageDataset, build_dataset_pairs,
                      print_expected_structure, resolve_dataset_paths)
from models import LearnableThresholdLIFNode, VentralNetSNN
from utils.common import (append_jsonl, environment_info, parse_args_with_config, save_json,
                          seed_worker, set_seed, sha256_file, write_per_image_csv)
from utils.engine import evaluate, train_one_epoch
from utils.losses import PerceptualLoss


def get_parser():
    p = argparse.ArgumentParser(description="Train VentralNetSNN")
    p.add_argument("--config", type=str, default=None, help="YAML file whose values become argument defaults")
    p.add_argument("--dataset", type=str, default=None, choices=sorted(DATASET_CONFIGS.keys()))
    p.add_argument("--data_root", type=str, default="./data/RESIDE-IN")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--T", type=int, default=4, help="number of time steps")
    p.add_argument("--lgn_ch", type=int, default=16)
    p.add_argument("--v1_ch", type=int, default=32)
    p.add_argument("--v2_ch", type=int, default=64)
    p.add_argument("--v4_ch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ssim_weight", type=float, default=0.15)
    p.add_argument("--perceptual_weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42, help="model initialisation seed")
    p.add_argument("--split_seed", type=int, default=2026, help="data shuffling and augmentation seed")
    p.add_argument("--num_workers", type=int, default=4,
                   help="training data-loader workers (4 reproduces the reported runs)")
    return p


def main():
    args = parse_args_with_config(get_parser())
    if args.dataset is None:
        sys.exit("error: --dataset is required (on the command line or in --config)")

    set_seed(args.seed)
    args.dataset = args.dataset.lower()
    dataset_config = DATASET_CONFIGS[args.dataset]
    if args.save_dir is None:
        args.save_dir = os.path.join(".", "checkpoints", args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_paths = resolve_dataset_paths(args.dataset, args.data_root)
    train_pairs, test_pairs = build_dataset_pairs(args.dataset, args.data_root)

    print("=" * 60)
    print(f"VentralNetSNN training - {dataset_config['display_name']}")
    print(f"Data root: {args.data_root}")
    print(f"Save dir:  {args.save_dir}")
    for key, value in dataset_paths.items():
        print(f"{key}: {value}")
    print(f"Train pairs: {len(train_pairs)}  Test pairs: {len(test_pairs)}  Device: {device}")
    print("=" * 60)
    sys.stdout.flush()

    if len(train_pairs) == 0:
        print("ERROR: no training pairs found.")
        print_expected_structure(args.dataset, args.data_root)
        sys.exit(1)

    # Data loaders. Shuffling uses a dedicated generator and every worker is re-seeded
    # with split_seed + worker_id, so the data stream is independent of --seed.
    g = torch.Generator()
    g.manual_seed(args.split_seed)
    train_loader = DataLoader(
        PairedImageDataset(train_pairs, (args.img_size, args.img_size), augment=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
        worker_init_fn=partial(seed_worker, base_seed=args.split_seed),
    )
    test_loader = DataLoader(
        PairedImageDataset(test_pairs, (args.img_size, args.img_size), augment=False),
        batch_size=1,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        pin_memory=True,
    ) if len(test_pairs) > 0 else None

    model = VentralNetSNN(args.lgn_ch, args.v1_ch, args.v2_ch, args.v4_ch, args.T).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    num_lif = sum(1 for m in model.modules() if isinstance(m, LearnableThresholdLIFNode))
    print(f"Params: {total_params:,}  Learnable-threshold LIF layers: {num_lif}  T={args.T}")
    sys.stdout.flush()

    save_json({"args": vars(args), "command": " ".join(sys.argv), "num_train_pairs": len(train_pairs),
               "num_test_pairs": len(test_pairs), "total_params": total_params, "num_lif_layers": num_lif,
               "environment": environment_info()}, os.path.join(args.save_dir, "config.json"))
    history_path = os.path.join(args.save_dir, "history.jsonl")
    if os.path.exists(history_path):
        os.remove(history_path)

    perceptual = PerceptualLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    best_psnr, best_epoch = 0, 0
    best_model_path = os.path.join(args.save_dir, "best.pth")
    last_model_path = os.path.join(args.save_dir, "last.pth")

    for epoch in range(1, args.epochs + 1):
        # The perceptual weight is halved during the first 30 epochs.
        perc_w = args.perceptual_weight * 0.5 if epoch <= 30 else args.perceptual_weight

        loss = train_one_epoch(model, train_loader, optimizer, device, perceptual,
                               ssim_weight=args.ssim_weight, perceptual_weight=perc_w, grad_clip=1.0)
        scheduler.step()

        if test_loader is not None:
            l1, ps, ss = evaluate(model, test_loader, device)
            is_best = ps > best_psnr
            if is_best:
                best_psnr, best_epoch = ps, epoch
                torch.save(model.state_dict(), best_model_path)
            print(f"[{epoch:03d}] Loss={loss:.4f} Test: L1={l1:.4f} PSNR={ps:.2f} SSIM={ss:.4f}"
                  + ("  * best" if is_best else ""))
            append_jsonl({"epoch": epoch, "train_loss": loss, "val_l1": l1, "val_psnr": ps, "val_ssim": ss,
                          "lr": scheduler.get_last_lr()[0]}, history_path)
            if epoch % 20 == 0:
                model.print_neuron_stats()
        else:
            torch.save(model.state_dict(), last_model_path)
            print(f"[{epoch:03d}] Loss={loss:.4f}")
            append_jsonl({"epoch": epoch, "train_loss": loss, "lr": scheduler.get_last_lr()[0]}, history_path)
        sys.stdout.flush()

    if test_loader is not None:
        # Reload the best checkpoint and evaluate once more.
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        l1, ps, ss, rows = evaluate(model, test_loader, device, per_image=True)
        names = [os.path.basename(a) for a, _ in test_pairs]
        write_per_image_csv(rows, names, os.path.join(args.save_dir, "per_image_metrics.csv"))
        save_json({"dataset": args.dataset, "l1": l1, "psnr": ps, "ssim": ss, "best_psnr": best_psnr,
                   "best_epoch": best_epoch, "num_test_images": len(test_pairs),
                   "checkpoint_sha256": sha256_file(best_model_path)},
                  os.path.join(args.save_dir, "result.json"))
        print("=" * 60)
        print(f"FINAL TEST RESULTS - {dataset_config['display_name']}")
        print(f"L1: {l1:.4f}  PSNR: {ps:.2f}  SSIM: {ss:.4f}  (best epoch {best_epoch})")
        model.print_neuron_stats()
        print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
