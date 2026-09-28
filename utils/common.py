"""Seeding, config loading and result-file helpers."""
import argparse
import hashlib
import json
import os
import platform
import random
import sys

import numpy as np
import torch


def set_seed(seed: int = 42):
    # 固定随机种子，提升实验可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id, base_seed):
    """DataLoader ``worker_init_fn``: seeds every worker with ``base_seed + worker_id``."""
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def parse_args_with_config(parser: argparse.ArgumentParser, strict: bool = True) -> argparse.Namespace:
    """Parse CLI arguments; values in ``--config`` (YAML) become defaults that the CLI can override.

    With ``strict=True`` unknown keys in the config are an error (catches typos); with
    ``strict=False`` they are ignored, so train and test scripts can share one config file.
    """
    pre, _ = parser.parse_known_args()
    if getattr(pre, "config", None):
        import yaml
        with open(pre.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - known)
        if unknown and strict:
            parser.error(f"unknown keys in {pre.config}: {unknown}")
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    return parser.parse_args()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for mod in ["torchvision", "spikingjelly", "numpy", "PIL"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[mod] = None
    return info


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(record, path):
    with open(path, "a", encoding="utf-8") as f:
        print(json.dumps(record, ensure_ascii=False), file=f)


def write_per_image_csv(rows, names, path):
    with open(path, "w", encoding="utf-8") as f:
        print("image,l1,psnr,ssim", file=f)
        for idx, l1, ps, ss in rows:
            print(f"{names[idx]},{l1:.6f},{ps:.4f},{ss:.6f}", file=f)
