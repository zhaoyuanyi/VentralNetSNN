"""Dataset layouts, image-pair discovery and the paired image dataset."""
import glob
import os
import random
from typing import Dict, List, Tuple

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


# 数据集配置表：统一描述路径结构与配对模式
DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "its": {
        "display_name": "ITS",
        "mode": "paired",
        "train_input_parts": ("train", "hazy"),
        "train_target_parts": ("train", "GT"),
        "test_input_parts": ("test", "hazy"),
        "test_target_parts": ("test", "GT"),
        "recursive_listing": True,
        "expected_lines": [
            "data_root/train/hazy/  - input images",
            "data_root/train/GT/    - ground truth images",
            "data_root/test/hazy/   - test input images",
            "data_root/test/GT/     - test ground truth images",
        ],
    },
    "ots": {
        "display_name": "OTS",
        "mode": "paired",
        "train_input_parts": ("train", "hazy"),
        "train_target_parts": ("train", "GT"),
        "test_input_parts": ("test", "hazy"),
        "test_target_parts": ("test", "GT"),
        "recursive_listing": True,
        "expected_lines": [
            "data_root/train/hazy/  - input images",
            "data_root/train/GT/    - ground truth images",
            "data_root/test/hazy/   - test input images",
            "data_root/test/GT/     - test ground truth images",
        ],
    },
    "rain100h": {
        "display_name": "Rain100H",
        "mode": "rain100",
        "train_parts": ("RainTrainH",),
        "test_input_parts": ("Rain100H", "rainy"),
        "test_target_parts": ("Rain100H",),
        "recursive_listing": False,
        "expected_lines": [
            "data_root/RainTrainH/       - train images named rain-* / norain-*",
            "data_root/Rain100H/rainy/   - rainy test images",
            "data_root/Rain100H/         - clean test images named norain-*",
        ],
    },
    "rain100l": {
        "display_name": "Rain100L",
        "mode": "rain100",
        "train_parts": ("RainTrainL",),
        "test_input_parts": ("Rain100L", "rainy"),
        "test_target_parts": ("Rain100L",),
        "recursive_listing": False,
        "expected_lines": [
            "data_root/RainTrainL/       - train images named rain-* / norain-*",
            "data_root/Rain100L/rainy/   - rainy test images",
            "data_root/Rain100L/         - clean test images named norain-*",
        ],
    },
}


# 枚举目录中的图像文件，支持递归和常见后缀大小写
def list_images(folder: str, recursive: bool = True) -> List[str]:
    """递归搜索图片文件"""
    exts = ["png", "jpg", "jpeg", "bmp"]
    paths = []
    for e in exts:
        pattern = f"**/*.{e}" if recursive else f"*.{e}"
        upper_pattern = f"**/*.{e.upper()}" if recursive else f"*.{e.upper()}"
        paths += glob.glob(os.path.join(folder, pattern), recursive=recursive)
        paths += glob.glob(os.path.join(folder, upper_pattern), recursive=recursive)
    # set(): on case-insensitive file systems (Windows) '*.png' and '*.PNG' match the same files.
    return sorted(set(paths))


# 通用配对：通过文件名归一化后建立 hazy/rainy 与 GT/clean 的映射
def build_pairs(hazy_dir: str, clear_dir: str, recursive: bool = True) -> List[Tuple[str, str]]:
    """构建配对数据集（通用格式）"""
    hazy_paths = list_images(hazy_dir, recursive=recursive)
    clear_paths = list_images(clear_dir, recursive=recursive)

    # 构建清晰图像的映射表，处理常见的命名格式
    clear_map = {}
    for cp in clear_paths:
        basename = os.path.splitext(os.path.basename(cp))[0].lower()
        # 移除常见后缀
        key = basename.replace("_gt", "").replace("gt", "").replace("_clean", "").replace("clean", "")
        key = key.replace("norain-", "").replace("norain_", "")
        clear_map[key] = cp

    pairs = []
    for hp in hazy_paths:
        basename = os.path.splitext(os.path.basename(hp))[0].lower()
        # 移除常见前缀/后缀
        key = basename.replace("_hazy", "").replace("hazy", "").replace("_rain", "").replace("rain-", "")
        key = key.replace("rain_", "")

        if key in clear_map:
            pairs.append((hp, clear_map[key]))

    return sorted(pairs, key=lambda x: os.path.basename(x[0]).lower())


# Rain100 训练集配对：rain-xxx 与 norain-xxx 按同编号配对
def build_train_pairs(train_dir: str) -> List[Tuple[str, str]]:
    all_images = list_images(train_dir, recursive=False)
    rain_images = {}
    norain_images = {}

    for img_path in all_images:
        basename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(basename)[0].lower()

        if name_without_ext.startswith("rain-"):
            key = name_without_ext.replace("rain-", "")
            rain_images[key] = img_path
        elif name_without_ext.startswith("norain-"):
            key = name_without_ext.replace("norain-", "")
            norain_images[key] = img_path

    pairs = []
    for key in rain_images:
        if key in norain_images:
            pairs.append((rain_images[key], norain_images[key]))

    return sorted(pairs, key=lambda x: os.path.basename(x[0]).lower())


# Rain100 测试集配对：rainy/rain-xxx 与根目录 norain-xxx 配对
def build_test_pairs(rainy_dir: str, clean_dir: str) -> List[Tuple[str, str]]:
    rainy_images = list_images(rainy_dir, recursive=False)
    clean_images = list_images(clean_dir, recursive=False)

    clean_map = {}
    for clean_path in clean_images:
        basename = os.path.basename(clean_path)
        name_without_ext = os.path.splitext(basename)[0].lower()
        key = name_without_ext.replace("norain-", "")
        clean_map[key] = clean_path

    pairs = []
    for rainy_path in rainy_images:
        basename = os.path.basename(rainy_path)
        name_without_ext = os.path.splitext(basename)[0].lower()
        key = name_without_ext.replace("rain-", "")

        if key in clean_map:
            pairs.append((rainy_path, clean_map[key]))

    return sorted(pairs, key=lambda x: os.path.basename(x[0]).lower())


# 根据 DATASET_CONFIGS 拼接出当前数据集的关键目录路径
def resolve_dataset_paths(dataset: str, data_root: str) -> Dict[str, str]:
    config = DATASET_CONFIGS[dataset]
    mode = config["mode"]

    if mode == "paired":
        return {
            "train_input": os.path.join(data_root, *config["train_input_parts"]),
            "train_target": os.path.join(data_root, *config["train_target_parts"]),
            "test_input": os.path.join(data_root, *config["test_input_parts"]),
            "test_target": os.path.join(data_root, *config["test_target_parts"]),
        }

    return {
        "train_dir": os.path.join(data_root, *config["train_parts"]),
        "test_input": os.path.join(data_root, *config["test_input_parts"]),
        "test_target": os.path.join(data_root, *config["test_target_parts"]),
    }


# 统一构建训练/测试配对，屏蔽不同数据集组织差异
def build_dataset_pairs(dataset: str, data_root: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    config = DATASET_CONFIGS[dataset]
    paths = resolve_dataset_paths(dataset, data_root)

    if config["mode"] == "paired":
        recursive = bool(config["recursive_listing"])
        train_pairs = build_pairs(paths["train_input"], paths["train_target"], recursive=recursive)
        test_pairs = build_pairs(paths["test_input"], paths["test_target"], recursive=recursive)
    else:
        train_pairs = build_train_pairs(paths["train_dir"])
        test_pairs = build_test_pairs(paths["test_input"], paths["test_target"])

    return train_pairs, test_pairs


def print_expected_structure(dataset: str, data_root: str):
    config = DATASET_CONFIGS[dataset]
    print(f"Please check your data directory: {data_root}")
    print(f"Expected structure for {config['display_name']}:")
    for line in config["expected_lines"]:
        print(f"  {line}")


class PairedImageDataset(Dataset):
    """Paired degraded / clean images.

    ``resize=(H, W)`` resizes both images with bilinear interpolation; ``resize=None``
    keeps the native resolution. With ``augment=True`` a random horizontal flip and a
    random vertical flip (each with probability 0.5) are applied jointly to the pair.
    """

    def __init__(self, pairs, resize=None, augment=False):
        self.pairs = pairs
        self.resize = resize
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        degraded_path, clean_path = self.pairs[idx]
        degraded = Image.open(degraded_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        if self.resize:
            degraded = degraded.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
            clean = clean.resize((self.resize[1], self.resize[0]), Image.BILINEAR)

        if self.augment:
            if random.random() < 0.5:
                degraded, clean = TF.hflip(degraded), TF.hflip(clean)
            if random.random() < 0.5:
                degraded, clean = TF.vflip(degraded), TF.vflip(clean)

        return TF.to_tensor(degraded), TF.to_tensor(clean)
