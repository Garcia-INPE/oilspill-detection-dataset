from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


NUM_CLASSES = 2
TARGET_CLASS_IDS = [1]
CLASS_MAP_VERSION = "v1"
PRIMARY_METRIC = "iou_slick"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_csv(dataset_root: Path, split_name: str) -> List[Sample]:
    split_csv = dataset_root / "splits" / f"{split_name}.csv"
    samples: List[Sample] = []
    with split_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                Sample(
                    sample_id=row["sample_id"],
                    image_path=dataset_root / row["image_path"],
                    mask_path=dataset_root / row["mask_path"],
                )
            )
    return samples


class OilSpillSegmentationDataset(Dataset):
    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        image = Image.open(s.image_path).convert("RGB")
        mask = Image.open(s.mask_path).convert("L")

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.int64)
        mask_np = remap_mask(mask_np)

        image_t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask_np).contiguous()

        return {
            "sample_id": s.sample_id,
            "image": image_t,
            "mask": mask_t,
        }


def get_num_classes() -> int:
    return NUM_CLASSES


def remap_mask(mask_np: np.ndarray) -> np.ndarray:
    # 0 stays SEA, 1/2 are merged into SLICK.
    return (mask_np > 0).astype(np.int64)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def update_confusion(conf: np.ndarray, pred: np.ndarray, target: np.ndarray, num_classes: int) -> None:
    valid = (target >= 0) & (target < num_classes)
    merged = num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    counts = np.bincount(merged, minlength=num_classes * num_classes)
    conf += counts.reshape(num_classes, num_classes)


def metrics_from_confusion(conf: np.ndarray) -> dict:
    eps = 1e-9
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp

    iou = tp / (tp + fp + fn + eps)
    acc = tp.sum() / (conf.sum() + eps)

    target_class_ids = TARGET_CLASS_IDS
    target_iou = iou[target_class_ids]

    return {
        "pixel_accuracy": float(acc),
        # Primary metrics (foreground-only).
        "mean_iou": float(np.mean(target_iou)),
        "iou_per_class": [float(iou[c]) for c in target_class_ids],
        "target_class_ids": target_class_ids,
        # Auxiliary metrics (including background) for debugging.
        "mean_iou_all": float(np.mean(iou)),
        "iou_per_class_all": [float(v) for v in iou.tolist()],
    }


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config_path() -> Path:
    return Path(__file__).resolve().parent / "pipeline_ml_config.json"


def get_config_dict() -> dict:
    """Load required configuration from scripts/pipeline_ml_config.json."""
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing required config file: {config_path}. "
            "Create it before running the pipeline."
        )
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format in {config_path}: expected JSON object")
    return cfg


def _require_path_in_config(config: dict, key: str, base_dir: Path) -> Path:
    raw_value = config.get(key)
    if not raw_value or not isinstance(raw_value, str):
        raise ValueError(f"Config key '{key}' is required and must be a non-empty string")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path)
    return path.resolve()


def resolve_dataset_root() -> Path:
    """Resolve dataset root exclusively from scripts/pipeline_ml_config.json."""
    config = get_config_dict()
    dataset_root = _require_path_in_config(config, "dataset_root", get_config_path().parent)
    if not (dataset_root / "splits").is_dir():
        raise FileNotFoundError(
            f"Configured dataset_root has no splits/ directory: {dataset_root}"
        )
    return dataset_root


def resolve_results_dir() -> Path:
    """Resolve results directory exclusively from scripts/pipeline_ml_config.json."""
    config = get_config_dict()
    return _require_path_in_config(config, "results_dir", get_config_path().parent)


def resolve_benchmark_dir() -> Path:
    """Resolve benchmark directory exclusively from scripts/pipeline_ml_config.json."""
    config = get_config_dict()
    return _require_path_in_config(config, "benchmark_dir", get_config_path().parent)
