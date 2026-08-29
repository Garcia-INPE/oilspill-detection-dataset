#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    OilSpillSegmentationDataset,
    ensure_dir,
    get_num_classes,
    load_split_csv,
    metrics_from_confusion,
    pick_device,
    resolve_dataset_root,
    resolve_results_dir,
    save_json,
    update_confusion,
)
from model import TinyUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a saved model checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--run-name", type=str, default="validation_run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    num_classes = get_num_classes()

    dataset_root = resolve_dataset_root()
    results_dir = resolve_results_dir()

    samples = load_split_csv(dataset_root, args.split)
    ds = OilSpillSegmentationDataset(samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyUNet(num_classes=num_classes).to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility with torch versions that do not accept weights_only.
        ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"validate {args.split}"):
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)

            pred = logits.argmax(dim=1).cpu().numpy()
            target = y.cpu().numpy()
            update_confusion(conf, pred, target, num_classes)

    metrics = metrics_from_confusion(conf)
    metrics["loss"] = total_loss / max(len(ds), 1)
    metrics["split"] = args.split
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["device"] = str(device)

    out_dir = results_dir / args.run_name
    ensure_dir(out_dir)
    out_path = out_dir / f"metrics_{args.split}.json"
    save_json(out_path, metrics)

    print(f"Validation metrics saved to: {out_path}")
    print(metrics)


if __name__ == "__main__":
    main()
