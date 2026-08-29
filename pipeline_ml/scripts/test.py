#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
    p = argparse.ArgumentParser(description="Test checkpoint and persist predictions")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--run-name", type=str, default="test_run")
    p.add_argument("--save-max-preds", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    num_classes = get_num_classes()

    dataset_root = resolve_dataset_root()
    results_dir = resolve_results_dir()

    samples = load_split_csv(dataset_root, "test")
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

    out_dir = results_dir / args.run_name
    preds_dir = out_dir / "predictions"
    ensure_dir(preds_dir)

    saved = 0
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            sample_ids = batch["sample_id"]

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)

            pred_t = logits.argmax(dim=1)
            pred = pred_t.cpu().numpy()
            target = y.cpu().numpy()
            update_confusion(conf, pred, target, num_classes)

            for i, sid in enumerate(sample_ids):
                if saved < args.save_max_preds:
                    pred_img = Image.fromarray(pred[i].astype(np.uint8), mode="L")
                    pred_img.save(preds_dir / f"{sid}_pred_mask.png")
                    saved += 1

                rows.append({
                    "sample_id": sid,
                    "pred_saved": int(saved <= args.save_max_preds),
                })

    metrics = metrics_from_confusion(conf)
    metrics["loss"] = total_loss / max(len(ds), 1)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["device"] = str(device)
    metrics["saved_predictions"] = saved

    ensure_dir(out_dir)
    save_json(out_dir / "metrics_test.json", metrics)

    with (out_dir / "test_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "pred_saved"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Test metrics saved to: {out_dir / 'metrics_test.json'}")
    print(f"Predicted masks saved to: {preds_dir}")


if __name__ == "__main__":
    main()
