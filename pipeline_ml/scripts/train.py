#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
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
    set_seed,
    update_confusion,
)
from model import TinyUNet


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)

            pred = logits.argmax(dim=1).cpu().numpy()
            target = y.cpu().numpy()
            update_confusion(conf, pred, target, num_classes)

    metrics = metrics_from_confusion(conf)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a simple segmentation model on the OilSpill package")
    p.add_argument("--run-name", type=str, default="train_run")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = pick_device(args.device)
    num_classes = get_num_classes()

    dataset_root = resolve_dataset_root()
    results_dir = resolve_results_dir()

    run_dir = results_dir / args.run_name
    checkpoints_dir = run_dir / "checkpoints"
    ensure_dir(checkpoints_dir)

    train_samples = load_split_csv(dataset_root, "train")
    val_samples = load_split_csv(dataset_root, "val")

    train_ds = OilSpillSegmentationDataset(train_samples)
    val_ds = OilSpillSegmentationDataset(val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyUNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * x.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = train_loss_sum / max(len(train_ds), 1)
        val_metrics = evaluate(model, val_loader, device, num_classes)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_iou_per_class": val_metrics["iou_per_class"],
            "timestamp": int(time.time()),
        }
        history.append(row)

        last_ckpt = checkpoints_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            },
            last_ckpt,
        )

        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                },
                checkpoints_dir / "best.pt",
            )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['mean_iou']:.4f}"
        )

    save_json(
        run_dir / "history.json",
        {"history": history, "best_miou": best_miou, "device": str(device)},
    )
    print(f"Saved artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
