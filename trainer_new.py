import argparse
import functools
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import wandb
import yaml

from abc import ABC, abstractmethod
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from typing import Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# use torch to get how much GPU ram sys has
def is_gpu_memory_gte_40_gb() -> bool | None:
  if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024 ** 3)  # Convert bytes to GB
    print(f"Total GPU Memory: {total_memory_gb:0.2f} GB")
    return total_memory_gb > 40

# === Logger Classes === #
class Logger(ABC):
    @abstractmethod
    def log(self, data: dict, step: int):
        pass


class WandbLogger(Logger):
    def __init__(self, project_name, run_name):
        self.run = wandb.init(project=project_name, name=run_name)

    def log(self, data: dict, step: int):
        self.run.log(data, step=step)

# ===
# LR Scheduler
# ===
class CosineLRScheduler:
  def __init__(self, warmup_steps, max_steps, max_lr, min_lr):
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps
    self.max_lr = max_lr
    self.min_lr = min_lr

  def get_lr(self, step):
    # linear warmup
    if step < self.warmup_steps:
      return self.max_lr * (step+1) / self.warmup_steps

    # constant lr
    if step > self.max_steps:
      return self.min_lr

    # cosine annealing
    decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return self.min_lr + coeff * (self.max_lr - self.min_lr)

# === Utility to load YAML configuration === #
def load_config():
    if len(sys.argv) < 2:
        print("Please provide a config.yaml path as arg")
        exit(0)
    config_file = sys.argv[1]
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if is_gpu_memory_gte_40_gb():
        config['batch_size'] *= 2
        config['grad_accumulation_steps'] /= 2
        config['num_steps'] /= 2
    return config


@functools.lru_cache(maxsize=2)
def load_shard(filename: str) -> np.ndarray:
    return np.load(filename)


class ShardedDataset(Dataset):
    def __init__(self, shard_paths: list[str], shard_size: int):
        super().__init__()
        self.shard_paths = shard_paths
        self.shard_size = shard_size

    def __len__(self):
        return len(self.shard_paths) * self.shard_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = index // self.shard_size
        shard = load_shard(self.shard_paths[shard_idx])
        example = shard[index % self.shard_size]
        image, label = example[:3], example[3:]
        return torch.tensor(image), torch.tensor(label).squeeze().long()


def build_model(model_name: str, pretrained_flag: bool, num_classes: int, dropout: float = 0.0) -> nn.Module:
    if pretrained_flag:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    else:
        config = Mask2FormerConfig.from_pretrained(model_name)
        config.num_labels = num_classes

        # cannot apply dropout because of some bug in HF lib
        #config.dropout = dropout
        #config.dropout = 0.1
        model = Mask2FormerForUniversalSegmentation(config)
    return model


def criterion(
    predictions: torch.Tensor, targets: torch.Tensor, grad_steps: int
) -> Dict[str, Union[torch.Tensor, float]]:
    total_loss: torch.Tensor = F.cross_entropy(predictions, targets) / grad_steps
    return {"total_loss": total_loss}


def load_dataset(dataset_path: str, shard_size: int) -> Tuple[Dataset, Dataset]:
    all_shards = glob.glob(f"{dataset_path}/*.npy")
    all_shards = sorted(
        map(lambda x: (x, int(os.path.basename(x).split(".")[0].split("_")[1])), all_shards), key=lambda x: x[1]
    )[
        :-1
    ]  # drop last shard
    random.shuffle(all_shards)
    all_shards = [x[0] for x in all_shards]
    val_shards = all_shards[:1]
    train_shards = all_shards[1:]
    train_dataset = ShardedDataset(shard_paths=train_shards, shard_size=shard_size)
    val_dataset = ShardedDataset(shard_paths=val_shards, shard_size=shard_size)
    return train_dataset, val_dataset


# ===
# Evaluation
# ===


def compute_iou(pred_mask, true_mask):
    """
    Computes the Intersection over Union (IoU) between predicted and true masks.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0  # Perfect match if both masks are empty
    return intersection / union


def compute_dice(pred_mask, true_mask):
    """
    Computes the Dice Coefficient between predicted and true masks.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    dice = (2.0 * intersection) / (pred_mask.sum() + true_mask.sum())

    if (pred_mask.sum() + true_mask.sum()) == 0:
        return 1.0  # Perfect match if both masks are empty
    return dice


def evaluate(
    model: nn.Module, val_loader: DataLoader, device: str, step: int, logger: Logger, class_mapping: Dict[str, int]
) -> None:
    model.eval()

    class_iou: np.ndarray = np.zeros(len(class_mapping))
    class_dice: np.ndarray = np.zeros(len(class_mapping))
    class_counts: np.ndarray = np.zeros(len(class_mapping))

    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader), desc="Evaluating"):
            images = images.to(device)
            labels = labels.cpu().numpy()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=images)
            pred_logits = outputs.masks_queries_logits
            pred_masks = F.interpolate(pred_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pred_masks = pred_masks.argmax(dim=1).cpu().numpy()

            for class_label, class_id in class_mapping.items():
                if class_label == "background":
                    continue

                for pred_mask, true_mask in zip(pred_masks, labels):
                    pred_class_mask = pred_mask == class_id
                    true_class_mask = true_mask == class_id

                    if true_class_mask.sum() == 0 and pred_class_mask.sum() == 0:
                        continue

                    iou_score: float = compute_iou(pred_class_mask, true_class_mask)
                    dice_score: float = compute_dice(pred_class_mask, true_class_mask)

                    class_iou[class_id] += iou_score
                    class_dice[class_id] += dice_score
                    class_counts[class_id] += 1

        mean_iou = {
            f"{label}": class_iou[idx] / class_counts[idx] if class_counts[idx] > 0 else np.nan
            for label, idx in class_mapping.items()
            if label != "background"
        }
        mean_dice = {
            f"{label}": class_dice[idx] / class_counts[idx] if class_counts[idx] > 0 else np.nan
            for label, idx in class_mapping.items()
            if label != "background"
        }
        mean_iou["mean"] = np.nanmean(list(mean_iou.values()))
        mean_dice["mean"] = np.nanmean(list(mean_dice.values()))

        eval_metrics = {
            "IoU": mean_iou,
            "Dice": mean_dice,
        }
        logger.log(eval_metrics, step)

        # Visualize some predictions
        test_images, test_labels = next(iter(val_loader))
        test_images = test_images.to(device)
        with torch.no_grad():
            outputs = model(pixel_values=test_images)
        test_pred_logits = outputs.masks_queries_logits
        test_pred_masks = F.interpolate(
            test_pred_logits, size=test_labels.shape[-2:], mode="bilinear", align_corners=False
        )
        test_pred_masks = test_pred_masks.argmax(dim=1).cpu().numpy()

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(4):
            axs[0, i].imshow(test_images[i].permute(1, 2, 0).cpu().numpy())
            axs[0, i].axis("off")
            axs[0, i].set_title("Original Image")

            axs[1, i].imshow(test_pred_masks[i], cmap="gray")
            axs[1, i].axis("off")
            axs[1, i].set_title("Predicted Mask")

        logger.log(dict(test_images=fig), step=step)
        plt.close()


def save_checkpoint(model: nn.Module, ckpt_dir: str, step: int) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    model.eval()
    model.save_pretrained(ckpt_dir)
    print(f"Checkpoint saved at step {step}")


# === Trainer Scaffold === #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    config = load_config()
    logger = WandbLogger(project_name=config.get("project_name", "default_project"), run_name=config["run_name"])
    model = build_model(config["model_name"], config["pretrained_flag"], num_classes=len(config["class_mapping"]), dropout=config['dropout'])
    model = model.to(device)
    print("Model has been loaded on device")
    train_dataset, val_dataset = load_dataset(config["dataset_path"], config["shard_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=config["prefetch_factor"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,  # Hardcoded based on your instructions
        prefetch_factor=config["prefetch_factor"],
    )
    print("DataLoaders created")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["min_lr"])
    lr_scheduler = CosineLRScheduler(config['warmup_steps'], config['max_steps'], config['max_lr'], config['min_lr'])
    # Main training loop
    step = 0
    while step < config["num_steps"]:
        for batch in train_loader:
            step += 1
            if step % config["eval_interval"] == 0:
                print(f"Step {step}: Performing evaluation")
                evaluate(model, val_loader, device, step, logger, config["class_mapping"])
            if step % config["ckpt_interval"] == 0 and config["do_ckpt"]:
                print(f"Step {step}: Saving checkpoint")
                save_checkpoint(model, os.path.join(config["ckpt_dir"], config["run_name"]), step)
            # train
            model.train()
            optimizer.zero_grad()
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=images)
                logits = outputs.masks_queries_logits
                logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                losses = criterion(logits, labels, config.get("grad_accumulation_steps", 1))
                loss = losses["total_loss"]
            loss.backward()

            lr = lr_scheduler.get_lr(step)
            for param_group in optimizer.param_groups: param_group['lr'] = lr
            optimizer.step()
            log_data = dict(train_loss=loss.item(), lr=lr)
            logger.log(log_data, step)

            if step >= config["num_steps"]:
                break

    logger.log({"status": "Training finished"}, step=step)


if __name__ == "__main__":
    main()
