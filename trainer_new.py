import argparse
import functools
import glob
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
torch.set_float32_matmul_precision('high')

# === Logger Classes === #
class Logger(ABC):
    @abstractmethod
    def log(self, data: dict, step: int):
        pass

class WandbLogger(Logger):
    def __init__(self, project_name, run_name): self.run = wandb.init(project=project_name, name=run_name)
    def log(self, data: dict, step: int): self.run.log(data, step=step)

# === Utility to load YAML configuration === #
def load_config():
    if len(sys.argv) < 2:
        print('Please provide a config.yaml path as arg')
        exit(0)
    config_file = sys.argv[1]
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


@functools.lru_cache(maxsize=2)
def load_shard(filename: str) -> np.ndarray: return np.load(filename)

class ShardedDataset(Dataset):
    def __init__(self, shard_paths: list[str], shard_size: int):
        super().__init__()
        self.shard_paths = shard_paths
        self.shard_size = shard_size
    def __len__(self): return len(self.shard_paths)*self.shard_size
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = index // self.shard_size
        shard = load_shard(self.shard_paths[shard_idx])
        example = shard[index % self.shard_size] 
        image, label = example[:3], example[3:]
        return torch.tensor(image), torch.tensor(label).squeeze().long()


def build_model(model_name: str, pretrained_flag: bool, num_classes: int) -> nn.Module:
    """Load or initialize a model (placeholder)."""
    if pretrained_flag:
      model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
      config = Mask2FormerConfig.from_pretrained(model_name)
      config.num_labels = num_classes
      model = Mask2FormerForUniversalSegmentation(config)
    return model


def criterion(predictions: torch.Tensor, targets: torch.Tensor, grad_steps: int) -> Dict[str, Union[torch.Tensor, float]]:
    total_loss: torch.Tensor = F.cross_entropy(predictions, targets) / grad_steps
    return {'total_loss': total_loss}

def load_dataset(dataset_path: str, shard_size: int) -> Tuple[Dataset, Dataset]:
    all_shards = glob.glob(f'{dataset_path}/*.npy')
    all_shards = sorted(map(lambda x: (x, int(os.path.basename(x).split('.')[0].split('_')[1])), all_shards), key=lambda x: x[1])[:-1] # drop last shard
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
    dice = (2. * intersection) / (pred_mask.sum() + true_mask.sum())
    
    if (pred_mask.sum() + true_mask.sum()) == 0:
        return 1.0  # Perfect match if both masks are empty
    return dice

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    step: int,
    logger: Logger,
    class_mapping: Dict[str, int]
) -> None:
    model.eval()
    
    class_iou: np.ndarray = np.zeros(len(class_mapping))
    class_dice: np.ndarray = np.zeros(len(class_mapping))
    class_counts: np.ndarray = np.zeros(len(class_mapping))

    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader), desc='Evaluating'):
            images = images.to(device)
            labels = labels.cpu().numpy()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(pixel_values=images)
            pred_logits = outputs.masks_queries_logits
            pred_masks = F.interpolate(pred_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            pred_masks = pred_masks.argmax(dim=1).cpu().numpy()

            for class_label, class_id in class_mapping.items():
                if class_label == 'background':
                    continue

                for pred_mask, true_mask in zip(pred_masks, labels):
                    pred_class_mask = (pred_mask == class_id)
                    true_class_mask = (true_mask == class_id)

                    if true_class_mask.sum() == 0 and pred_class_mask.sum() == 0:
                        continue

                    iou_score: float = compute_iou(pred_class_mask, true_class_mask)
                    dice_score: float = compute_dice(pred_class_mask, true_class_mask)

                    class_iou[class_id] += iou_score
                    class_dice[class_id] += dice_score
                    class_counts[class_id] += 1

    mean_iou = {f'{label}_iou': class_iou[idx] / class_counts[idx] if class_counts[idx] > 0 else np.nan for label, idx in class_mapping.items() if label != 'background'}
    mean_dice = {f'{label}_dice': class_dice[idx] / class_counts[idx] if class_counts[idx] > 0 else np.nan for label, idx in class_mapping.items() if label != 'background'}

    print({**mean_iou, **mean_dice})

    eval_metrics = {
        **mean_iou,
        **mean_dice,
        "Mean IoU": np.nanmean(list(mean_iou.values())),
        "Mean Dice Coefficient": np.nanmean(list(mean_dice.values()))
    }
    logger.log(eval_metrics, step)


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
    logger = WandbLogger(project_name=config.get('project_name', 'default_project'), run_name=config['run_name'])
    model = build_model(config['model_name'], config['pretrained_flag'], num_classes=len(config['class_mapping']))
    model = model.to(device)
    print('Model has been loaded on device')
    train_dataset, val_dataset = load_dataset(config['dataset_path'], config['shard_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=config['prefetch_factor']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,  # Hardcoded based on your instructions
        prefetch_factor=config['prefetch_factor']
    )
    print('DataLoaders created')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    # Main training loop
    step = 0
    while step < config['num_steps']:
        for batch in train_loader:
            step += 1
            if step % config['eval_interval'] == 0:
                print(f"Step {step}: Performing evaluation")
                evaluate(model, val_loader, device, step, logger, config['class_mapping'])
            if step % config['ckpt_interval'] == 0 and config['do_ckpt']:
                print(f"Step {step}: Saving checkpoint")
                save_checkpoint(model, os.path.join(config['ckpt_dir'], config['run_name']), step)
            # train
            model.train()
            optimizer.zero_grad()
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
              outputs = model(pixel_values=images)
              logits = outputs.masks_queries_logits
              logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
              logits = logits.squeeze(1)  # Remove extra dimension (B, 1, H, W) -> (B, H, W)
              losses = criterion(logits, labels, config.get('grad_accumulation_steps', 1))
              loss = losses['total_loss']
            loss.backward()
            optimizer.step()
            log_data = dict(train_loss=loss.item())
            logger.log(log_data, step)

            if step >= config['num_steps']:
                break

    logger.log({"status": "Training finished"}, step=step)

if __name__ == "__main__":
    main()
