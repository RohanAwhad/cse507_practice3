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
def load_config(config_file):
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
        return image, label


def build_model(model_name: str, pretrained_flag: bool, num_classes: int) -> nn.Module:
    """Load or initialize a model (placeholder)."""
    if pretrained_flag:
      model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
      config = Mask2FormerConfig.from_pretrained(model_name)
      config.num_labels = num_classes
      model = Mask2FormerForUniversalSegmentation(config)
    return model


def criterion(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Union[torch.Tensor, float]]:
    total_loss: torch.Tensor = F.cross_entropy(predictions, targets)
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



def prepare_coco_format(pred_masks, true_masks, image_sizes):
    """
    Converts prediction and true masks into COCO-like format for mAP evaluation using pycocotools.
    """
    coco_format_preds = []
    coco_format_gts = []
    coco_format_imgs = []  # This will hold image metadata
    ann_id = 1  # Initialize annotation ID

    for i in range(len(pred_masks)):
        # Prepare a single image entry (height and width should come from the original image)
        image_info = {
            'id': i,
            'width': image_sizes[i][0],  # Assuming image_sizes is a list of (width, height) tuples
            'height': image_sizes[i][1]
        }
        coco_format_imgs.append(image_info)

        # Calculate area for ground truth (number of non-zero pixels)
        gt_area = np.sum(true_masks[i] > 0)

        # Prepare a single ground truth entry
        gt = {
            'image_id': i,
            'category_id': 1,  # Assuming single class for simplicity
            'segmentation': true_masks[i].tolist(),  # Use mask for segmentation (you can further encode it if needed)
            'iscrowd': 0,
            'id': ann_id,  # Unique ID for each annotation
            'area': gt_area  # Area of the ground truth mask
        }
        coco_format_gts.append(gt)

        # Calculate area for predicted mask (number of non-zero pixels)
        pred_area = np.sum(pred_masks[i] > 0)

        # Prepare a single prediction entry
        pred = {
            'image_id': i,
            'category_id': 1,  # Assuming single class for simplicity
            'segmentation': pred_masks[i].tolist(),  # Use mask for segmentation (you can further encode it if needed)
            'score': 1.0,  # Assume high confidence for this example
            'id': ann_id,  # Unique ID for each annotation
            'area': pred_area  # Area of the predicted mask
        }
        coco_format_preds.append(pred)

        ann_id += 1  # Increment annotation ID for the next entry

    # Define the 'categories' field with one class (e.g., "object")
    categories = [{
        'id': 1,
        'name': 'object',
        'supercategory': 'object'
    }]

    return coco_format_gts, coco_format_preds, coco_format_imgs, categories

def calculate_coco_map(true_masks, pred_masks, image_sizes):
    """
    Calculate mAP using pycocotools on true and predicted masks in COCO-like format.
    """
    # Convert to COCO format
    coco_gt = COCO()  # COCO ground truth object
    coco_dt = COCO()  # COCO detections object

    # Prepare annotations, image info, and categories
    gts, preds, imgs, categories = prepare_coco_format(pred_masks, true_masks, image_sizes)

    # Add the 'categories' and 'images' field to both ground truth and prediction datasets
    coco_gt.dataset = {'annotations': gts, 'images': imgs, 'categories': categories}
    coco_dt.dataset = {'annotations': preds, 'images': imgs, 'categories': categories}

    coco_gt.createIndex()
    coco_dt.createIndex()

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')  # Specify 'segm' for segmentation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return the summary of mAP (AP@[IoU=0.50:0.95])
    return coco_eval.stats[0]  # stats[0] is the mAP@[IoU=0.50:0.95]

def evaluate(model: nn.Module, val_loader: DataLoader, device: str, step: int, logger: Logger) -> None:
    model.eval()

    all_true_masks = []
    all_pred_masks = []
    image_sizes = []  # Store original image sizes

    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader), desc='Evaluating'):
            images = images.to(device)
            labels = labels.cpu().numpy()  # Keep labels on CPU for metrics

            # Get original image sizes (before preprocessing)
            for image in images:
                original_size = (image.shape[-1], image.shape[-2])  # (width, height)
                image_sizes.append(original_size)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
              outputs = model(pixel_values=images)
            pred_logits = outputs.masks_queries_logits

            # Resize predicted logits to the original size and convert to masks
            pred_masks = F.interpolate(pred_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            pred_masks = pred_masks.argmax(dim=1).cpu().numpy()  # (B, H, W)

            # Append true labels and predictions for each batch
            all_true_masks.append(labels)
            all_pred_masks.append(pred_masks)

    # Convert lists to numpy arrays for evaluation
    all_true_masks = np.concatenate(all_true_masks, axis=0)  # (N, H, W)
    all_pred_masks = np.concatenate(all_pred_masks, axis=0)  # (N, H, W)

    # Calculate mAP using pycocotools (on the entire dataset)
    map_result = calculate_coco_map(all_true_masks, all_pred_masks, image_sizes)
    print(f"Mean Average Precision (AP@[IoU=0.50:0.95]): {map_result:.4f}")
    eval_metrics = {"Mean Average Precision (AP@[IoU=0.50:0.95])" : map_result}
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
                evaluate(model, val_loader, device, step, logger)
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
              losses = criterion(logits, labels)
              loss = losses['total_loss']
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            log_data = dict(train_loss=loss.item())
            logger.log(log_data, step)

            if step >= config['num_steps']:
                break

    logger.log({"status": "Training finished"}, step=step)

if __name__ == "__main__":
    main()
