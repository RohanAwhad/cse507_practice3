import argparse
import dataclasses
import json
import os
import pandas as pd
import warnings
from tqdm import tqdm
from typing import Callable, Dict, List

import logger

parser = argparse.ArgumentParser(description='Training configuration')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--use_pretrained', action='store_true', help='if set, will be finetuning')
args = parser.parse_args()
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
use_pretrained = args.use_pretrained

# ===
# Data Loaders
# ===
import functools
import glob
import numpy as np
import random
import torch

TRAIN_IMAGE_WIDTH = 512
TRAIN_IMAGE_HEIGHT = 512
CLASS_MAPPING = {'background': 0, 'right_lung': 1, 'left_lung': 2, 'heart': 3}
SHARD_SIZE = 1032
ROOT_DIR = '/scratch/rawhad/CSE507/practice_3/tiny_ds'
all_shards = glob.glob(f'{ROOT_DIR}/*.npy')
all_shards = sorted(map(lambda x: (x, int(os.path.basename(x).split('.')[0].split('_')[1])), all_shards), key=lambda x: x[1])[:-1] # drop last shard
random.shuffle(all_shards)
all_shards = [x[0] for x in all_shards]
val_shards = all_shards[:1]
train_shards = all_shards[1:]

class ShardedDataset(torch.utils.data.Dataset):
  def __init__(self, shard_paths: list[str]):
    super().__init__()
    self.shard_paths = shard_paths

  def __len__(self):
    return len(self.shard_paths)*SHARD_SIZE

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    shard_idx = index // SHARD_SIZE
    shard = load_shard(self.shard_paths[shard_idx])
    example = shard[index % SHARD_SIZE] 
    image, label = torch.from_numpy(example[:3]).float(), torch.from_numpy(example[3:]).long().squeeze(0)
    return image, label


@functools.lru_cache(maxsize=2)
def load_shard(filename: str) -> np.ndarray:
  return np.load(filename)


# Create sharded datasets
train_sharded_dataset = ShardedDataset(shard_paths=train_shards)
val_sharded_dataset = ShardedDataset(shard_paths=val_shards)

# Create DataLoaders for sharded datasets
train_loader = torch.utils.data.DataLoader(train_sharded_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=16)
val_loader = torch.utils.data.DataLoader(val_sharded_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=2)

# Check a batch of training data from the sharded dataset
for images, labels in train_loader:
    print(f"Sharded batch image shape: {images.shape}, Sharded batch label shape: {labels.shape}")
    break

# ===
# Eval
# ===
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F


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

def calculate_metrics(model, dataloader, device):
    model.eval()

    all_true_masks = []
    all_pred_masks = []
    image_sizes = []  # Store original image sizes

    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader), desc='Evaluating'):
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
    return {"Mean Average Precision (AP@[IoU=0.50:0.95])" : map_result}


# ===
# Model Training
# ===
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import time
ROOT_SAVE_DIR = "/scratch/rawhad/CSE507/practice_3/models"
MODEL_SAVE_DIR = f"segmentation_model_{use_pretrained}_{learning_rate}_{num_epochs}_{int(time.time())}"
MODEL_PATH = os.path.join(ROOT_SAVE_DIR, MODEL_SAVE_DIR)
os.makedirs(MODEL_PATH, exist_ok=True)
LOGGER = logger.WandbLogger(project_name='cse507_practice3', run_name=MODEL_SAVE_DIR)

def train_one_epoch(model, dataloader, optimizer, device, logger, offset: int = 0):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
          outputs = model(pixel_values=images)
          logits = outputs.masks_queries_logits
          logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
          logits = logits.squeeze(1)  # Remove extra dimension (B, 1, H, W) -> (B, H, W)
          loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        log_data = dict(train_loss=loss.item())
        if ((i+1) % 500) == 0:
          print(f'{i} / {len(dataloader)} steps done')
          model.save_pretrained(MODEL_PATH)
          print('Model saved at:', MODEL_PATH)
          print('Evaluating')
          log_data.update(calculate_metrics(model, val_loader, device))
        if logger: logger.log(log_data, step=offset+i)

    avg_loss = running_loss / len(dataloader)
    print(f"Training Loss: {avg_loss}")
    return avg_loss

def validate(model, dataloader, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
              outputs = model(pixel_values=images)
              logits = outputs.masks_queries_logits
              logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
              logits = logits.squeeze(1)
              loss = criterion(logits, labels)
            val_loss += loss.item()
            if (i % 1000) == 0:
              print(f'{i} / {len(dataloader)} validation steps done')

    avg_val_loss = val_loss / len(dataloader)
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(CLASS_MAPPING)
checkpoint_path = 'facebook/mask2former-swin-small-coco-instance'
if use_pretrained:
  model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_path, num_labels=num_classes, ignore_mismatched_sizes=True)
else:
  config = Mask2FormerConfig.from_pretrained(checkpoint_path)
  config.num_labels = num_classes
  model = Mask2FormerForUniversalSegmentation(config)

model.to(device)
#print('Compiling Model ...')
#model = torch.compile(model)
#print('Model Compiled!')
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
  print(f"Epoch {epoch+1}/{num_epochs}")
  train_loss = train_one_epoch(model, train_loader, optimizer, device, LOGGER, offset=epoch*len(train_loader))
  val_loss = validate(model, val_loader, device)

calculate_metrics(model, val_loader, device)

model.save_pretrained(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

