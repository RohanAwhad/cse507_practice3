import argparse
import dataclasses
import json
import os
import pandas as pd
import warnings
from tqdm import tqdm
from typing import Callable, Dict, List

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

def get_files_with_absolute_paths(root_dir: str, index_name: str) -> dict[str, str]:
    index_path = f'/home/rawhad/CSE507/practice_3/{index_name}_index.json'
    if os.path.exists(index_path):
        with open(index_path, 'r') as f: return json.load(f)
    file_dict: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            absolute_path = os.path.abspath(os.path.join(dirpath, filename))
            file_dict[filename] = absolute_path
    with open(index_path, 'w') as f: json.dump(file_dict, f)
    return file_dict
chestxray14_index = get_files_with_absolute_paths('/data/courses/2024/class_ImageSummerFall2024_jliang12/chestxray14', 'chestxray14')
print('Chest XRay 14 index loaded')
padchest_index = get_files_with_absolute_paths('/data/jliang12/shared/dataset/PadChest/image_zips', 'padchest')
print('PadChest index loaded')


@dataclasses.dataclass
class DS:
    name: str
    csv_path: str
    col_name: str
    get_image_path: Callable[[str], str]

csv_directory: str = "/data/courses/2024/class_ImageSummerFall2024_jliang12/CheXMask/physionet.org/files/chexmask-cxr-segmentation-data/0.4/OriginalResolution"
chestxray14_ds = DS(
    name="chestxray14",
    csv_path=os.path.join(csv_directory, "ChestX-Ray8.parquet"),
    col_name="Image Index",
    get_image_path=lambda x: chestxray14_index.get(x, None)
)
padchest_ds = DS(
    name="padchest",
    csv_path=os.path.join(csv_directory, "Padchest.parquet"),
    col_name="ImageID",
    get_image_path=lambda x: padchest_index.get(x, None)
)

def get_image_path(path: str) -> str | None: return path if os.path.exists(path) else None
chexpert_ds = DS(
    name="chexpert",
    csv_path=os.path.join(csv_directory, "CheXpert.parquet"),
    col_name="Path",
    get_image_path=lambda x: get_image_path(f"/data/courses/2024/class_ImageSummerFall2024_jliang12/chexpertchestxrays-u20210408/CheXpert-v1.0/{x}")
)

vindr_cxr_ds = DS(
    name="vindr_cxr",
    csv_path=os.path.join(csv_directory, "VinDr-CXR.parquet"),
    col_name="image_id",
    get_image_path=lambda x: get_image_path(f"/data/courses/2024/class_ImageSummerFall2024_jliang12/vinbigdata/train/{x}.dicom"),
)

datasets_dict: Dict[str, DS] = {
    chestxray14_ds.name: chestxray14_ds,
    padchest_ds.name: padchest_ds,
    chexpert_ds.name: chexpert_ds,
    vindr_cxr_ds.name: vindr_cxr_ds,
}

# ===
# Dataset Class
# ===
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from pandas import DataFrame
from typing import Optional, Callable
from torchvision import transforms
from torch.utils.data import DataLoader, random_split



def convert_to_rgb(image_path: str) -> Image.Image:
  # Check if file is a DICOM file
  if image_path.endswith('.dicom'):
    dicom_image: pydicom.dataset.FileDataset = pydicom.dcmread(image_path)
    image_array: np.ndarray = dicom_image.pixel_array
    image: Image.Image = Image.fromarray(image_array)
    image = image.convert('RGB')
  else:
    image: Image.Image = Image.open(image_path)
    if image.mode == 'I;16':
      image = (np.array(image) / 256).astype(np.uint8)
      image = Image.fromarray(image)
    if image.mode == 'L':
      image = image.convert('RGB')
    elif image.mode == 'RGBA':
      image = image.convert('RGB')
  return image

def get_mask_from_RLE(rle, height, width):
    runs = np.array([int(x) for x in rle.split()])
    starts = runs[::2]
    lengths = runs[1::2]
    mask = np.zeros((height * width), dtype=np.uint8)
    for start, length in zip(starts, lengths):
        start -= 1  
        end = start + length
        mask[start:end] = 255
    mask = mask.reshape((height, width))
    return mask

class SegmentationDataset(Dataset):
  def __init__(self, dataset: DS, class_mapping: dict[str, int], transform: Optional[Callable] = None, img_height: int | None = None, img_width: int | None = None):
    self.dataset = dataset
    self.image_height = img_height
    self.image_width = img_width

    self.csv_data = pd.read_parquet(dataset.csv_path)
    #csv_data: pd.DataFrame = pd.read_csv(dataset.csv_path, nrows=20)
    #self.csv_data: pd.DataFrame = csv_data.sample(10)
    #warnings.warn("Only using 10 random rows from the first 20 rows of the dataset")

    self.transform = transform
    self.img_names: List[str] = self.csv_data[dataset.col_name].tolist()
    self.classes = class_mapping

  def __len__(self) -> int:
    return len(self.img_names)

  def __getitem__(self, idx: int) -> tuple:
    img_name: str = self.img_names[idx]
    img_path: Optional[str] = self.dataset.get_image_path(img_name)

    example = self.csv_data.iloc[idx]
    height: int = example['Height']
    width: int = example['Width']

    if img_path is None:
      image = Image.new('RGB', (width, height), color=255)
      label = torch.tensor(np.zeros((height, width), dtype=np.uint8), dtype=torch.long)
    else:
      rightLungMask = get_mask_from_RLE(example['Right Lung'], height, width)
      leftLungMask = get_mask_from_RLE(example['Left Lung'], height, width)
      heartMask = get_mask_from_RLE(example['Heart'], height, width)

      label = np.zeros((height, width), dtype=np.uint8)
      label[rightLungMask == 255] = self.classes['right_lung']
      label[leftLungMask == 255] = self.classes['left_lung']
      label[heartMask == 255] = self.classes['heart']
      label = torch.tensor(label, dtype=torch.long)

      image = convert_to_rgb(img_path)
    if self.transform:
      image = self.transform(image)
      label = F.interpolate(label.float().unsqueeze(0).unsqueeze(0), size=(self.image_height, self.image_width), mode='nearest').squeeze(0).squeeze(0).long()
    return image, label


# Dataset Constants
TRAIN_IMAGE_WIDTH = 512
TRAIN_IMAGE_HEIGHT = 512
CLASS_MAPPING = {'background': 0, 'right_lung': 1, 'left_lung': 2, 'heart': 3}

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Using ImageNet statistics
])

# Initialize datasets
chestxray14_torch_ds = SegmentationDataset(dataset=chestxray14_ds, transform=preprocess, img_height=TRAIN_IMAGE_HEIGHT, img_width=TRAIN_IMAGE_WIDTH, class_mapping=CLASS_MAPPING)
padchest_torch_ds = SegmentationDataset(dataset=padchest_ds, transform=preprocess, img_height=TRAIN_IMAGE_HEIGHT, img_width=TRAIN_IMAGE_WIDTH, class_mapping=CLASS_MAPPING)
chexpert_torch_ds = SegmentationDataset(dataset=chexpert_ds, transform=preprocess, img_height=TRAIN_IMAGE_HEIGHT, img_width=TRAIN_IMAGE_WIDTH, class_mapping=CLASS_MAPPING)
vindr_cxr_torch_ds = SegmentationDataset(dataset=vindr_cxr_ds, transform=preprocess, img_height=TRAIN_IMAGE_HEIGHT, img_width=TRAIN_IMAGE_WIDTH, class_mapping=CLASS_MAPPING)


class UnifiedSegmentationDataset(Dataset):
  def __init__(self, datasets: List[SegmentationDataset], transform: Optional[Callable] = None):
    self.datasets = datasets
    self.transform = transform
    self.dataset_lengths = [len(ds) for ds in datasets]
    self.total_length = sum(self.dataset_lengths)

  def __len__(self) -> int:
    return self.total_length

  def __getitem__(self, idx: int) -> tuple:
    for ds_index, length in enumerate(self.dataset_lengths):
      if idx < length:
        return self.datasets[ds_index][idx]
      idx -= length
    # If the index is out of bounds
    raise IndexError("Index out of range for the combined dataset")



# Initialize unified dataset
unified_dataset = UnifiedSegmentationDataset(
  datasets=[chestxray14_torch_ds, padchest_torch_ds, chexpert_torch_ds, vindr_cxr_torch_ds]
)

val_size = batch_size * 1000
train_size = len(unified_dataset) - val_size
train_dataset, val_dataset = random_split(unified_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=16)

# Check a batch of training data
for images, labels in train_loader:
    print(f"Batch image shape: {images.shape}, Batch label shape: {labels.shape}")
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
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.cpu().numpy()  # Keep labels on CPU for metrics

            # Get original image sizes (before preprocessing)
            for image in images:
                original_size = (image.shape[-1], image.shape[-2])  # (width, height)
                image_sizes.append(original_size)

            # Get model predictions
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
MODEL_SAVE_DIR = f"segmentation_model_{int(time.time())}_{learning_rate}_{num_epochs}_{use_pretrained}"
MODEL_PATH = os.path.join(ROOT_SAVE_DIR, MODEL_SAVE_DIR)
os.makedirs(MODEL_PATH, exist_ok=True)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
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
        if (i % 1000) == 0:
          print(f'{i} / {len(dataloader)} steps done')
          print('Evaluating')
          calculate_metrics(model, val_loader, device)
          model.save_pretrained(MODEL_PATH)
          print('Model saved at:', MODEL_PATH)

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
checkpoint_path = 'facebook/mask2former-swin-large-coco-instance'
if use_pretrained:
  model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_path, num_labels=num_classes, ignore_mismatched_sizes=True)
else:
  config = Mask2FormerConfig.from_pretrained(checkpoint_path)
  config.num_labels = num_classes
  model = Mask2FormerForUniversalSegmentation(config)
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
  print(f"Epoch {epoch+1}/{num_epochs}")
  train_loss = train_one_epoch(model, train_loader, optimizer, device)
  val_loss = validate(model, val_loader, device)

calculate_metrics(model, val_loader, device)

model.save_pretrained(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
