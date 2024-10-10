import argparse
import dataclasses
import json
import os
import pandas as pd
import warnings
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
    csv_path=os.path.join(csv_directory, "ChestX-Ray8.csv"),
    col_name="Image Index",
    get_image_path=lambda x: chestxray14_index.get(x, None)
)
padchest_ds = DS(
    name="padchest",
    csv_path=os.path.join(csv_directory, "Padchest.csv"),
    col_name="ImageID",
    get_image_path=lambda x: padchest_index.get(x, None)
)

chexpert_ds = DS(
    name="chexpert",
    csv_path=os.path.join(csv_directory, "CheXpert.csv"),
    col_name="Path",
    get_image_path=lambda x: f"/data/courses/2024/class_ImageSummerFall2024_jliang12/chexpertchestxrays-u20210408/CheXpert-v1.0/{x}"
)

vindr_cxr_ds = DS(
    name="vindr_cxr",
    csv_path=os.path.join(csv_directory, "VinDr-CXR.csv"),
    col_name="image_id",
    get_image_path=lambda x: f"/data/courses/2024/class_ImageSummerFall2024_jliang12/vinbigdata/train/{x}.dicom",
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
    #self.csv_data = pd.read_csv(dataset.csv_path)

    csv_data: pd.DataFrame = pd.read_csv(dataset.csv_path, nrows=20)
    self.csv_data: pd.DataFrame = csv_data.sample(10)
    warnings.warn("Only using 10 random rows from the first 20 rows of the dataset")
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
      image = Image.new('L', (width, height), color=255)
      label = np.zeros((height, width), dtype=np.uint8)
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
      label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=(self.image_height, self.image_width), mode='nearest').squeeze(0).squeeze(0)
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

# svae the first index image and label side by side
import matplotlib.pyplot as plt

def plot_sample(dataset: Dataset, idx: int):
  image, label = dataset[idx]
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  axes[0].imshow(image.permute(1, 2, 0))  # Convert CHW to HWC
  axes[0].set_title('Image')
  axes[0].axis('off')

  axes[1].imshow(label.numpy(), cmap='gray')
  axes[1].set_title('Label')
  axes[1].axis('off')

  plt.tight_layout()
  fig.savefig(f'sample_{idx}.png')
# Plot and save a sample from the unified dataset
plot_sample(unified_dataset, 0)
