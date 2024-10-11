import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, List
from PIL import Image
import pyarrow.parquet as pq
import torch.nn.functional as F
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Batch size and chunk size for lazy loading
batch_size = 1000
chunk_size = 5000  # Number of rows to read per chunk from parquet

# Lazy Loading Parquet with PyArrow
class LazyLoadingSegmentationDataset(Dataset):
    def __init__(self, dataset: DS, class_mapping: dict[str, int], transform: Optional[Callable] = None, img_height: int | None = None, img_width: int | None = None):
        self.dataset = dataset
        self.transform = transform
        self.image_height = img_height
        self.image_width = img_width
        self.classes = class_mapping
        self.data_chunks = pq.ParquetFile(dataset.csv_path)  # Initialize the PyArrow Parquet file reader
        self.num_rows = self.data_chunks.metadata.num_rows  # Total number of rows

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, idx: int) -> tuple:
        chunk_idx = idx // chunk_size
        offset = idx % chunk_size
        
        # Read a chunk lazily from the parquet file
        df_chunk = self.data_chunks.read_row_group(chunk_idx, columns=[self.dataset.col_name, 'Height', 'Width', 'Right Lung', 'Left Lung', 'Heart']).to_pandas()

        example = df_chunk.iloc[offset]
        img_name = example[self.dataset.col_name]
        img_path = self.dataset.get_image_path(img_name)

        height, width = example['Height'], example['Width']

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

def lazy_load_datasets(datasets: List[DS], transform: Optional[Callable] = None, img_height: int | None = None, img_width: int | None = None) -> List[LazyLoadingSegmentationDataset]:
    torch_datasets = []
    for dataset in datasets:
        logging.info(f"Initializing lazy loading for dataset: {dataset.name}")
        ds = LazyLoadingSegmentationDataset(
            dataset=dataset, 
            transform=transform, 
            img_height=img_height, 
            img_width=img_width, 
            class_mapping=CLASS_MAPPING
        )
        torch_datasets.append(ds)
    return torch_datasets

# Initialize lazy-loaded datasets
lazy_datasets = lazy_load_datasets(
    datasets=[chestxray14_ds, padchest_ds, chexpert_ds, vindr_cxr_ds],
    transform=preprocess,
    img_height=TRAIN_IMAGE_HEIGHT,
    img_width=TRAIN_IMAGE_WIDTH
)

class UnifiedLazySegmentationDataset(Dataset):
    def __init__(self, datasets: List[LazyLoadingSegmentationDataset], transform: Optional[Callable] = None):
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.total_length = sum(self.dataset_lengths)

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> tuple:
        for ds_index, length in enumerate(self.dataset_lengths):
            if idx < length:
                logging.info(f"Fetching data from dataset {self.datasets[ds_index].dataset.name} at index {idx}")
                return self.datasets[ds_index][idx]
            idx -= length
        raise IndexError("Index out of range for the combined dataset")


# Initialize unified dataset
unified_lazy_dataset = UnifiedLazySegmentationDataset(datasets=lazy_datasets)

# DataLoader with batch-level shuffling
unified_loader = DataLoader(unified_lazy_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Saving shards
shard_dir = '/scratch/rawhad/CSE507/practice_3/shards'
os.makedirs(shard_dir, exist_ok=True)
shard_count = 0

logging.info("Starting to save shards")
for images, labels in tqdm(unified_loader, desc="Saving shards"):
    labels = labels.unsqueeze(1)  # (B, 1, H, W)
    stacked = torch.cat((images, labels.float()), dim=1)  # (B, 4, H, W)
    np_stacked = stacked.numpy()
    
    shard_path = os.path.join(shard_dir, f"shard_{shard_count:04d}.npy")
    np.save(shard_path, np_stacked)
    
    logging.info(f"Shard {shard_count} saved with shape {np_stacked.shape}")
    shard_count += 1
