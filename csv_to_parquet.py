import argparse
import dataclasses
import json
import os
import pandas as pd
import warnings
from typing import Callable, Dict, List


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

# Define the file paths
def to_parquet(csv_file_path):
  parquet_file_path = os.path.splitext(csv_file_path)[0] + '.parquet'
  df = pd.read_csv(csv_file_path)
  df.to_parquet(parquet_file_path)
  print(f"CSV file successfully converted to Parquet: {parquet_file_path}")

# convert csvs to parquets
for ds in datasets_dict.values(): to_parquet(ds.csv_path)
