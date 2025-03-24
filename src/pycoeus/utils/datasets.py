import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pycoeus.logging_config import setup_logger
from pycoeus.utils.io import read_geotiff

logger = setup_logger(__name__)

class MonochromeFlairDataset(Dataset):
    def __init__(self, root_path, limit=None, split="train"):
        self.root_path = root_path
        self.limit = limit
        self.images = sorted([str(p) for p in (Path(root_path) / split / "input").glob("*.tif")])[:self.limit]

        def image_path_to_mask_path(image_path: Path) -> Path:
            return image_path.parent.parent / "labels" / f"MSK{image_path.stem[3:-2]}_0{image_path.suffix}"  # -2 for "_b" where b is band#

        self.masks = [str(image_path_to_mask_path(Path(p))) for p in self.images][:self.limit]
        non_existing_masks = [p for p in self.masks if Path(p).exists() == False]
        if non_existing_masks:
            logging.getLogger().warning(f"{len(non_existing_masks)} of a total of {len(self.masks)} masks not found.")

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img_arr = read_geotiff(self.images[index]).data
        img = normalize_single_band_to_tensor(img_arr[0], debug_note=f"path: {self.images[index]}")[None, :, :]
        mask = load_and_one_hot_encode(self.masks[index])
        return img, mask

    def __len__(self):
        return min(len(self.images), self.limit)

def normalize_single_band_to_tensor(img_arr: np.ndarray, debug_note="") -> torch.Tensor:
    
    # Remove min and max values from the array
    img_arr_data = img_arr.flatten()
    data_min = img_arr_data.min()
    data_max = img_arr_data.max()

    if data_min != data_max:
        img_arr_data = img_arr_data[img_arr_data != data_min]
        img_arr_data = img_arr_data[img_arr_data != data_max]

    std = img_arr_data.std()
    if std == 0:
        msg = "Standard deviation = 0 " + debug_note
        logger.debug(msg)
        std = 1
    mean = img_arr_data.mean()
    normalized = (img_arr - mean) / std
    return torch.from_numpy(normalized)


def load_and_one_hot_encode(image_path, num_classes=19):
    """
    Loads a greyscale (labels) image from the specified path and performs one-hot encoding.

    Args:
        image_path (str): The file path to the image.
        num_classes (int, optional): The number of classes for one-hot encoding. Default is 20.

    Returns:
        torch.Tensor: A one-hot encoded tensor with shape (num_classes, height, width).
    """
    image = Image.open(image_path).convert("L")  # Load as grayscale

    image_array = np.array(image, dtype=np.int64) -1  # -1 to convert to zero-based
    image_tensor = torch.from_numpy(image_array)

    one_hot = torch.nn.functional.one_hot(image_tensor, num_classes=num_classes)

    return one_hot.permute(2, 0, 1).float()
