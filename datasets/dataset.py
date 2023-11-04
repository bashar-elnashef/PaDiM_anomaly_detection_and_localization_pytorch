import os
import torch
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.util import image_trsfm, mask_trsfm
from torchvision import transforms as T
from typing import Tuple, List, Optional, Callable, Union
import numpy as np

CLASS_NAMES = ['test0']

class WaferDataset(Dataset):
    def __init__(self, 
                 dataset_path: str='datasets\wafer', 
                 class_name: str='test0', 
                 is_train: bool=True,
                 image_transforms: T.Compose=image_trsfm, 
                 mask_transforms: T.Compose=mask_trsfm, 
                 defected_only: bool=False) -> None:
        assert (
            class_name in CLASS_NAMES
        ), f'class_name: {class_name}, should be in {CLASS_NAMES}'

        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.defected_only = defected_only
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.image_transforms(x)

        if y == 0:
            mask = torch.zeros([1, x.shape[1], x.shape[2]])
        else:
            mask = Image.open(mask)
            mask = self.mask_transforms(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        print(img_types)
        for img_type in img_types:
            if self.defected_only and img_type == 'good':
                continue
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, f'{img_fname}_mask.png')
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
