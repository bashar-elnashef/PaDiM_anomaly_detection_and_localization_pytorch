"""
Provides utility functions for anomaly detection.
"""

import numpy as np
import torch
from typing import List, Optional, Callable, Union
from torchvision import transforms as T
from PIL import Image
import os
import pickle
import glob
import cv2
 
image_trsfm = T.Compose([T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        ])

mask_trsfm = T.Compose([T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor()
                        ])


def read_images(image_paths: List[str]) -> List:
    """
    Read and return a list of images from the given list of image file paths.

    Args:
        image_paths (List[str]): A list of image file paths.

    Returns:
        List: A list of images loaded from the given image paths.
    """
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)  # Read the image using OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


def to_batch(images: List[np.ndarray], transforms: T.Compose, device: torch.device) -> torch.Tensor:
    """Convert a list of numpy array images to a pytorch tensor batch with given transforms."""
    assert images

    transformed_images = []
    for image in images:
        image = Image.fromarray(image).convert('RGB')
        transformed_images.append(transforms(image))

    height, width = transformed_images[0].shape[1:3]
    batch = torch.zeros((len(images), 3, height, width))

    for i, transformed_image in enumerate(transformed_images):
        batch[i] = transformed_image

    return batch.to(device)


# From: https://github.com/pytorch/pytorch/issues/19037
def pytorch_cov(tensor: torch.Tensor, rowvar: bool = True, bias: bool = False) -> torch.Tensor:
    """Estimate a covariance matrix (np.cov)."""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bias))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def mahalanobis(mean: torch.Tensor, cov_inv: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Calculate the mahalonobis distance

    Calculate the mahalanobis distance between a multivariate normal distribution
    and a point or elementwise between a set of distributions and a set of points.

    Args:
        mean: A mean vector or a set of mean vectors.
        cov_inv: A inverse of covariance matrix or a set of covariance matricies.
        batch: A point or a set of points.

    Returns:
        mahalonobis_distance: A distance or a set of distances or a set of sets of distances.

    """

    # Assert that parameters has acceptable dimensions
    assert len(mean.shape) in {
        1,
        2,
    }, 'mean must be a vector or a set of vectors (matrix)'
    assert len(batch.shape) in {
        1,
        2,
        3,
    }, 'batch must be a vector or a set of vectors (matrix) or a set of sets of vectors (3d tensor)'
    assert len(cov_inv.shape) in {
        2,
        3,
    }, 'cov_inv must be a matrix or a set of matrices (3d tensor)'

    # Standardize the dimensions
    if len(mean.shape) == 1:
        mean = mean.unsqueeze(0)
    if len(cov_inv.shape) == 2:
        cov_inv = cov_inv.unsqueeze(0)
    if len(batch.shape) == 1:
        batch = batch.unsqueeze(0)
    if len(batch.shape) == 3:
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2])

    # Assert that parameters has acceptable shapes
    assert mean.shape[0] == cov_inv.shape[0]
    assert mean.shape[1] == cov_inv.shape[1] == cov_inv.shape[2] == batch.shape[1]
    assert batch.shape[0] % mean.shape[0] == 0

    # Set shape variables
    mini_batch_size, length = mean.shape
    batch_size = batch.shape[0]
    ratio = int(batch_size/mini_batch_size)

    # If a set of sets of distances is to be computed, expand mean and cov_inv
    if batch_size > mini_batch_size:
        mean = mean.unsqueeze(0)
        mean = mean.expand(ratio, mini_batch_size, length)
        mean = mean.reshape(batch_size, length)
        cov_inv = cov_inv.unsqueeze(0)
        cov_inv = cov_inv.expand(ratio, mini_batch_size, length, length)
        cov_inv = cov_inv.reshape(batch_size, length, length)

    # Make sure tensors are correct type
    mean = mean.float()
    cov_inv = cov_inv.float()
    batch = batch.float()

    # Calculate mahalanobis distance
    diff = mean-batch
    mult1 = torch.bmm(diff.unsqueeze(1), cov_inv)
    mult2 = torch.bmm(mult1, diff.unsqueeze(2))
    sqrt = torch.sqrt(mult2)
    mahalanobis_distance = sqrt.reshape(batch_size)

    # If a set of sets of distances is to be computed, reshape output
    if batch_size > mini_batch_size:
        mahalanobis_distance = mahalanobis_distance.reshape(ratio, mini_batch_size)

    return mahalanobis_distance


def image_score(patch_scores: torch.Tensor) -> torch.Tensor:
    """Calculate image scores from patch scores.

    Args:
        patch_scores: A batch of patch scores.

    Returns:
        image_scores: A batch of image scores.

    """

    return torch.max(patch_scores.reshape(patch_scores.shape[0], -1), -1).values


def classification(image_scores: torch.Tensor, thresh: float) -> torch.Tensor:
    """Calculate image classifications from image scores.

    Args:
        image_scores: A batch of image scores.
        thresh: A treshold value. If an image score is larger than
                or equal to thresh it is classified as anomalous.

    Returns:
        image_classifications: A batch of image classifcations.

    """
    # Apply threshold
    image_classifications = image_scores.clone()
    image_classifications[image_classifications < thresh] = 1
    image_classifications[image_classifications >= thresh] = 0
    return image_classifications

 
def rename_files(
            source_path: str,
            destination_path: Optional[str] = None
        ) -> None:
    """Rename all files in a directory path with increasing integer name.
    Ex. 0001.png, 0002.png ...
    Write files to destination path if argument is given.

    Args:
        source_path: Path to folder.
        destination_path: Path to folder.

    """
    for count, filename in enumerate(os.listdir(source_path), 1):
        file_source_path = os.path.join(source_path, filename)
        file_extension = os.path.splitext(filename)[1]

        new_name = str(count).zfill(4) + file_extension
        if destination_path:
            new_destination = os.path.join(destination_path, new_name)
        else:
            new_destination = os.path.join(source_path, new_name)

        os.rename(file_source_path, new_destination)


def split_tensor_and_run_function(
            func: Callable[[torch.Tensor], List],
            tensor: torch.Tensor,
            split_size: Union[int, List]
        ) -> torch.Tensor:
    """Splits the tensor into chunks in given split_size and run a function on each chunk.

    Args:
        func: Function to be run on a chunk of tensor.
        tensor: Tensor to split.
        split_size: Size of a single chunk or list of sizes for each chunk.

    Returns:
        output_tensor: Tensor of same size as input tensor

    """
    tensors_list = [
        func(sub_tensor) for sub_tensor in torch.split(tensor, split_size)
    ]
    return torch.cat(tensors_list)


def save_pickle(file: any, filepath: str) -> None:
    """
    Save data as a pickle file at the specified filepath. Ensure the directory exists.

    Args:
        file (any): The data to be saved.
        filepath (str): The file path where the data should be saved as a pickle file.

    Returns:
        None: This function does not return a value.

    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)
    return 

def load_pickle(filepath: str) -> any:
    """
    Load data from a pickle file at the specified filepath.

    Args:
        filepath (str): The file path from which to load the data as a pickle file.
    Returns:
        any: The loaded data.

    Raises:
        FileNotFoundError: If the file specified by `filepath` does not exist.
    """
    if not os.path.exists(filepath):
        msg = f"The file '{filepath}' does not exist."
        raise FileNotFoundError(msg)
    with open(filepath, 'rb') as f:
        loaded = pickle.load(f)
    return loaded


def get_image_paths(directory: str) -> List[str]:
    """
    Get a list of image file paths within a directory.

    Args:
        directory (str): The path to the directory where you want to find image files.

    Returns:
        List[str]: A list of image file paths found in the directory.

    """
    # Define a list of image file extensions (you can add more if needed)
    extension = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']

    # Initialize an empty list to store image paths
    image_paths = []

    # Use glob to find image files within the directory
    for ext in extension:
        pattern = os.path.join(directory, f'*.{ext}')
        image_paths.extend(glob.glob(pattern))

    return image_paths
