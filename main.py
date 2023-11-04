import os
from model.padim import PADIM
from datasets.dataset import WaferDataset
import utils.util as util
import numpy as np
import torch
import cv2
import pickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from visualization import boundary, frame, heatmap, highlight, vis_utils
import argparse
import random
import torch.nn as nn
from itertools import chain

def parse_args():
    parser = argparse.ArgumentParser('Wafer anomaly detection and localization')
    parser.add_argument('--data_path', type=str, default='datasets/wafer')
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'wide_resnet50'], default='wide_resnet50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--class_name', type=str, default='test0')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_stats', type=bool, default=True)
    parser.add_argument('--distributions', type=str,  default='./distributions/')
    parser.add_argument('--train', type=bool,  default=False)

    return parser.parse_args()

def train(args: argparse.Namespace,
          model_data_path: str,
          dataset_path: str, 
          device: torch.device,
          ) -> nn.Module:
    """
    Train the PADIM model.

    Args:
        args (argparse.Namespace): The command line arguments.
        model_data_path (str): The path to the model data.
        dataset_path (str): The path to the dataset.
        device (torch.device): The device to run the model on.

    Returns:
        nn.Module: The trained PADIM model.
    """

    # init dataset and dataloader 
    dataset = WaferDataset(dataset_path=dataset_path, class_name=args.class_name)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    print(f'[INFO] Number of images in dataset: {len(dataloader.dataset)}')

    mean_path = os.path.join(model_data_path, f'{args.class_name}_mean.pt')
    cov_inv_path = os.path.join(model_data_path, f'{args.class_name}_cov_inv.pt')

    if args.train:
        # Create a model
        padim = PADIM(backbone=args.backbone, device=device)
        padim.fit(dataloader)
        if args.save_stats:
            # save_stat_data(model_data_path, args.class_name, padim)
            print(f'[INFO] Saving stats to {len(model_data_path)}')
            util.save_pickle(padim.mean, mean_path)
            util.save_pickle(padim.cov_inv, cov_inv_path)
    else:
        # Load stats
        mean = util.load_pickle(mean_path)
        cov_inv = util.load_pickle(cov_inv_path)
        # Create a model
        padim = PADIM(backbone=args.backbone, mean=mean, cov_inv=cov_inv, device=device)

    return padim


def test(args: argparse.Namespace,
         model_data_path: str,
         dataset_path: str, 
         device: torch.device,
         model: torch.nn.Module
         ) -> None:
    """
    Test the PADIM model.

    Args:
        args (argparse.Namespace): The command line arguments.
        model_data_path (str): The path to the model data.
        dataset_path (str): The path to the dataset.
        device (torch.device): The device to run the model on.
        model (torch.nn.Module): The PADIM model to be tested.

    Returns:
        None
    """
    dataset = WaferDataset(dataset_path=dataset_path, class_name=args.class_name, 
                                                    is_train=False, defected_only=True)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # TODO: Change format of score_maps and masks_target 
    images, image_classifications_target, \
             masks_target, image_scores, score_maps = model.evaluate(dataloader)

    # TODO: Save results into the output folder.
    # TODO: Add additional visualization options.


def main():
    """
    Main function that performs wafer anomaly detection.

    Returns:
        None
    """
    args = parse_args()
    dataset_path = os.path.realpath(args.data_path)
    model_data_path = os.path.realpath(args.distributions)

    # Set device
    device = torch.device(args.device)
    print(f'[INFO] device: {device}')
    
    # Set seed
    random.seed(1024)
    torch.manual_seed(1024)
    if args.device == 'cuda': torch.cuda.manual_seed_all(1024)  
    
    padim = train(args=args, model_data_path=model_data_path, 
                            dataset_path=dataset_path, device=device)


    test(args=args, model_data_path=model_data_path, 
                            dataset_path=dataset_path, device=device, model=padim)


if __name__ == '__main__':  
    main()





