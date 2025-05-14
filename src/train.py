# Training unet for coastline detection
# Conor O'Sullivan
# 07 Feb 2023

# Imports
import numpy as np
import pandas as pd
import random
import glob
import argparse
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from network_unet import U_Net, AttU_Net
from network_hed import HED, SimpleCNNBackbone, EfficientNetBackbone

import utils


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Train a model on specified dataset")

    # Adding arguments
    parser.add_argument("--model_name", type=str, help="Name of the model to train")
    parser.add_argument("--sample", action="store_true", help="Whether to use a sample dataset")
    parser.add_argument("--model_type", type=str, default="UNET", help="Type of model to train")
    parser.add_argument("--backbone", type=str, default="SimpleCNN", help="Backbone for the model (if applicable)",choices=["SimpleCNN", "ImageNet"])
    parser.add_argument("--freeze_backbone", action="store_true", help="Whether to freeze the backbone parameters")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    #parser.add_argument("--loss", type=str, default="BCEWithLogitsLoss", help="Loss function to use")
    #parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--split", type=float, default=0.9, help="Train/Validation split")
    parser.add_argument("--early_stopping", type=int, default=-1, help="Number of epochs to wait before stopping training. -1 to disable.")

    parser.add_argument("--train_path", type=str, default="../data/training/", help="Path to the training data")
    parser.add_argument("--save_path", type=str, default="../models/", help="Path template for saving the model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset")

    # Parse the arguments
    args = parser.parse_args()

    # Set device based on argument
    args.device = torch.device(args.device)

    print("\nTraining model with the following arguments:")
    print(vars(args))  # Print all arguments

    # Load data
    train_loader, valid_loader = load_data(args)
    data_sense_check(train_loader)

    # Train the model
    for loss in ["wBCE","DICE"]:
        print(f"\n--- Training with Loss: {loss} ---")
        args.loss = loss
        train_model(train_loader, valid_loader, args)
    

def data_sense_check(loader):
    """Check the data loader by iterating through it and printing the shapes of the images and targets"""

    print("\nData sense check:")
    for i, (images, target) in enumerate(loader):
        print("Batch {}:".format(i))
        print("Image shape:", images.shape)
        print("Min:", images.min(), "Max:", images.max())
        print("Target shape:", target.shape)
        print("Target unique values:", torch.unique(target))

        if i == 1:
            break  # Just check the first two batches
    print()


# Classes
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
        instance = np.load(path)

        # Get spectral bands
        bands = instance[0:4] # Blue, Green, Red, NIR
        bands = bands.astype(np.float32) 

        # Normalise bands
        bands = utils.scale_bands(bands, satellite='sentinel')

        # Convert to tensor
        bands = torch.tensor(bands)

        # Get target
        mask = instance[4].astype(np.int8)  # Binary edge mask
        target = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return bands, target

    def __len__(self):
        return len(self.paths)


# Functions
def load_data(args):
    """Load data from disk"""

    # Correct path format
    if args.train_path[-1] != "/":
        args.train_path += "/"

    paths = glob.glob(args.train_path + "*.npy")
    print("\nTotal images: {}".format(len(paths)))

    if args.sample:
        paths = paths[:100]

    # Shuffle the paths
    random.seed(args.seed)
    random.shuffle(paths)

    # Create datasets
    split = int(args.split * len(paths))
    train_data = TrainDataset(paths[:split])
    valid_data = TrainDataset(paths[split:])

    # Prepare data for PyTorch model
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)

    return train_loader, valid_loader

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation with sigmoid"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)


def initialize_model(args, lr):
    """Initialize the model, optimizer and criterion based on the specified model type"""

    out_channels = 1
    n_bands = 4

    if args.model_type == "UNET":
        model = U_Net(n_bands, out_channels)
    elif args.model_type == "ATTUNET":
        model = AttU_Net(n_bands, out_channels)
    elif args.model_type == "HED":

        if args.backbone == "SimpleCNN":
            backbone = SimpleCNNBackbone(in_channels=n_bands)
        elif args.backbone == "ImageNet":
            backbone = EfficientNetBackbone(in_channels=n_bands)
        
        model = HED(backbone=backbone, out_channels=out_channels,
                                        freeze_backbone=args.freeze_backbone)
    else:
        raise ValueError("Unsupported model type")

    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

   
    if args.loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()

    elif args.loss == "wBCE":
        pos_weight = 183 # weight given to edge pixels
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(args.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    elif args.loss == "DICE":
        criterion = DiceLoss()

    return model, optimizer, criterion

def compute_loss(output, target, criterion):
    """Compute the loss for the model"""
    if isinstance(output, list):
        # If the model outputs a list of side outputs (i.e. HED)
        loss = sum(criterion(out, target) for out in output)
    else:
        # If the model outputs a single tensor (i.e. UNET)
        loss = criterion(output, target)
    return loss


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""

    model.train()
    for images, target in train_loader:
        images = images.to(device)
        target = target.to(device).float()

        optimizer.zero_grad()
        output = model(images)

        loss = compute_loss(output, target, criterion)
        
        loss.backward()
        optimizer.step()


def evaluate_model(model, valid_loader, criterion, device):
    """Evaluate the model on the validation set"""
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, target in valid_loader:
            images = images.to(device)
            target = target.to(device).float()
            output = model(images)
            valid_loss += compute_loss(output, target, criterion).item()

    return valid_loss / len(valid_loader)


def train_model(train_loader, valid_loader, args):
    """Train the model with different learning rates and save the best model"""

    learning_rates = [0.1, 0.01, 0.001]
    global_min_loss = np.inf # Compare across all learning rates

    for lr in learning_rates:
        print(f"\n--- Training with Learning Rate: {lr} ---")

        model, optimizer, criterion = initialize_model(args, lr)

        min_loss = np.inf # Compare within this learning rate
        epochs_no_improve = 0 # Early stopping counter
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1} | ", end=" ")

            train_one_epoch(model, train_loader, criterion, optimizer, args.device)
            valid_loss = evaluate_model(model, valid_loader, criterion, args.device)

            print(f"Validation Loss: {valid_loss:.5f}")

            if valid_loss < global_min_loss:
                # Save the model with the best validation loss across all learning rates
                model_name = f"{args.model_name}_{args.model_type}_{args.backbone}_{args.freeze_backbone}_{args.loss}.pth"
                save_path = os.path.join(args.save_path, model_name)
                torch.save(model.state_dict(), save_path)
                print("Model saved (best across all LRs).")
                global_min_loss = valid_loss

            # Update early stopping counter
            if valid_loss < min_loss:
                min_loss = valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
