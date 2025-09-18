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
from torch.utils.data import DataLoader

from network_hed import HED, SimpleCNNBackbone, ResNet50Backbone

import utils


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Train a model on specified dataset")

    # Adding arguments
    parser.add_argument("--model_name", type=str, help="Name of the model to train")
    parser.add_argument("--sample", action="store_true", help="Whether to use a sample dataset")
    parser.add_argument("--model_type", type=str, default="UNET", help="Type of model to train")
    parser.add_argument("--backbone_dataset", type=str, default="SimpleCNN", help="Backbone for the model (if applicable)",choices=["SimpleCNN", "ImageNet","BigEarthNet"])
    parser.add_argument("--freeze_backbone", action="store_true", help="Whether to freeze the backbone parameters")
    parser.add_argument("--guidance",action="store_true", help="Whether to use guidance band or not")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    #parser.add_argument("--loss", type=str, default="BCEWithLogitsLoss", help="Loss function to use")
    #parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--split", type=float, default=0.7, help="Train/Validation split")
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
    

    # Train the model
    for guidance in [True, False]:
        print(f"\n--- Training with Guidance Band: {guidance} ---")
        args.guidance = guidance

        train_loader, valid_loader = load_data(args)
        data_sense_check(train_loader)
        
        for loss in ["wBCE","DICE"]:
            print(f"\n--- Training with Loss: {loss} ---")
            args.loss = loss

            if args.backbone_dataset == "SimpleCNN":
                train_model(train_loader, valid_loader, args)
            else:
                for freeze in [True, False]:
                    print(f"\n--- Training with Freeze Backbone: {freeze} ---")
                    args.freeze_backbone = freeze
                    train_model(train_loader, valid_loader, args)
    

def data_sense_check(loader):
    """Check the data loader by iterating through it and printing the shapes of the images and targets"""

    print("\nData sense check:")
    for i, (images, target) in enumerate(loader):
        print("Batch {}:".format(i))
        print("Image shape:", images.shape)
        if images.shape[1] == 5:
            print("Guidnance band unique values:", torch.unique(images[:, 4, :, :]))
        print("Min:", images.min(), "Max:", images.max())
        print("Target shape:", target.shape)
        print("Target unique values:", torch.unique(target))
        print("Target %:", torch.sum(target) / target.numel())

        if i == 1:
            break  # Just check the first two batches
    print()


# Classes
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths,guidance=False):
        self.paths = paths
        self.guidance = guidance

    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
        instance = np.load(path)

        bands = instance[0:4] # Blue, Green, Red, NIR bands
        bands = utils.scale_bands(bands, satellite='sentinel')

        if self.guidance:
            # Use all 5 bands (Blue, Green, Red, NIR, Guidance)
            bands = np.concatenate((bands, instance[4:5]), axis=0)  # Add Guidance band
       
        bands = bands.astype(np.float32) 

        # Convert to tensor
        bands = torch.tensor(bands)

        # Get target
        mask = instance[-1].astype(np.int8)  # Binary edge mask
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
    train_data = TrainDataset(paths[:split], guidance=args.guidance)
    valid_data = TrainDataset(paths[split:], guidance=args.guidance)

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
    if args.guidance:
        # If using guidance band, we have 5 bands (Blue, Green, Red, NIR, Guidance)
        n_bands = 5
    else:
        n_bands = 4

    if args.model_type == "UNET":
        model = U_Net(n_bands, out_channels)
    elif args.model_type == "ATTUNET":
        model = AttU_Net(n_bands, out_channels)
    elif args.model_type == "HED":

        if args.backbone_dataset == "SimpleCNN":
            backbone = SimpleCNNBackbone(in_channels=n_bands)
        else:
            # Use ResNet50 backbone for ImageNet or BigEarthNet
            backbone = ResNet50Backbone(in_channels=n_bands,
                                        backbone_dataset=args.backbone_dataset,
                                        freeze_backbone=args.freeze_backbone)
 
        model = HED(backbone=backbone, 
                    in_channels=n_bands,
                    out_channels=out_channels)
    else:
        raise ValueError("Unsupported model type")

    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

   
    if args.loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()

    elif args.loss == "wBCE":
        pos_weight = 166 # weight given to edge pixels
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

    learning_rates = [0.1, 0.01, 0.001,0.0001]
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
                s1 = "frozen" if args.freeze_backbone else "trainable"
                
                s2 = "guided" if args.guidance else "unguided"
                
                model_name = f"{args.model_name}_{args.model_type}_{args.backbone_dataset}_{s1}_{s2}_{args.loss}.pth"
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
