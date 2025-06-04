from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import numpy as np
from train import *  # load dataset and model classes
import argparse

import utils

def shuffle_band(band):
    """Shuffle a single spectral band of an image"""

    perm_image = band.ravel()
    random.shuffle(perm_image)
    perm_image.resize(256, 256)

    return perm_image

def permutate_bands(img, bands):
    """Permuate bands in list of bands"""

    img = np.array(img)

    img = img.copy()

    band_dict = {
        "Blue": 0,
        "Green": 1,
        "Red": 2,
        "NIR": 3,
    }

    bands_i = [band_dict[b] for b in bands]

    for i in bands_i:
        img[i, :, :] = shuffle_band(img[i, :, :])

    img = torch.from_numpy(img).float()

    return img


def get_preds(model, 
              test_paths, 
              threshold=0.5,
              batch_size=10, 
              perm_bands=None,
              device='mps'):
    
    """Get model predictions for a given dataloader"""
    
    test_data = TrainDataset(test_paths)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    sm = nn.Sigmoid() # Sigmoid activation function

    arr_targets = []
    arr_preds = []

    model.eval()
    with torch.no_grad():

        for images, target in iter(test_loader):
            # permutate bands
            if perm_bands != None:
                for i in range(len(images)):
                    images[i] = permutate_bands(images[i], perm_bands)

            images = images.to(device)

            # Get model output
            output = model(images)

            # Handle HED model: use fused output
            if isinstance(output, list):
                output = output[-1]  # Use final fused output


            output = sm(output)

            pos = 0

            # Get model predictions
            probs = output.cpu().detach().numpy()
            probs =[np.array(p[pos]) for p in probs]

            #apply thresgold to all pixels
            preds = []
            for p in probs:
                pred = np.zeros_like(p)
                pred[p>threshold]=1
                preds.append(pred)
            preds = np.array(preds)
            
            # Get model predictions
            targets = [np.array(t[pos]) for t in target]

            arr_targets.extend(targets)
            arr_preds.extend(preds)

    return arr_targets, arr_preds


def calc_fom(img, ref_img, alpha=1.0 / 9.0):
    """
    Computes Pratt's Figure of Merit for the given image img, using a gold
    standard image as source of the ideal edge pixels.
    """

    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(1 - ref_img)

    N, M = img.shape
    fom = 0
    for i in range(N):
        for j in range(M):
            if img[i, j]:
                fom += 1.0 / (1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(np.count_nonzero(img), np.count_nonzero(ref_img))

    return fom


def calc_accuracy(preds, targets):
    preds = np.array(preds).ravel()
    targets = np.array(targets).ravel()

    return sum(targets == preds) / len(targets)


def confusion_metrics(pred, target):
    """Returns confusion matrix metrics"""

    TP = np.sum((pred == 1) & (target == 1))
    TN = np.sum((pred == 0) & (target == 0))
    FP = np.sum((pred == 1) & (target == 0))
    FN = np.sum((pred == 0) & (target == 1))

    return TP, TN, FP, FN


def eval_metrics(targets, preds):
    """Evaluate model performance on test set"""

    r_accuracy = []
    r_balanced_accuracy = []
    r_precision = []
    r_recall = []
    r_f1 = []
    r_iou = []
    r_fom = []
    # Calculate metrics for each image
    for i in range(len(targets)):
        target = np.array(targets[i])
        pred = np.array(
            preds[i],
        )

        # Confusion matrix metrics
        TP_, TN_, FP_, FN_ = confusion_metrics(pred, target)

        accuracy = (TP_ + TN_) / (TP_ + TN_ + FP_ + FN_)
        precision = TP_ / (TP_ + FP_)
        recall = TP_ / (TP_ + FN_)
        f1 = 2 * (precision * recall) / (precision + recall)
        iou = TP_ / (TP_ + FP_ + FN_)

        r_accuracy.append(accuracy)
        r_precision.append(precision)
        r_recall.append(recall)
        r_f1.append(f1)
        r_iou.append(iou)

        # Figure of Merit
        fom = calc_fom(pred, target)
        r_fom.append(fom)


    accuracy = np.mean(np.nan_to_num(r_accuracy, nan=0.0))
    precision = np.mean(np.nan_to_num(r_precision, nan=0.0))
    recall    = np.mean(np.nan_to_num(r_recall, nan=0.0))
    f1        = np.mean(np.nan_to_num(r_f1, nan=0.0))
    iou       = np.mean(np.nan_to_num(r_iou, nan=0.0))
    fom       = np.mean(np.nan_to_num(r_fom, nan=0.0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fom": fom,
    }, {"accuracy": r_accuracy, "fom": r_fom}


def show_preds(test_paths, preds, arr, n=20, show_edge=False):
    """Show random predictions on test set"""

    for i in range(n):
        ri = np.random.randint(0, len(test_paths))
        test = np.load(test_paths[ri])
        target = test[:, :, -2]
        rgb = utils.rgb_from_stack(test)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(rgb)
        ax[0].set_title("RGB {}".format(ri))
        ax[1].imshow(target)
        ax[1].set_title("Mask")
        ax[2].imshow(preds[ri])
        ax[2].set_title("Accuracy: {:.2f}%".format(arr["accuracy"][ri] * 100))

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        plt.show()

        if show_edge:
            target_edge = utils.edge_from_mask(target)
            pred_edge = utils.edge_from_mask(preds[ri])

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(rgb)
            ax[0].set_title("RGB {}".format(ri))
            ax[1].imshow(target_edge)
            ax[1].set_title("Mask")
            ax[2].imshow(pred_edge)
            ax[2].set_title("FOM: {:.4f}".format(arr["fom"][ri]))

            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            plt.show()


def display_metrics(metrics, arr, hist=True):
    """Display metrics for test set"""

    for key in metrics.keys():
        m = np.round(metrics[key] * 100, 4)
        print("{}: {}".format(key, m))

    if hist:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))

        ax[0].hist(arr["accuracy"])
        ax[0].set_xlabel("Accuracy", size=15)
        ax[0].set_ylabel("Frequency", size=15)

        ax[1].hist(arr["fom"])
        ax[1].set_xlabel("FOM", size=15)
        ax[1].set_ylabel("Frequency", size=15)


def select_pixels_near_edge(edge, distance_threshold=10):
    """Select pixels near the coastline edge"""
    # Find the distance from each pixel to the nearest edge pixel

    distance_map = distance_transform_edt(1 - edge)

    # Select pixels that are less than the specified distance_threshold away from an edge
    selected_pixels = np.where(distance_map <= distance_threshold, True, np.nan)

    return selected_pixels


def get_coastline_pixels(targets, preds):
    """Return only pixels near the coastline"""
    new_targets = targets.copy()
    new_targets = np.array(new_targets)
    new_targets = new_targets.astype(np.float32)

    new_preds = preds.copy()
    new_preds = np.array(new_preds)
    new_preds = new_preds.astype(np.float32)

    for i in range(len(targets)):
        edge = utils.edge_from_mask(targets[i])
        selected_pixels = select_pixels_near_edge(edge, distance_threshold=10)

        new_targets[i] = new_targets[i] * selected_pixels
        new_preds[i] = new_preds[i] * selected_pixels

    return new_targets, new_preds


def plot_importance(importance, channels_, label=None, save=None):
    """Plot feature importance"""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.set_facecolor("white")

    plt.bar(
        height=importance,
        x=np.arange(0, len(channels_)),
    )

    plt.ylabel(label, size=20)

    ax.set_xticks(np.arange(0, len(channels_)), channels_, rotation=90, size=15)
    ax.set_yscale("log")

    plt.yticks(size=15)
