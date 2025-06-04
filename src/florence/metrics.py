"""Utility metrics for model evaluation."""

from __future__ import annotations

import numpy as np


def accuracy(preds, labels):
    """Compute classification accuracy.

    Parameters
    ----------
    preds : sequence of int or str
        Predicted labels.
    labels : sequence of int or str
        Ground truth labels.

    Returns
    -------
    float
        Fraction of correct predictions.
    """
    if len(preds) != len(labels):
        raise ValueError("preds and labels must have the same length")
    correct = sum(p == t for p, t in zip(preds, labels))
    return correct / len(labels) if labels else 0.0


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int) -> float:
    """Compute mean Intersection over Union (IoU) for segmentation masks.

    Parameters
    ----------
    pred_mask : ndarray
        Predicted mask with integer class values.
    true_mask : ndarray
        Ground truth mask with integer class values.
    num_classes : int
        Number of classes present in the masks.

    Returns
    -------
    float
        Mean IoU across classes (ignoring classes with no pixels).
    """
    ious = []
    for cls in range(num_classes):
        pred = pred_mask == cls
        true = true_mask == cls
        intersection = np.logical_and(pred, true).sum()
        union = np.logical_or(pred, true).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0
