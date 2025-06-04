import numpy as np


def classification_accuracy(preds, labels):
    """Compute accuracy for classification.

    Parameters
    ----------
    preds : Sequence
        Predicted labels.
    labels : Sequence
        Ground truth labels.

    Returns
    -------
    float
        The fraction of correct predictions.
    """
    if len(preds) != len(labels):
        raise ValueError("preds and labels must have the same length")
    if len(labels) == 0:
        return 0.0
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels)


def segmentation_iou(pred_mask, target_mask):
    """Compute the Intersection over Union (IoU) metric for segmentation masks.

    Parameters
    ----------
    pred_mask : array-like
        Predicted binary mask.
    target_mask : array-like
        Ground truth binary mask.

    Returns
    -------
    float
        IoU score between 0 and 1.
    """
    pred = np.array(pred_mask, dtype=bool)
    target = np.array(target_mask, dtype=bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection) / float(union) if union > 0 else 0.0
