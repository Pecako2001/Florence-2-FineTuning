import numpy as np
from florence.metrics import classification_accuracy, segmentation_iou


def test_classification_accuracy():
    preds = [1, 0, 1, 1]
    labels = [1, 0, 0, 1]
    acc = classification_accuracy(preds, labels)
    assert acc == 0.75


def test_segmentation_iou():
    pred = np.array([[1, 0], [1, 0]])
    target = np.array([[1, 0], [0, 0]])
    iou = segmentation_iou(pred, target)
    assert iou == 0.5
