import numpy as np
import pytest
from florence.metrics import accuracy, iou_score


def test_accuracy():
    preds = [0, 1, 2]
    labels = [0, 0, 2]
    assert accuracy(preds, labels) == pytest.approx(2/3)

def test_iou_score():
    pred = np.array([[0, 1], [1, 1]])
    true = np.array([[0, 0], [1, 1]])
    iou = iou_score(pred, true, num_classes=2)
    expected = (1/2 + 2/3) / 2
    assert iou == pytest.approx(expected)
