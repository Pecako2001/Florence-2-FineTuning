"""Example script showing segmentation IoU calculation."""
import numpy as np
from PIL import Image
from florence.metrics import segmentation_iou


if __name__ == "__main__":
    # create simple binary masks
    pred = Image.new("1", (4, 4), 0)
    gt = Image.new("1", (4, 4), 0)
    for x in range(2):
        for y in range(2):
            pred.putpixel((x, y), 1)
    for x in range(3):
        gt.putpixel((x, 0), 1)
    iou = segmentation_iou(np.array(pred), np.array(gt))
    print(f"IoU: {iou:.2f}")
