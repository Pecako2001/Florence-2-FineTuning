"""Example Florence-2 segmentation script."""

import argparse
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation, AutoProcessor
from florence.metrics import iou_score
from florence.dataset_utils import load_local_dataset, SegmentationDataset


def run(model_name: str, dataset_folder: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    data = load_local_dataset(dataset_folder, task_type="Segmentation")
    dataset = SegmentationDataset(data)

    ious = []
    for image, mask in dataset:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_mask = outputs.logits.argmax(dim=1)[0].cpu().numpy()
        true_mask = np.array(mask)
        ious.append(iou_score(pred_mask, true_mask, num_classes=outputs.logits.shape[1]))

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"Mean IoU: {mean_iou:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Florence-2 segmentation example")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Folder with JSON/PNG and mask files")
    parser.add_argument("--model_name", type=str, default="microsoft/Florence-2-base-ft", help="Model name")
    args = parser.parse_args()
    run(args.model_name, args.dataset_folder)


if __name__ == "__main__":
    main()
