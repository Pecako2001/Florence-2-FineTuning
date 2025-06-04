"""Example Florence-2 image classification script."""

import argparse
import torch
from transformers import AutoModelForImageClassification, AutoProcessor

from florence.metrics import accuracy
from florence.dataset_utils import load_local_dataset, ClassificationDataset


def run(model_name: str, dataset_folder: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    data = load_local_dataset(dataset_folder, task_type="Classification")
    dataset = ClassificationDataset(data)

    preds, labels = [], []
    for image, label in dataset:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = logits.argmax(dim=-1).item()
        preds.append(pred)
        labels.append(label)

    acc = accuracy(preds, labels)
    print(f"Accuracy: {acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Florence-2 classification example")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Folder with JSON/PNG pairs and labels")
    parser.add_argument("--model_name", type=str, default="microsoft/Florence-2-base-ft", help="Model name")
    args = parser.parse_args()
    run(args.model_name, args.dataset_folder)


if __name__ == "__main__":
    main()
