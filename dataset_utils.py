import os
import json
from PIL import Image
import torch
try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - fallback when torch isn't available
    class Dataset:
        def __init__(self):
            pass


def load_local_dataset(folder_name, task_type="DocVQA"):
    data = []
    for file_name in os.listdir(folder_name):
        if not file_name.endswith('.json'):
            continue
        with open(os.path.join(folder_name, file_name), 'r') as f:
            entry = json.load(f)
        base_name = os.path.splitext(file_name)[0]
        image_path = os.path.join(folder_name, f"{base_name}.png")
        image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None
        if task_type == "ObjectDetection":
            objects = entry.get('objects', [])
            boxes = [obj.get('bbox') for obj in objects]
            labels = [obj.get('label') for obj in objects]
            data.append({'image': image, 'boxes': boxes, 'labels': labels})
        else:
            entry['image'] = image
            data.append(entry)
    return data


class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<DocVQA>" + example['question']
        answers = example['answers']
        if answers is None:
            answers = [""]
        elif isinstance(answers, dict):
            answers = list(answers.values())
        elif not isinstance(answers, list):
            answers = [str(answers)]
        first_answer = answers[0] if answers else ""
        image = example['image']
        if image and image.mode != "RGB":
            image = image.convert("RGB")
        return question, first_answer, image


class ObjectDetectionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example['image']
        if image and image.mode != "RGB":
            image = image.convert("RGB")
        return image, {"boxes": example['boxes'], "labels": example['labels']}


def cache_preprocessed_dataset(data, processor, cache_dir, task_type="DocVQA"):
    """Preprocess a dataset and save the features to ``cache_dir``.

    Parameters
    ----------
    data : list
        Loaded dataset entries as returned by :func:`load_local_dataset`.
    processor : object
        Processor with ``tokenizer`` and image processing methods.
    cache_dir : str
        Directory where cached ``.pt`` files will be stored.
    task_type : str, optional
        Either ``"DocVQA"`` or ``"ObjectDetection"``. Defaults to ``"DocVQA"``.

    Returns
    -------
    list
        A list of dictionaries containing preprocessed tensors.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cached_data = []
    for i, example in enumerate(data):
        if task_type == "DocVQA":
            question = "<DocVQA>" + example["question"]
            answers = example.get("answers")
            if answers is None:
                answers = [""]
            elif isinstance(answers, dict):
                answers = list(answers.values())
            elif not isinstance(answers, list):
                answers = [str(answers)]
            first_answer = answers[0] if answers else ""
            image = example["image"]
            if image and image.mode != "RGB":
                image = image.convert("RGB")
            enc = processor(
                text=[question],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=getattr(processor.tokenizer, "model_max_length", None),
            )
            labels = processor.tokenizer(
                text=[first_answer],
                return_tensors="pt",
                padding="max_length",
                max_length=getattr(processor.tokenizer, "model_max_length", None),
                return_token_type_ids=False,
            ).input_ids
            item = {
                "input_ids": enc["input_ids"].squeeze(0),
                "pixel_values": enc["pixel_values"].squeeze(0),
                "labels": labels.squeeze(0),
            }
        else:
            image = example["image"]
            if image and image.mode != "RGB":
                image = image.convert("RGB")
            target = {"boxes": example["boxes"], "labels": example["labels"]}
            enc = processor(images=[image], annotations=[target], return_tensors="pt")
            item = {k: v.squeeze(0) if hasattr(v, "squeeze") else v for k, v in enc.items()}

        file_path = os.path.join(cache_dir, f"{i}.pt")
        torch.save(item, file_path)
        cached_data.append(item)

    return cached_data


def load_cached_dataset(cache_dir):
    """Load preprocessed tensors from ``cache_dir``."""
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    return [torch.load(os.path.join(cache_dir, f)) for f in files]


class CachedDataset(Dataset):
    """Dataset that serves items from cached tensors."""

    def __init__(self, cache_dir):
        self.data = load_cached_dataset(cache_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

