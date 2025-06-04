import os
import json
from PIL import Image
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
