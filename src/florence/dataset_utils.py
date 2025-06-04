import os
import json
import random
from collections import defaultdict
from PIL import Image
try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - fallback when torch isn't available
    class Dataset:
        def __init__(self):
            pass


def load_local_dataset(folder_name, task_type="DocVQA"):
    """Load a dataset stored as ``JSON``/``PNG`` pairs.

    Parameters
    ----------
    folder_name : str
        Directory containing ``*.json`` annotation files.
    task_type : str, optional
        Task type, one of ``DocVQA``, ``ObjectDetection``, ``Classification`` or
        ``Segmentation``.

    Returns
    -------
    list of dict
        Each entry contains the image and task specific fields.
    """
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
        elif task_type == "Classification":
            label = entry.get('label')
            data.append({'image': image, 'label': label})
        elif task_type == "Segmentation":
            mask_path = os.path.join(folder_name, f"{base_name}_mask.png")
            mask = Image.open(mask_path) if os.path.exists(mask_path) else None
            data.append({'image': image, 'mask': mask})
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


class ClassificationDataset(Dataset):
    """Simple image classification dataset."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example['image']
        if image and image.mode != "RGB":
            image = image.convert("RGB")
        label = example.get('label')
        return image, label


class SegmentationDataset(Dataset):
    """Image segmentation dataset returning an image and mask."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example['image']
        mask = example.get('mask')
        if image and image.mode != "RGB":
            image = image.convert("RGB")
        return image, mask


def kfold_split(data, k, shuffle=True, seed=0):
    """Split a dataset into ``k`` folds for cross validation.

    Parameters
    ----------
    data : Sequence
        Iterable dataset to split.
    k : int
        Number of folds to create.
    shuffle : bool, optional
        Whether to shuffle the data before splitting, by default ``True``.
    seed : int, optional
        Random seed used when ``shuffle`` is ``True``.

    Returns
    -------
    list of tuples
        A list containing ``(train, val)`` pairs for each fold.
    """

    if k <= 1:
        raise ValueError("k must be greater than 1")

    indices = list(range(len(data)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(indices)

    fold_size = len(data) // k
    remainder = len(data) % k
    folds = []
    start = 0
    for i in range(k):
        extra = 1 if i < remainder else 0
        end = start + fold_size + extra
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        train = [data[j] for j in train_idx]
        val = [data[j] for j in val_idx]
        folds.append((train, val))
        start = end
    return folds


def stratified_split(data, labels, test_ratio=0.2, seed=0):
    """Perform a stratified train/test split.

    Parameters
    ----------
    data : Sequence
        Dataset items.
    labels : Sequence
        Label corresponding to each item in ``data``.
    test_ratio : float, optional
        Fraction of each class to include in the test set, by default ``0.2``.
    seed : int, optional
        Random seed for shuffling.

    Returns
    -------
    tuple of lists
        ``(train_data, test_data)`` maintaining label proportions.
    """

    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    rng = random.Random(seed)
    test_indices = []
    for inds in label_to_indices.values():
        rng.shuffle(inds)
        n_test = int(len(inds) * test_ratio)
        test_indices.extend(inds[:n_test])

    test_set = [data[i] for i in test_indices]
    train_set = [data[i] for i in range(len(data)) if i not in set(test_indices)]
    return train_set, test_set
