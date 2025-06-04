import json
import sys
from pathlib import Path
import pytest
PIL_Image = pytest.importorskip("PIL.Image")
Image = PIL_Image.Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from florence.dataset_utils import (
    load_local_dataset,
    ObjectDetectionDataset,
    kfold_split,
    stratified_split,
    cache_preprocessed_dataset,
    CachedDataset,
)

def test_object_detection_dataset(tmp_path):
    img_path = tmp_path / "1.png"
    Image.new("RGB", (10, 10), color="white").save(img_path)
    ann = {"objects": [{"bbox": [1, 2, 3, 4], "label": "cat"}]}
    with open(tmp_path / "1.json", "w") as f:
        json.dump(ann, f)
    data = load_local_dataset(str(tmp_path), task_type="ObjectDetection")
    assert len(data) == 1
    dataset = ObjectDetectionDataset(data)
    image, target = dataset[0]
    assert target["boxes"][0] == [1, 2, 3, 4]
    assert target["labels"][0] == "cat"


def test_kfold_split():
    data = list(range(10))
    folds = kfold_split(data, k=5, shuffle=False)
    assert len(folds) == 5
    combined_val = []
    for train, val in folds:
        assert len(train) == 8
        assert len(val) == 2
        combined_val.extend(val)
    assert sorted(combined_val) == data


def test_stratified_split():
    data = list(range(10))
    labels = ["a"] * 5 + ["b"] * 5
    train, test = stratified_split(data, labels, test_ratio=0.2, seed=0)
    assert len(train) == 8
    assert len(test) == 2
    label = lambda x: "a" if x < 5 else "b"
    assert [label(x) for x in train].count("a") == 4
    assert [label(x) for x in train].count("b") == 4
    assert [label(x) for x in test].count("a") == 1
    assert [label(x) for x in test].count("b") == 1
    
class DummyProcessor:
    class tokenizer:
        model_max_length = 4

        def __call__(self, text, return_tensors=None, padding=None, max_length=None, return_token_type_ids=False):
            import torch
            return {"input_ids": torch.tensor([[1, 1, 1, 1]])}

    def __call__(self, text=None, images=None, return_tensors=None, padding=None, max_length=None, annotations=None):
        import torch
        return {"input_ids": torch.tensor([[0, 0, 0, 0]]), "pixel_values": torch.zeros((1, 3, 2, 2))}


def test_cache_preprocessed_dataset(tmp_path):
    img_path = tmp_path / "1.png"
    Image.new("RGB", (10, 10), color="white").save(img_path)
    ann = {"question": "q", "answers": ["a"]}
    with open(tmp_path / "1.json", "w") as f:
        json.dump(ann, f)
    data = load_local_dataset(str(tmp_path))
    cache_dir = tmp_path / "cache"
    processor = DummyProcessor()
    cached = cache_preprocessed_dataset(data, processor, str(cache_dir))
    assert len(cached) == 1
    assert (cache_dir / "0.pt").exists()
    dataset = CachedDataset(str(cache_dir))
    item = dataset[0]
    assert "input_ids" in item and "pixel_values" in item and "labels" in item

