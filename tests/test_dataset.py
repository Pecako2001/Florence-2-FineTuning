import json
import sys
from pathlib import Path
import pytest
PIL_Image = pytest.importorskip("PIL.Image")
Image = PIL_Image.Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_utils import (
    load_local_dataset,
    ObjectDetectionDataset,
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

