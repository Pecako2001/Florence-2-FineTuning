import json
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from florence.dataset_utils import (
    load_local_dataset,
    ObjectDetectionDataset,
    kfold_split,
    stratified_split,
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
