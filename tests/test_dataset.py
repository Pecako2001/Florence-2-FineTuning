import json
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_utils import load_local_dataset, ObjectDetectionDataset

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
