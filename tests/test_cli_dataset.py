import csv
import json
import subprocess
import sys
from pathlib import Path
from PIL import Image

def test_cli_dataset_csv(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10)).save(image_path)

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "question",
                "question_types",
                "docId",
                "ucsf_document_id",
                "ucsf_document_page_no",
                "answers",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": str(image_path),
                "question": "q1",
                "question_types": "type",
                "docId": "1",
                "ucsf_document_id": "doc",
                "ucsf_document_page_no": "1",
                "answers": "a",
            }
        )

    out_dir = tmp_path / "out"
    subprocess.run(
        [sys.executable, "scripts/create_dataset_cli.py", "--metadata_csv", str(csv_path), "--dataset_folder", str(out_dir)],
        check=True,
    )

    json_files = list(out_dir.glob("*.json"))
    png_files = list(out_dir.glob("*.png"))
    assert json_files and png_files
    with open(json_files[0]) as f:
        data = json.load(f)
    assert data["question"] == "q1"
