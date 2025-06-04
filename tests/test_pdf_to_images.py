from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pdf_to_images import convert_pdf_to_images

def test_convert_pdf_to_images(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    Image.new("RGB", (10, 10), color="white").save(pdf_path, "PDF")
    out_dir = tmp_path / "images"
    result = convert_pdf_to_images(str(pdf_path), str(out_dir), dpi=50)
    assert len(result) == 1
    assert Path(result[0]).exists()
