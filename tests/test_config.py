from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from florence.config import load_config


def test_load_yaml(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("batch_size: 4\nlr: 0.01")
    cfg = load_config(str(cfg_path))
    assert cfg["batch_size"] == 4
    assert cfg["lr"] == 0.01
