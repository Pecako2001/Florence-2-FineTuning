from importlib import import_module
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_import_train():
    pytest.importorskip('torch')
    mod = import_module('scripts.train')
    assert hasattr(mod, 'main')
