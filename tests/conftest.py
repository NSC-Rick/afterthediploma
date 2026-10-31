import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import scp  # noqa: E402


@pytest.fixture(autouse=True)
def _no_disk_writes(monkeypatch):
    """App flows persist profiles via scp.save_profile - keep tests off disk."""
    monkeypatch.setattr(scp, "save_profile", lambda *a, **k: None)
