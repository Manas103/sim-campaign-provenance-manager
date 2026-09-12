import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from campaign_manager import storage  # noqa: E402


@pytest.fixture
def repo_path():
    return REPO_ROOT


@pytest.fixture
def db_conn(tmp_path):
    db_path = tmp_path / "campaign.sqlite"
    with storage.connect(db_path) as conn:
        yield conn


@pytest.fixture
def hdf5_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def base_deck(**overrides):
    deck = {
        "n": 12,
        "iterations": 80,
        "boundary_temp": 1.0,
        "source_amplitude": 5.0,
        "seed": 42,
    }
    deck.update(overrides)
    return deck
