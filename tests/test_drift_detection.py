import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from drift_detection_check import run_drift_detection_sweep  # noqa: E402


def test_thirty_seeded_drift_scenarios_are_caught():
    result = run_drift_detection_sweep(n_scenarios=30)
    assert result["total"] == 30
    assert result["caught"] == 30, f"missed scenarios: {result['missed']}"
