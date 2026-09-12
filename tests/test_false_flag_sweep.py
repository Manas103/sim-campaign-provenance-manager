import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from false_flag_sweep import run_false_flag_sweep  # noqa: E402


def test_two_hundred_clean_campaigns_never_false_flag():
    result = run_false_flag_sweep(n_legit_change=100, n_exact_rerun=100)
    assert result["total_campaigns"] == 200
    assert len(result["false_flags"]) == 0, f"false flags: {result['false_flags']}"
