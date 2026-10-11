"""Tests for the mining/ extension: C++ miner vs. independent Python oracle,
seeded near-duplicate precision/recall, per-frame lineage, and the
manifest-pinned provenance gate reused from campaign_manager.

Uses a small (10,015-frame) corpus, not the full 2,000,000-frame one: the
full-scale run and its measured numbers live in
scripts/mining_full_scale_run.py and docs/mining_benchmark_output.txt.
These tests are about correctness, not scale.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mining import generate_corpus, lineage, pipeline, reference_miner  # noqa: E402

MINER_BINARY = REPO_ROOT / "mining" / "mine_frames"
DEDUP_EPSILON = 1e-2


@pytest.fixture(scope="module")
def small_corpus(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("mining_corpus")
    saved = (
        generate_corpus.NUM_SEGMENTS,
        generate_corpus.FRAMES_PER_SEGMENT,
        generate_corpus.NUM_QUERY_SEGMENTS,
    )
    generate_corpus.NUM_SEGMENTS = 20
    generate_corpus.FRAMES_PER_SEGMENT = 500
    generate_corpus.NUM_QUERY_SEGMENTS = 5
    try:
        generate_corpus.generate(7, out_dir)
    finally:
        (
            generate_corpus.NUM_SEGMENTS,
            generate_corpus.FRAMES_PER_SEGMENT,
            generate_corpus.NUM_QUERY_SEGMENTS,
        ) = saved
    return out_dir


def _read_csv_rows(path: Path):
    with open(path) as f:
        return {(int(r["frame_id"]), int(r["segment_id"])) for r in csv.DictReader(f)}


@pytest.mark.skipif(not MINER_BINARY.exists(), reason="run mining/build.sh first")
def test_cpp_miner_matches_independent_reference_exactly(small_corpus, tmp_path):
    cpp_out = tmp_path / "cpp_out.csv"
    subprocess.run(
        [
            str(MINER_BINARY),
            f"--corpus={small_corpus / 'corpus.bin'}",
            f"--queries={small_corpus / 'queries.bin'}",
            f"--out={cpp_out}",
            f"--dedup-epsilon={DEDUP_EPSILON}",
        ],
        check=True,
        capture_output=True,
    )
    cpp_rows = _read_csv_rows(cpp_out)

    kept_ids, seg_of, pairs, _n = reference_miner.mine(
        small_corpus / "corpus.bin", small_corpus / "queries.bin", DEDUP_EPSILON
    )
    ref_rows = {(fid, seg_of[fid]) for fid in kept_ids}

    assert cpp_rows == ref_rows
    assert len(cpp_rows) > 0


@pytest.mark.skipif(not MINER_BINARY.exists(), reason="run mining/build.sh first")
def test_seeded_near_duplicates_all_suppressed_with_zero_false_suppressions(
    small_corpus, tmp_path
):
    import json

    gt = json.loads((small_corpus / "ground_truth.json").read_text())
    cpp_out = tmp_path / "cpp_out.csv"
    subprocess.run(
        [
            str(MINER_BINARY),
            f"--corpus={small_corpus / 'corpus.bin'}",
            f"--queries={small_corpus / 'queries.bin'}",
            f"--out={cpp_out}",
            f"--dedup-epsilon={DEDUP_EPSILON}",
        ],
        check=True,
        capture_output=True,
    )
    kept_frame_ids = {fid for fid, _seg in _read_csv_rows(cpp_out)}

    duplicate_frame_ids = {dup for _orig, dup in gt["duplicate_pairs"]}
    suppressed = duplicate_frame_ids - kept_frame_ids
    assert suppressed == duplicate_frame_ids, (
        f"expected all {len(duplicate_frame_ids)} seeded duplicates suppressed, "
        f"only {len(suppressed)} were"
    )

    control_frame_ids = set(gt["control_frame_ids"])
    falsely_suppressed = control_frame_ids - kept_frame_ids
    assert falsely_suppressed == set(), (
        f"{len(falsely_suppressed)} of {len(control_frame_ids)} distinct control "
        "frames were incorrectly suppressed"
    )


@pytest.mark.skipif(not MINER_BINARY.exists(), reason="run mining/build.sh first")
def test_every_selected_frame_traces_to_a_log_segment(small_corpus, tmp_path):
    cpp_out = tmp_path / "cpp_out.csv"
    subprocess.run(
        [
            str(MINER_BINARY),
            f"--corpus={small_corpus / 'corpus.bin'}",
            f"--queries={small_corpus / 'queries.bin'}",
            f"--out={cpp_out}",
            f"--dedup-epsilon={DEDUP_EPSILON}",
        ],
        check=True,
        capture_output=True,
    )
    rows = _read_csv_rows(cpp_out)
    index = lineage.SegmentIndex(small_corpus / "segments.json")
    resolved = index.resolve_all(rows)
    assert len(resolved) == len(rows)
    for entry in resolved:
        assert entry.log_file.startswith("logs/segment_")

    # Every base (non-seeded-duplicate) frame must resolve to a concrete,
    # in-range offset; only the appended synthetic duplicates (frame_id at
    # or beyond the base corpus size) are allowed a None offset.
    base_corpus_size = 20 * 500
    for entry in resolved:
        if entry.frame_id < base_corpus_size:
            assert entry.offset_in_segment is not None


@pytest.mark.skipif(not MINER_BINARY.exists(), reason="run mining/build.sh first")
def test_gate_accepts_first_run_and_identical_rerun(small_corpus, db_conn, hdf5_dir, tmp_path):
    decision1, selected1, _ = pipeline.run_mining_case(
        db_conn,
        hdf5_dir,
        MINER_BINARY,
        small_corpus / "corpus.bin",
        small_corpus / "queries.bin",
        DEDUP_EPSILON,
        tmp_path / "scratch",
    )
    assert decision1.accepted
    assert decision1.reason == "first_run_for_case"

    decision2, selected2, _ = pipeline.run_mining_case(
        db_conn,
        hdf5_dir,
        MINER_BINARY,
        small_corpus / "corpus.bin",
        small_corpus / "queries.bin",
        DEDUP_EPSILON,
        tmp_path / "scratch",
    )
    assert decision2.accepted
    assert decision2.reason == "matches_baseline"
    assert np.array_equal(selected1, selected2)


@pytest.mark.skipif(not MINER_BINARY.exists(), reason="run mining/build.sh first")
def test_gate_refuses_unexplained_drift(small_corpus, db_conn, hdf5_dir, tmp_path):
    # A second corpus (different seed) stands in for "the corpus silently
    # changed on disk." Both runs declare the identical forced corpus_hash,
    # so the manifest comparison sees no declared change; only the actual
    # mined content differs, which is exactly the shape evaluate_gate exists
    # to catch (campaign.py's input_hash_override tests the same idea for
    # the toy solver).
    other_corpus = tmp_path / "other_corpus"
    saved = (
        generate_corpus.NUM_SEGMENTS,
        generate_corpus.FRAMES_PER_SEGMENT,
        generate_corpus.NUM_QUERY_SEGMENTS,
    )
    generate_corpus.NUM_SEGMENTS = 20
    generate_corpus.FRAMES_PER_SEGMENT = 500
    generate_corpus.NUM_QUERY_SEGMENTS = 5
    try:
        generate_corpus.generate(99, other_corpus)
    finally:
        (
            generate_corpus.NUM_SEGMENTS,
            generate_corpus.FRAMES_PER_SEGMENT,
            generate_corpus.NUM_QUERY_SEGMENTS,
        ) = saved

    forced_hash = "forced-identical-declared-corpus-hash"
    decision1, _selected1, _ = pipeline.run_mining_case(
        db_conn,
        hdf5_dir,
        MINER_BINARY,
        small_corpus / "corpus.bin",
        small_corpus / "queries.bin",
        DEDUP_EPSILON,
        tmp_path / "scratch",
        corpus_hash_override=forced_hash,
    )
    assert decision1.accepted

    decision2, _selected2, _ = pipeline.run_mining_case(
        db_conn,
        hdf5_dir,
        MINER_BINARY,
        other_corpus / "corpus.bin",
        small_corpus / "queries.bin",
        DEDUP_EPSILON,
        tmp_path / "scratch",
        corpus_hash_override=forced_hash,
    )
    assert not decision2.accepted
    assert decision2.reason == "unexplained_drift"
