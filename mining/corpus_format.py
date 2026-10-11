"""Shared binary layout for the mining corpus and query files.

Both generate_corpus.py (writer) and reference_miner.py (reader, for the
independent oracle) import this module so the two never silently drift
apart on field order or dtype. mine_frames.cpp reads the identical layout
by hand (documented in a comment at the top of that file) since it has no
Python import to share; test_mining_reference_match.py is what catches the
two ever disagreeing.
"""
from __future__ import annotations

import struct

import numpy as np

FEATURE_DIM = 8

CORPUS_MAGIC = 0x454E494D  # 'MINE' as little-endian uint32
CORPUS_HEADER = struct.Struct("<IIII")  # magic, num_frames, feature_dim, num_segments

FRAME_DTYPE = np.dtype(
    [("frame_id", "<u8"), ("segment_id", "<u4"), ("feature", "<f4", (FEATURE_DIM,))]
)

QUERY_MAGIC = 0x59524551  # 'QREY' as little-endian uint32
QUERY_HEADER = struct.Struct("<III")  # magic, num_queries, feature_dim

QUERY_DTYPE = np.dtype(
    [("query_id", "<u4"), ("feature", "<f4", (FEATURE_DIM,)), ("radius", "<f4")]
)


def write_corpus(path, frames: np.ndarray, num_segments: int) -> None:
    assert frames.dtype == FRAME_DTYPE
    with open(path, "wb") as f:
        f.write(CORPUS_HEADER.pack(CORPUS_MAGIC, len(frames), FEATURE_DIM, num_segments))
        frames.tofile(f)


def read_corpus(path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num_frames, feature_dim, _num_segments = CORPUS_HEADER.unpack(
            f.read(CORPUS_HEADER.size)
        )
        if magic != CORPUS_MAGIC or feature_dim != FEATURE_DIM:
            raise ValueError(f"bad corpus header in {path}")
        return np.fromfile(f, dtype=FRAME_DTYPE, count=num_frames)


def write_queries(path, queries: np.ndarray) -> None:
    assert queries.dtype == QUERY_DTYPE
    with open(path, "wb") as f:
        f.write(QUERY_HEADER.pack(QUERY_MAGIC, len(queries), FEATURE_DIM))
        queries.tofile(f)


def read_queries(path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num_queries, feature_dim = QUERY_HEADER.unpack(f.read(QUERY_HEADER.size))
        if magic != QUERY_MAGIC or feature_dim != FEATURE_DIM:
            raise ValueError(f"bad query header in {path}")
        return np.fromfile(f, dtype=QUERY_DTYPE, count=num_queries)
