"""Resolves a mined frame back to the log segment that produced it.

Every frame in the corpus carries its segment_id (mining/corpus_format.py);
this module is the one place that turns that integer into the thing a
human or a downstream tool actually wants: which log file, and (for a
frame that came directly from the simulated log rather than being a
seeded near-duplicate stand-in) its offset within that file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Lineage:
    frame_id: int
    segment_id: int
    log_file: str
    offset_in_segment: Optional[int]  # None for a seeded synthetic duplicate


class SegmentIndex:
    def __init__(self, segments_json_path: Path):
        with open(segments_json_path) as f:
            raw = json.load(f)
        self._segments = {int(k): v for k, v in raw.items()}

    def resolve(self, frame_id: int, segment_id: int) -> Lineage:
        seg = self._segments[segment_id]
        start, end = seg["start_frame_id"], seg["end_frame_id"]
        offset = frame_id - start if start <= frame_id <= end else None
        return Lineage(
            frame_id=frame_id,
            segment_id=segment_id,
            log_file=seg["log_file"],
            offset_in_segment=offset,
        )

    def resolve_all(self, frame_segment_pairs) -> list[Lineage]:
        return [self.resolve(fid, sid) for fid, sid in frame_segment_pairs]
