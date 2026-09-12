"""Command-line entry point used by both the local fallback executor and,
were it ever handed to a real scheduler, the sbatch job script itself.
`python -m campaign_manager.cli run-case ...` runs exactly one case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import campaign, storage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="campaign_manager")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run-case")
    run_p.add_argument("--case-name", required=True)
    run_p.add_argument("--input-deck", required=True, help="path to a JSON input deck file")
    run_p.add_argument("--run-id", required=False, default=None)
    run_p.add_argument("--db-path", required=True)
    run_p.add_argument("--hdf5-dir", required=True)
    run_p.add_argument("--repo-path", required=False, default=None)
    run_p.add_argument("--checkpoint-at", required=False, type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == "run-case":
        deck = json.loads(Path(args.input_deck).read_text())
        with storage.connect(Path(args.db_path)) as conn:
            decision = campaign.run_case(
                conn,
                case_name=args.case_name,
                deck=deck,
                hdf5_dir=Path(args.hdf5_dir),
                repo_path=Path(args.repo_path) if args.repo_path else None,
                checkpoint_at=args.checkpoint_at,
                run_id=args.run_id,
            )
        print(json.dumps({"accepted": decision.accepted, "reason": decision.reason}))
        return 0 if decision.accepted else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
