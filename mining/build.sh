#!/usr/bin/env bash
# Single-file program, no external deps beyond the standard library: a
# one-line g++ invocation is the whole build.
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -Wall -Wextra -o mine_frames mine_frames.cpp
echo "built mining/mine_frames"
