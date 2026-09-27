#!/usr/bin/env bash
# macOS launcher: double-click in Finder (first time: right-click -> Open).
# Uses the Apple GPU (Metal/MPS via PyTorch, installed automatically) or the CPU.
cd "$(dirname "$0")"
exec bash ./start_linux.sh "$@"
