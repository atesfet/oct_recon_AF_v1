#!/usr/bin/env bash
# macOS launcher: double-click in Finder (first time: right-click -> Open).
# Runs in CPU mode (CUDA is not available on macOS).
cd "$(dirname "$0")"
exec bash ./start_linux.sh "$@"
