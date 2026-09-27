#!/usr/bin/env python3
"""Start the OCT reconstruction web app and open it in the default browser.

    python launch.py [--port 8765] [--no-browser]

The platform launchers (start_linux.sh, start_mac.command, start_windows.bat)
create/activate the Python environment and then run this file.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    from octrecon.webapp.server import serve
    serve(a.host, a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
