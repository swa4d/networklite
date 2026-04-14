#!/usr/bin/env python3
"""
NetworkLite v2  –  Entry point.

Usage:
    python run.py              # start server on port 5050
    python run.py --port 8080
    python run.py --debug
"""

import sys
import os
import argparse

# Ensure packages are importable
_root = os.path.dirname(os.path.abspath(__file__))
for _p in [_root, "/mnt/project"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def main():
    parser = argparse.ArgumentParser(description="NetworkLite v2 Server")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from dashboard.app import app, run_server
    print(f"\n  NetworkLite v2")
    print(f"  ─────────────────────────────────")
    print(f"  http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print(f"  Debug: {args.debug}")
    print()
    run_server(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
