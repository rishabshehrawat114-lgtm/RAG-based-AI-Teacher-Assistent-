"""
scripts/run_app.py
==================
Cross-platform production launcher script for RAG AI Teacher Assistant.

Runs system health checks and starts the Streamlit web application.
Works across Windows, macOS, Linux, and Docker containers without hardcoded paths.

Usage:
    python scripts/run_app.py
    python scripts/run_app.py --port 8501 --address 0.0.0.0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.health import check_system_health


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RAG AI Teacher Assistant Streamlit Application."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit server on (default: 8501).",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0",
        help="Network address to bind server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip startup health checks and launch immediately.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run Streamlit in headless server mode (default: True).",
    )

    args, extra_args = parser.parse_known_args()

    print("=" * 65)
    print(" RAG AI Teacher Assistant - Application Launcher")
    print("=" * 65)

    if not args.skip_health_check:
        print("[1/2] Executing startup health checks...")
        report = check_system_health()
        print(f"      Status: [{report.status}] - {report.summary}")
        if not report.is_healthy:
            print("[ERROR] Critical health check failure. Resolve errors above before launching.")
            return 1
    else:
        print("[1/2] Skipping startup health check (--skip-health-check specified)")

    print("[2/2] Launching Streamlit web server...")
    app_path = PROJECT_ROOT / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={args.port}",
        f"--server.address={args.address}",
    ]
    if args.headless:
        cmd.append("--server.headless=true")
    cmd.extend(extra_args)

    print(f"      Command: {' '.join(cmd)}")
    print(f"      Local URL: http://localhost:{args.port}")
    print("=" * 65)

    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[Application stopped by user]")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Failed to start Streamlit: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
