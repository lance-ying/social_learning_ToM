#!/usr/bin/env python3
"""Run all 4-panel correlation plot scripts in one command."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "correlation_exp1_4panel.py",
    "correlation_exp2_4panel.py",
    "correlation_exp3_4panel.py",
    "correlation_exp4_4panel.py",
]


def run_script(python_bin: str, script_path: Path, headless: bool) -> int:
    env = os.environ.copy()
    if headless:
        env["MPLBACKEND"] = "Agg"

    cmd = [python_bin, str(script_path)]
    print(f"\n==> Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all correlation_exp*_4panel scripts."
    )
    parser.add_argument(
        "--python",
        default=sys.executable or "python3",
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive matplotlib backend (default is headless Agg).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run all scripts even if one fails.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    scripts = [script_dir / name for name in SCRIPT_ORDER]

    print("=" * 72, flush=True)
    print("RUNNING ALL 4-PANEL CORRELATION SCRIPTS", flush=True)
    print("=" * 72, flush=True)
    print(f"Python: {args.python}", flush=True)
    print(f"Headless mode: {not args.interactive}", flush=True)

    failures = []
    for script_path in scripts:
        exit_code = run_script(
            python_bin=args.python,
            script_path=script_path,
            headless=not args.interactive,
        )
        if exit_code != 0:
            failures.append((script_path.name, exit_code))
            print(f"FAILED: {script_path.name} (exit code {exit_code})", flush=True)
            if not args.continue_on_error:
                break
        else:
            print(f"OK: {script_path.name}", flush=True)

    print("\n" + "=" * 72, flush=True)
    if failures:
        print("Completed with failures:", flush=True)
        for name, code in failures:
            print(f"  - {name}: exit code {code}", flush=True)
        return 1

    print("All 4-panel scripts completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
