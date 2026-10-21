"""Reproduce every result from the raw data: verify -> clean -> dictionary -> analyse -> figures.
Usage: python run_all.py"""
import subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
for step in ("verify_raw.py", "clean.py", "data_dictionary.py", "analysis.py", "figures.py"):
    print(f"\n=== {step} ===", flush=True)
    r = subprocess.run([sys.executable, step], cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"{step} failed (exit {r.returncode})")
print("\nAll steps completed. Results in results/, cleaned data in data/processed/.")
