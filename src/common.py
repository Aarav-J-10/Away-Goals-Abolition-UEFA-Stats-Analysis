"""Shared paths and helpers for the away-goals abolition analysis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "UEFA_Tie_Data_RAW.xlsx"
EXT = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
TAB = RES / "tables"
FIG = RES / "figures"
DOCS = ROOT / "docs"
CLEAN_CSV = PROC / "uefa_two_legged_ties_clean.csv"

SEED = 20260621  # fixed seed for every resampling procedure

for d in (PROC, TAB, FIG, DOCS):
    d.mkdir(parents=True, exist_ok=True)
