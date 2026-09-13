"""Repository paths used by the command-line scripts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
OUTPUTS = ROOT / "outputs"
OUTPUT_API = OUTPUTS / "api"

TASK_ITEMS = INPUTS / "task_items.csv"
SELECTED_UTILITY_PAIRS = INPUTS / "selected_utility_pairs.csv"

GENERATION_JOBS = OUTPUT_API / "generation_jobs.jsonl"
GENERATIONS = OUTPUT_API / "generations.jsonl"
GENERATION_FAILURES = OUTPUT_API / "generation_failures.jsonl"
CHUNKS = OUTPUTS / "chunks" / "chunks.jsonl"
LABELED_CHUNKS = OUTPUTS / "chunks" / "chunks_labeled.jsonl"
