# Thought Branches

This repo is now a small workflow for generating base output distributions under a few prompt conditions, then chunking and labeling those outputs for thought-branch analysis.

The current source conditions are:

- `best_effort`: the base user prompt plus a "give your absolute best effort" instruction.
- `high_utility`: a utility-contingent prompt using the high-utility side of selected pairs from `utility_behavior_gap`.
- `low_utility`: the matched low-utility side of those pairs.
- `role_play`: the base task with a stronger role/system prompt, such as "world-class essayist."

The old blackmail, whistleblower, faithfulness, and resume-analysis experiments have been removed.

## Setup

```bash
python3 -m pip install -e .
```

For live generations, put this in `.env` or your shell:

```bash
OPENROUTER_API_KEY=...
```

## Workflow

Prepare generation jobs:

```bash
tb-prepare-base-distributions \
  --actors gpt-5.4-mini-or \
  --tasks essay,translation \
  --items-per-task 2 \
  --samples-per-condition 3
```

Run the jobs. Use `--dry-run` first to check the pipeline without API calls:

```bash
tb-run-generation --dry-run --limit 8
tb-run-generation --temperature 0.7 --max-tokens 900
```

For models that expose readable reasoning traces through OpenRouter, request them:

```bash
tb-run-generation --temperature 0.7 --max-tokens 1200 --reasoning-effort low
```

Generation rows store both:

- `raw_response`: the full provider response, including any `reasoning` / `reasoning_details`.
- `output_text`: the extracted final answer content, such as the completed essay.
- `reasoning_text`: readable reasoning text extracted from `raw_response` when available.

Chunk generated final outputs:

```bash
tb-chunk-outputs --source output
```

Chunk readable reasoning traces:

```bash
tb-chunk-outputs --source reasoning --out outputs/chunks/reasoning_chunks.jsonl
```

Label chunks with the built-in heuristic labels:

```bash
tb-label-chunks
```

If your provider returns readable reasoning traces in `raw_response`, you can prepare motivated/unmotivated branch continuation jobs:

```bash
tb-prepare-branch-continuations --max-seeds 20
tb-run-generation --jobs outputs/api/branch_continuation_jobs.jsonl
```

## Inputs And Outputs

Inputs live in `data/inputs/`:

- `task_items.csv`: task prompts copied from `utility_behavior_gap`.
- `selected_utility_pairs.csv`: selected high/low utility pairs copied from `utility_behavior_gap`.

Outputs are written under `outputs/`:

- `outputs/api/generation_jobs.jsonl`
- `outputs/api/generations.jsonl`
- `outputs/chunks/chunks.jsonl`
- `outputs/chunks/chunks_labeled.jsonl`

## Notes

This is intentionally minimal. The goal is to keep the repo centered on branching: generate condition distributions, chunk outputs, label chunks, and prepare branch follow-ups. Add your own conditions by extending `src/thought_branches/prompts.py` and `src/thought_branches/jobs.py`.
