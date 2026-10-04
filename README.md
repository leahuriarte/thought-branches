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

Before treating a hosted endpoint as a native reasoning-continuation backend, prepare a small
prefix-fidelity check (three prefix locations, including a mid-sentence cut, with three identical
requests per prefix):

```bash
tb-validate-deepseek-prefix
```

This command is offline by default and writes the exact request payloads for inspection. After
setting `DEEPSEEK_API_KEY`, add `--live` to make the nine small paid requests. The validator uses
DeepSeek's documented beta `assistant.prefix` and `reasoning_content` fields, requires manual review
of every prefix/suffix seam, and never marks the current direct-API models as eligible for the V3.2
utility experiment. DeepSeek's current direct models and `deepseek/deepseek-v3.2` are different
experimental subjects.

Protocol details: [DeepSeek Chat Prefix Completion](https://api-docs.deepseek.com/guides/chat_prefix_completion),
[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/), and
[current Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/).

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

The built-in heuristic labels remain available for exploratory summaries, but they are not accepted
for branch selection:

```bash
tb-label-chunks
```

Classify reasoning chunks with the LLM labeler, then prepare native motivated/unmotivated prefixes.
Branch preparation rejects heuristic labels and older LLM label schemas:

```bash
tb-label-reasoning-llm \
  --generations outputs/api/generations.jsonl \
  --chunks outputs/chunks/reasoning_chunks.jsonl \
  --out outputs/chunks/reasoning_chunks_llm_labeled.jsonl

tb-prepare-branch-continuations \
  --generations outputs/api/generations.jsonl \
  --labeled-chunks outputs/chunks/reasoning_chunks_llm_labeled.jsonl \
  --model deepseek-flash \
  --max-seeds 1 \
  --samples-per-branch 2

tb-run-branch-continuations
tb-run-branch-continuations --live
```

The LLM schema gives each chunk one `primary_function` and zero or more `feature_flags`. In
particular, `motivation_cue_repetition` marks a restatement such as "the user asked for my best
effort," while `motivation_commitment` marks a resulting resolve such as "so I should write a better
answer." Both can appear on the same chunk, along with a more specific motivation flag. Native branch
selection requires `motivation_commitment`, so cue repetition alone is not treated as motivation.

The runner is validation-only unless `--live` is supplied. It sends the original prompt followed by
an assistant message containing the selected `reasoning_content` prefix and `prefix: true`; it does
not quote the prefix in a new user prompt. Repeated samples for each source/branch are rejected unless
their API payloads are identical. Generated `reasoning_text` contains only the new suffix so it can be
chunked and LLM-classified afterward. `source_model` and target `model` are recorded separately because
continuing a V3.2 trace with Flash changes the experimental subject.

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
