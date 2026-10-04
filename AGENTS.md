# Repository Guidelines

## Project Overview

Thought Branches is a small Python workflow for generating prompt-condition output distributions, chunking generated text or readable reasoning traces, labeling chunks, preparing branch-continuation jobs, and running pairwise judging.

Keep changes centered on that workflow. The main code lives in `src/thought_branches/`, CLI entry points are registered in `pyproject.toml`, tests live in `tests/`, fixed inputs live in `data/inputs/`, and generated artifacts are written under `outputs/`.

## Setup

Install the package in editable mode before running commands:

```bash
python3 -m pip install -e .
```

Run tests with:

```bash
python3 -m pytest
```

The project currently has no declared runtime dependencies. Prefer the standard library unless a new dependency is clearly worth adding.

## Live API Calls

Live generation uses OpenRouter through `src/thought_branches/openrouter.py`.

- Put `OPENROUTER_API_KEY=...` in `.env` or the shell before live calls.
- Use `tb-run-generation --dry-run` for pipeline checks that should not call the API.
- Do not commit `.env`, API keys, or provider responses that contain sensitive data.
- Be careful with commands that can spend money or consume credits, especially `tb-run-generation` without `--dry-run` and `tb-label-reasoning-llm`.

## Core Commands

Prepare base-distribution jobs:

```bash
tb-prepare-base-distributions \
  --actors gpt-5.4-mini-or \
  --tasks essay,translation \
  --items-per-task 2 \
  --samples-per-condition 3
```

Run generation safely first:

```bash
tb-run-generation --dry-run --limit 8
```

Run live generation when explicitly intended:

```bash
tb-run-generation --temperature 0.7 --max-tokens 900
```

Request readable reasoning traces when the selected provider supports them:

```bash
tb-run-generation --temperature 0.7 --max-tokens 1200 --reasoning-effort low
```

Chunk final outputs:

```bash
tb-chunk-outputs --source output
```

Chunk readable reasoning traces:

```bash
tb-chunk-outputs --source reasoning --out outputs/chunks/reasoning_chunks.jsonl
```

Apply heuristic labels:

```bash
tb-label-chunks
```

Prepare and run branch-continuation jobs:

```bash
tb-prepare-branch-continuations --max-seeds 20
tb-run-generation --jobs outputs/api/branch_continuation_jobs.jsonl
```

Build the viewer, when needed:

```bash
tb-build-viewer
```

Run pairwise judging, when needed:

```bash
tb-run-judging
```

## Code Conventions

- Keep the package importable from `src/`; tests rely on `pythonpath = ["src"]` in `pyproject.toml`.
- Prefer small, pure helper functions in modules such as `jobs.py`, `prompts.py`, `chunking.py`, `labeling.py`, and `branching.py`.
- Keep CLI scripts thin: parse arguments in `src/thought_branches/scripts/`, then call reusable package functions.
- Use `pathlib.Path` and the shared constants in `src/thought_branches/paths.py` for repository paths.
- Preserve deterministic identifiers where possible. Job IDs use stable prompt digests, and tests may rely on predictable behavior.
- Avoid broad refactors unless the user asks for them; this repo is intentionally minimal.

## Data And Outputs

Inputs under `data/inputs/` are source fixtures for the workflow:

- `task_items.csv`
- `selected_utility_pairs.csv`

Outputs are generated and may be large:

- `outputs/api/generation_jobs.jsonl`
- `outputs/api/generations.jsonl`
- `outputs/api/generation_failures.jsonl`
- `outputs/chunks/chunks.jsonl`
- `outputs/chunks/chunks_labeled.jsonl`

When adding tests, use tiny in-memory rows or temporary files rather than relying on large generated outputs.

## Testing Expectations

For ordinary code changes, run:

```bash
python3 -m pytest
```

For pipeline or CLI changes, also run a small dry-run command relevant to the change, for example:

```bash
tb-prepare-base-distributions --actors gpt-5.4-mini-or --tasks essay --items-per-task 1 --samples-per-condition 1
tb-run-generation --dry-run --limit 1
```

If a command writes to `outputs/`, mention that in the final summary.

## Adding Conditions Or Actors

To add a new prompt condition, update `BASE_CONDITIONS` and job construction in `src/thought_branches/jobs.py`, then add the prompt behavior in `src/thought_branches/prompts.py`.

To add or rename actors/models, update `src/thought_branches/constants.py` and any tests or docs that reference the actor key.

## Verified Utility Data And Model Selection

Local artifact audit (2026-09-19): six configured actors have their own fitted utility data and selected pairs available in this repo. Do not infer utility coverage from membership in `ACTORS` alone.

| Actor key | Configured OpenRouter model ID | Fitted options in source repo | Selected pairs here |
| --- | --- | --- | --- |
| `deepseek-v3.2-or` | `deepseek/deepseek-v3.2` | 610 | 560 |
| `gpt-5.4-mini-or` | `openai/gpt-5.4-mini` | 610 | 560 |
| `glm-5.1-or` | `z-ai/glm-5.1` | 610 | 560 |
| `kimi-k2.5-or` | `moonshotai/kimi-k2.5` | 610 | 560 |
| `qwen3.5-9b-or` | `qwen/qwen3.5-9b` | 610 | 560 |
| `qwen3.6-plus-or` | `qwen/qwen3.6-plus` | 610 | 560 |

Evidence and scope:

- This repo's `data/inputs/selected_utility_pairs.csv` contains 3,360 rows: 320 `default` and 240 `same_count` pairs per actor. `jobs.py` currently uses only `default` pairs.
- The source is the sibling repo `../utility_behavior_gap` (on this machine: `/Users/leah/Repos/utility_behavior_gap`). Its `data/inputs/utility_options.csv` stores fitted means, variances, and train/holdout accuracy and log-loss fields. Each actor has 168 religion, 156 animal, 150 country, and 136 political options.
- Every selected pair was checked against source options keyed by `(actor, domain, option_id)`: descriptions match, means/variances agree within numerical tolerance, and `delta_u` equals high minus low. All 3,360 pairs passed. All selected rows also match the source `outputs/inputs/selected_pairs.csv`.
- All six model IDs agree between the two repos' `constants.py` files. This verifies saved artifact/configuration consistency; the utility CSV does not record exact serving snapshots, inference settings, or raw elicitation requests, so this is not proof of current-provider equivalence or a fresh utility measurement.
- The source loader supports `outputs/inputs/utility_options__*.csv` overlays; none were present at audit time. Recheck overlays and source files when updating this inventory.
- Audited SHA-256: selected pairs `b512bd21af7351cf32d94892c8c8cd4d1cfe56c78be6524894a36c422c310617`; source utility options `f078c48e4c8f329288aef4d6430d23e2aa2a441af63504a4962778d4f74db9b9`.

Important exclusions:

- Source utilities also exist for `mimo-v2-pro-or` / `xiaomi/mimo-v2-pro` (610 options), but this repo configures `mimo-v25-pro-or` / `xiaomi/mimo-v2.5-pro`. These are different models. No V2.5 utility fit or selected pairs were found in the audited files; do not transfer V2 utilities to V2.5.
- No actor-specific fitted utilities or selected pairs were found in the audited files for `openrouter-free`, `nemotron-nano-reasoning-free`, `nemotron-super-free`, `liquid-lfm-free`, `ling-flash-fin-free`, `gpt-oss-20b-or`, `qwen3.7-flash-or`, or `deepseek-v4-flash-or`.
- No utility fit for `DeepSeek-R1-Distill-Qwen-14B` was found in these inputs. Being a DeepSeek model does not make its utilities interchangeable with V3.2's.

Experimental rules:

- `jobs.py` currently silently falls back to other actors' pairs when an actor has none. A successfully prepared high/low job therefore does NOT prove that actor's utility was measured. For model-specific utility experiments, use an actor in the verified table or obtain its own fit first. The current job's `utility_pair` metadata also omits the source actor, so audit against the input CSV rather than trusting the job label.
- DeepSeek V3.2 has verified saved utility coverage. Qwen3.5 9B also has coverage and should be considered when investigating a smaller self-hosted model, before assuming a new R1 distillation is necessary. Utility coverage alone does not establish native reasoning-prefix capability.
- Before changing model version, provider, reasoning mode, or quantization, check the elicitation configuration where available and validate that the utility ordering still holds; refit when necessary. Generate base traces and branches with a consistent model/backend/configuration.
- Native reasoning continuation remains unvalidated. The existing DeepSeek branch pilot used quoted prefixes in user messages; treat it as a prompting pilot, not evidence of natural branch continuation or causal motivation effects. Validate continuation before scaling another run.

## OpenRouter Completion Audit For Utility-Covered Models

Checked OpenRouter's current model-catalog and endpoint metadata on 2026-09-19 for every actor with verified saved utility data. None advertises a raw `prompt` completion interface or an assistant-prefix/reasoning-prefix field such as `prefix` or `reasoning_content.

| Actor | OpenRouter model | Metadata supports | Native prefix completion documented? |
| --- | --- | --- | --- |
| `deepseek-v3.2-or` | `deepseek/deepseek-v3.2` | reasoning, readable reasoning, temperature/top-p, seed, tools | No |
| `gpt-5.4-mini-or` | `openai/gpt-5.4-mini` | reasoning effort, seed, structured output, tools | No |
| `glm-5.1-or` | `z-ai/glm-5.1` | reasoning, temperature/top-p, seed, logprobs, tools | No |
| `kimi-k2.5-or` | `moonshotai/kimi-k2.5` | reasoning, temperature/top-p, seed, logprobs, tools | No |
| `qwen3.5-9b-or` | `qwen/qwen3.5-9b` | reasoning, temperature/top-p, seed, structured output, tools | No |
| `qwen3.6-plus-or` | `qwen/qwen3.6-plus` | reasoning, temperature/top-p, seed, structured output, tools | No |

These models are usable through OpenRouter's ordinary chat-completions interface. A `max_tokens` or `max_completion_tokens` field only caps the generated answer; it does not make the endpoint a raw completion or token-prefix continuation API. A request that quotes a prefix in a user message remains an off-policy prompt intervention. An assistant-role message with an undocumented provider-specific field should be treated as an experiment requiring validation, not as a supported interface.

For native prefix continuation, use a provider/model that explicitly documents assistant-prefix or reasoning-prefix completion, or run a local checkpoint where the exact token prefix can be passed to the generation loop. Before using any OpenRouter route for this experiment, inspect model metadata and run a small prefix-fidelity test; do not infer completion capability from the model name or from the fact that the endpoint returns a `completion` field in ordinary chat responses.

Sources checked: [OpenRouter model catalog](https://openrouter.ai/api/v1/models), [DeepSeek V3.2 endpoints](https://openrouter.ai/api/v1/models/deepseek/deepseek-v3.2/endpoints), [Qwen3.5-9B endpoints](https://openrouter.ai/api/v1/models/qwen/qwen3.5-9b/endpoints), [Qwen3.6 Plus endpoints](https://openrouter.ai/api/v1/models/qwen/qwen3.6-plus/endpoints), and [OpenRouter chat-completions API](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).

## Existing Worktree Changes

Before editing, check `git status --short`. Do not overwrite unrelated local changes. If a file already has user edits, read it carefully and make only the requested, compatible change.
