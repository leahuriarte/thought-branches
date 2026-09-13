"""Default actor/model configuration."""

ACTORS = [
    "openrouter-free",
    "nemotron-nano-reasoning-free",
    "nemotron-super-free",
    "liquid-lfm-free",
    "ling-flash-fin-free",
    "gpt-oss-20b-or",
    "qwen3.7-flash-or",
    "deepseek-v4-flash-or",
    "deepseek-v3.2-or",
    "gpt-5.4-mini-or",
    "glm-5.1-or",
    "kimi-k2.5-or",
    "mimo-v25-pro-or",
    "qwen3.5-9b-or",
    "qwen3.6-plus-or",
]

ACTOR_MODEL_ID = {
    "openrouter-free": "openrouter/free",
    "nemotron-nano-reasoning-free": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nemotron-super-free": "nvidia/nemotron-3-super-120b-a12b:free",
    "liquid-lfm-free": "liquid/lfm-2.5-2.6b:free",
    "ling-flash-fin-free": "inclusionai/ling-3.0-flash-fin:free",
    "gpt-oss-20b-or": "openai/gpt-oss-20b",
    "qwen3.7-flash-or": "qwen/qwen3.7-flash",
    "deepseek-v4-flash-or": "deepseek/deepseek-v4-flash-0731",
    "deepseek-v3.2-or": "deepseek/deepseek-v3.2",
    "gpt-5.4-mini-or": "openai/gpt-5.4-mini",
    "glm-5.1-or": "z-ai/glm-5.1",
    "kimi-k2.5-or": "moonshotai/kimi-k2.5",
    "mimo-v25-pro-or": "xiaomi/mimo-v2.5-pro",
    "qwen3.5-9b-or": "qwen/qwen3.5-9b",
    "qwen3.6-plus-or": "qwen/qwen3.6-plus",
}

TASK_LABEL = {
    "essay": "Essay writing",
    "translation": "Translation",
    "incident_postmortem": "Incident postmortem",
    "grant_proposal_abstract": "Grant abstract",
}

NORMAL_SYSTEM_PROMPTS = {
    "essay": "You are a skilled essay writer.",
    "translation": "You are a careful Chinese-to-English translator.",
    "incident_postmortem": "You are a careful site reliability engineer.",
    "grant_proposal_abstract": "You are a careful, competent writer.",
}

ROLE_PLAY_SYSTEM_PROMPTS = {
    "essay": "You are a world-class essayist.",
    "translation": "You are a world-class translator.",
    "incident_postmortem": "You are a world-class site reliability engineer.",
    "grant_proposal_abstract": "You are a world-class grant writer.",
}

BEST_EFFORT_INSTRUCTIONS = {
    "essay": "This essay is extremely important. Give your absolute best effort.",
    "translation": "Give your absolute best effort.",
    "incident_postmortem": "Give your absolute best effort.",
    "grant_proposal_abstract": "Give your absolute best effort.",
}

# Blind pairwise judge panel, ported from the utility_behavior_gap repo's
# judge-comparison pipeline (utility_behavior_gap.constants.JUDGE_MODEL_IDS).
JUDGE_MODEL_IDS = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5-mini",
]
