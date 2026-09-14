"""Versioned, deterministic features of the request text alone."""

import math
import re
import string

import pandas as pd

FEATURE_VERSION = "prompt-text-v1"
FEATURE_COLUMNS = [
    "character_length", "word_count", "token_estimate", "line_count",
    "code_fence_count", "contains_code", "contains_math", "answer_choice_count",
    "digit_count", "question_mark_count", "punctuation_count", "comma_count",
    "period_count", "colon_count", "semicolon_count", "bracket_count",
]


def extract_prompt_features(prompt):
    if not isinstance(prompt, str):
        raise ValueError("Feature input must be prompt text")
    return dict(zip(FEATURE_COLUMNS, [
        len(prompt), len(re.findall(r"\b\w+\b", prompt)), math.ceil(len(prompt) / 4),
        prompt.count("\n") + 1 if prompt else 0,
        prompt.count("```"),
        int(bool(re.search(r"```|\b(?:def |class |import |SELECT |function\s*\(|#include)", prompt))),
        int(bool(re.search(r"\\(?:frac|sqrt|sum|int|boxed)|\$|[=^∑∫√]|\d\s*[+*/<>]\s*\d", prompt))),
        len(re.findall(r"(?m)^\s*(?:[A-J][.)]|\([A-J]\))\s+", prompt)),
        sum(c.isdigit() for c in prompt), prompt.count("?"),
        sum(c in string.punctuation for c in prompt),
        prompt.count(","), prompt.count("."), prompt.count(":"), prompt.count(";"),
        sum(c in "()[]{}" for c in prompt),
    ]))


def feature_table(prompts):
    # Explicit input projection: metadata, grades, costs, and outputs cannot enter.
    if prompts.prompt_id.duplicated().any():
        raise ValueError("Duplicate prompt IDs in feature input")
    rows = [dict(prompt_id=pid, **extract_prompt_features(prompt))
            for pid, prompt in prompts[["prompt_id", "prompt"]].itertuples(index=False, name=None)]
    return pd.DataFrame(rows, columns=["prompt_id"] + FEATURE_COLUMNS).sort_values("prompt_id").reset_index(drop=True)
