"""Deterministic graders, kept independent from provider integrations."""

import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Callable, Mapping, Sequence


class MalformedModelOutput(ValueError):
    pass


CodeTestRunner = Callable[[str, Sequence[Mapping[str, str]]], bool]


def _require_output(output: object) -> str:
    if not isinstance(output, str) or not output.strip():
        raise MalformedModelOutput("Model output must be a non-empty string")
    return output.strip()


def grade_multiple_choice(
    output: object, reference_answer: str, valid_options: Sequence[str]
) -> bool:
    normalized = _require_output(output).upper()
    valid = {option.upper() for option in valid_options}
    if normalized not in valid:
        raise MalformedModelOutput(
            f"Multiple-choice output must be exactly one of {sorted(valid)}"
        )
    return normalized == reference_answer.strip().upper()


def _normalize_math(value: str) -> str:
    value = value.strip().replace("$", "").replace(",", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\s+", "", value)
    return value.rstrip(".")


def grade_math(output: object, reference_answer: str) -> bool:
    candidate = _normalize_math(_require_output(output))
    reference = _normalize_math(reference_answer)
    if not reference:
        raise ValueError("Math reference answer is empty")
    try:
        return Decimal(candidate) == Decimal(reference)
    except InvalidOperation:
        try:
            return Fraction(candidate) == Fraction(reference)
        except (ValueError, ZeroDivisionError):
            return candidate == reference


def extract_code(output: object) -> str:
    code = _require_output(output)
    fenced = re.fullmatch(r"```(?:python)?\s*\n?(.*?)\n?```", code, re.DOTALL | re.IGNORECASE)
    if fenced:
        code = fenced.group(1).strip()
    if not code:
        raise MalformedModelOutput("Code output is empty after removing fences")
    return code


def grade_code(
    output: object,
    tests: Sequence[Mapping[str, str]],
    test_runner: CodeTestRunner,
) -> bool:
    code = extract_code(output)
    if not tests:
        raise ValueError("Code grading requires at least one executable test")
    return test_runner(code, tests)


def run_trusted_dry_run_tests(
    code: str, tests: Sequence[Mapping[str, str]]
) -> bool:
    """Run committed mock code only; never pass live model output to this runner."""
    for test in tests:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                input=test["input"],
                text=True,
                capture_output=True,
                timeout=2,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        if completed.returncode != 0 or completed.stdout.strip() != test["output"].strip():
            return False
    return True

