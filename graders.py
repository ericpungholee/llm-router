"""Deterministic graders, kept independent from provider integrations."""

import ast
import re
import subprocess
import sys
from fractions import Fraction
from typing import Callable, Mapping, Sequence


class MalformedModelOutput(ValueError):
    pass


CodeTestRunner = Callable[[str, Sequence[Mapping[str, str]]], bool]


class UnsafeGeneratedCode(ValueError):
    pass


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


def parse_multiple_choice(output: object, valid_options: Sequence[str]) -> str:
    """Extract one unambiguous answer option from common model formats."""
    text = _require_output(output)
    valid = {option.upper() for option in valid_options}
    direct = re.fullmatch(r"\s*[\(\[]?([A-Za-z])[\)\].]?\s*", text)
    if direct is None:
        direct = re.fullmatch(r"\*\*([A-Za-z])\*\*", text)
    if direct and direct.group(1).upper() in valid:
        return direct.group(1).upper()

    patterns = (
        r"(?im)^\s*(?:final\s+)?answer\s*(?:is|:)\s*[\(\[]?([A-Za-z])[\)\]]?\s*[.!]?\s*$",
        r"(?im)^\s*option\s*[\(\[]?([A-Za-z])[\)\]]?\s*[.!]?\s*$",
        r"\\boxed\{\s*([A-Za-z])\s*\}",
    )
    matches = {
        match.upper()
        for pattern in patterns
        for match in re.findall(pattern, text)
        if match.upper() in valid
    }
    if len(matches) != 1:
        raise MalformedModelOutput(
            "Could not extract one unambiguous multiple-choice option"
        )
    return next(iter(matches))


def _braced_value(text: str, start: int) -> tuple:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise ValueError("Expected a braced LaTeX value")
    depth = 1
    for position in range(start + 1, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : position], position + 1
    raise ValueError("Unbalanced LaTeX braces")


def _replace_latex_fractions(value: str) -> str:
    pattern = re.compile(r"\\(?:dfrac|tfrac|frac)")
    output = []
    cursor = 0
    while True:
        match = pattern.search(value, cursor)
        if match is None:
            output.append(value[cursor:])
            return "".join(output)
        output.append(value[cursor : match.start()])
        try:
            numerator, after_numerator = _braced_value(value, match.end())
            denominator, after_denominator = _braced_value(value, after_numerator)
        except ValueError:
            output.append(value[match.start() :])
            return "".join(output)
        output.append(
            f"(({_replace_latex_fractions(numerator)})/"
            f"({_replace_latex_fractions(denominator)}))"
        )
        cursor = after_denominator


def _strip_math_wrappers(value: str) -> str:
    value = value.strip()
    changed = True
    while changed:
        changed = False
        pairs = (("**", "**"), ("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
        for opening, closing in pairs:
            if value.startswith(opening) and value.endswith(closing) and len(value) > len(opening) + len(closing):
                value = value[len(opening) : -len(closing)].strip()
                changed = True
                break
        if value.startswith(r"\boxed{"):
            try:
                boxed, end = _braced_value(value, len(r"\boxed"))
            except ValueError:
                pass
            else:
                if not value[end:].strip():
                    value = boxed.strip()
                    changed = True
    return value


def _normalize_math(value: str) -> str:
    value = _strip_math_wrappers(value)
    assignment = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\s*=\s*(.+)", value, re.DOTALL)
    if assignment:
        value = _strip_math_wrappers(assignment.group(1))
    value = value.replace("−", "-").replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\\(?:,|!|;|:|quad|qquad)", "", value)
    value = _replace_latex_fractions(value)
    value = value.replace(r"\pi", "pi")
    value = re.sub(r"(?<=\d)pi", "*pi", value)
    value = re.sub(r"\s+", "", value)
    return value.rstrip(".")


def _simple_math_value(node: ast.AST):
    """Evaluate only exact rational arithmetic and rational multiples of pi."""
    if isinstance(node, ast.Expression):
        return _simple_math_value(node.body)
    if isinstance(node, ast.Tuple):
        return tuple(_simple_math_value(item) for item in node.elts)
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, (int, float)):
            return Fraction(str(node.value)), Fraction(0)
        raise ValueError("Unsupported math constant")
    if isinstance(node, ast.Name) and node.id == "pi":
        return Fraction(0), Fraction(1)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        rational, pi_coefficient = _simple_math_value(node.operand)
        if isinstance(node.op, ast.USub):
            return -rational, -pi_coefficient
        return rational, pi_coefficient
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _simple_math_value(node.left)
        right = _simple_math_value(node.right)
        if not (
            isinstance(left, tuple)
            and len(left) == 2
            and all(isinstance(item, Fraction) for item in left)
            and isinstance(right, tuple)
            and len(right) == 2
            and all(isinstance(item, Fraction) for item in right)
        ):
            raise ValueError("Arithmetic on structured math values is unsupported")
        left_rational, left_pi = left
        right_rational, right_pi = right
        if isinstance(node.op, ast.Add):
            return left_rational + right_rational, left_pi + right_pi
        if isinstance(node.op, ast.Sub):
            return left_rational - right_rational, left_pi - right_pi
        if isinstance(node.op, ast.Mult):
            if left_pi and right_pi:
                raise ValueError("Products containing pi squared are unsupported")
            return (
                left_rational * right_rational,
                left_rational * right_pi + left_pi * right_rational,
            )
        if right_pi or not right_rational:
            raise ValueError("Division by a symbolic or zero value is unsupported")
        return left_rational / right_rational, left_pi / right_rational
    raise ValueError("Unsupported math expression")


def _parse_simple_math(value: str):
    return _simple_math_value(ast.parse(value, mode="eval"))


def grade_math(output: object, reference_answer: str) -> bool:
    candidate = _normalize_math(_require_output(output))
    reference = _normalize_math(reference_answer)
    if not reference:
        raise ValueError("Math reference answer is empty")
    try:
        return _parse_simple_math(candidate) == _parse_simple_math(reference)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return candidate == reference


def _last_boxed_value(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return ""
    index = start + len(marker)
    depth = 1
    for position in range(index, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[index:position].strip()
    return ""


def parse_math_answer(output: object) -> str:
    """Deterministically extract a final math answer without evaluating it."""
    text = _require_output(output)
    boxed = _last_boxed_value(text)
    if boxed:
        return boxed

    marked = re.findall(
        r"(?im)^\s*(?:therefore,?\s*)?(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*(.+?)\s*$",
        text,
    )
    if marked:
        candidate = marked[-1].strip().strip("$ ")
        if candidate:
            return candidate

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and len(lines[0]) <= 200:
        candidate = lines[0].strip().strip("$ ")
        if candidate and not re.search(r"\b(?:because|therefore|thus|answer)\b", candidate, re.I):
            return candidate
    raise MalformedModelOutput("Could not extract a deterministic final math answer")


def extract_code(output: object) -> str:
    code = _require_output(output)
    fenced_blocks = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```", code, re.DOTALL | re.IGNORECASE
    )
    if fenced_blocks:
        code = fenced_blocks[0].strip()
    elif not re.search(r"(?m)^\s*(?:from\s+\S+\s+import|import\s+|def\s+|class\s+|print\s*\(|\w+\s*=)", code):
        raise MalformedModelOutput("Could not extract executable Python code")
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


def run_code_tests(
    code: str, tests: Sequence[Mapping[str, str]]
) -> bool:
    """Run a restricted Python subset with isolation flags and strict timeouts."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise MalformedModelOutput(f"Generated code is not valid Python: {error}") from error
    allowed_imports = {
        "bisect",
        "collections",
        "functools",
        "heapq",
        "itertools",
        "math",
        "sys",
    }
    forbidden_names = {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "locals",
        "open",
        "setattr",
        "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".", 1)[0] for alias in node.names}
            disallowed = imported - allowed_imports
            if disallowed:
                raise UnsafeGeneratedCode(
                    f"Import is not allowed in the pilot code grader: {sorted(disallowed)}"
                )
        if isinstance(node, ast.ImportFrom):
            imported = (node.module or "").split(".", 1)[0]
            if node.level or imported not in allowed_imports:
                raise UnsafeGeneratedCode(
                    f"Import is not allowed in the pilot code grader: {node.module or '<relative>'}"
                )
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise UnsafeGeneratedCode(f"Forbidden name in generated code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeGeneratedCode("Dunder attribute access is disabled in generated code")
    for test in tests:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", code],
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


def run_trusted_dry_run_tests(
    code: str, tests: Sequence[Mapping[str, str]]
) -> bool:
    """Backward-compatible name for the deterministic subprocess runner."""
    return run_code_tests(code, tests)
