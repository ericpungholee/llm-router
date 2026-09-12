"""Preflight and incremental spend controls for live evaluation runs."""

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence, Tuple, Union

from model_registry import ModelConfig
from providers import ProviderResponse


DEFAULT_RUN_SPEND_CAP_USD = Decimal("1.00")
DEFAULT_GLOBAL_MAX_SPEND_USD = Decimal("10.00")


class SpendLimitError(RuntimeError):
    """Raised when a live run cannot safely make or record a model call."""


class SpendPreflightError(SpendLimitError):
    """Raised before a live run starts when its maximum cannot be bounded."""


class SpendCapExceeded(SpendLimitError):
    """Raised before a call that could exceed the configured run cap."""


def parse_usd(value: object, *, field_name: str) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field_name} must be a valid dollar amount") from error
    if not amount.is_finite() or amount <= 0:
        raise ValueError(f"{field_name} must be greater than $0.00")
    return amount


def global_max_spend_usd() -> Decimal:
    configured = os.getenv("GLOBAL_MAX_SPEND_USD", str(DEFAULT_GLOBAL_MAX_SPEND_USD))
    return parse_usd(configured, field_name="GLOBAL_MAX_SPEND_USD")


def run_spend_cap_usd(value: Optional[object] = None) -> Decimal:
    if value is None:
        return DEFAULT_RUN_SPEND_CAP_USD
    return parse_usd(value, field_name="run spend cap")


def conservative_input_tokens(prompt: str) -> int:
    """Bound input tokens conservatively before a provider has tokenized text."""
    # Byte length bounds byte-level tokenization even for non-ASCII benchmark text.
    return max(1, len(prompt.encode("utf-8")))


PlannedCall = Union[Tuple[ModelConfig, str], Tuple[ModelConfig, str, int]]


def max_call_cost_usd(
    model: ModelConfig, prompt: str, max_output_tokens: Optional[int] = None
) -> Decimal:
    if (
        model.input_price_per_million_usd is None
        or model.output_price_per_million_usd is None
    ):
        raise SpendPreflightError(
            f"Cannot estimate maximum spend: {model.canonical_model_name} has unresolved pricing."
        )
    input_price = Decimal(str(model.input_price_per_million_usd))
    output_price = Decimal(str(model.output_price_per_million_usd))
    output_limit = max_output_tokens or model.generation.max_output_tokens
    if output_limit <= 0 or output_limit > model.generation.max_output_tokens:
        raise SpendPreflightError("Invalid per-call maximum output token limit.")
    return (
        Decimal(conservative_input_tokens(prompt)) * input_price
        + Decimal(output_limit) * output_price
    ) / Decimal(1_000_000)


def estimate_max_spend_usd(
    calls: Sequence[PlannedCall],
) -> Decimal:
    """Estimate the maximum token spend for all planned calls."""
    return sum(
        (
            max_call_cost_usd(call[0], call[1], call[2] if len(call) == 3 else None)
            for call in calls
        ),
        Decimal("0"),
    ).quantize(Decimal("0.000001"))


@dataclass(frozen=True)
class RunPlan:
    planned_calls: int
    enabled_models: Tuple[str, ...]
    estimated_max_cost_usd: Decimal
    configured_spend_cap_usd: Decimal
    global_max_spend_usd: Decimal

    def confirmation_text(self) -> str:
        models = ", ".join(self.enabled_models) or "none"
        return "\n".join(
            (
                "LIVE RUN SPEND CONFIRMATION",
                f"- Planned calls: {self.planned_calls}",
                f"- Enabled models: {models}",
                f"- Estimated maximum cost: ${self.estimated_max_cost_usd:.6f}",
                f"- Configured spend cap: ${self.configured_spend_cap_usd:.2f}",
                f"- Global maximum spend: ${self.global_max_spend_usd:.2f}",
            )
        )


def build_run_plan(
    calls: Sequence[PlannedCall],
    *,
    spend_cap: Optional[object] = None,
) -> RunPlan:
    global_cap = global_max_spend_usd()
    configured_cap = run_spend_cap_usd(spend_cap)
    if configured_cap > global_cap:
        raise SpendPreflightError(
            f"Run spend cap ${configured_cap:.2f} exceeds global maximum ${global_cap:.2f}."
        )
    estimated = estimate_max_spend_usd(calls)
    return RunPlan(
        planned_calls=len(calls),
        enabled_models=tuple(dict.fromkeys(call[0].canonical_model_name for call in calls)),
        estimated_max_cost_usd=estimated,
        configured_spend_cap_usd=configured_cap,
        global_max_spend_usd=global_cap,
    )


class SpendTracker:
    """Track actual cost monotonically and reject unsafe calls before dispatch."""

    def __init__(self, plan: RunPlan, *, initial_spend_usd: object = "0") -> None:
        self.plan = plan
        try:
            self._spent_usd = Decimal(str(initial_spend_usd))
        except (InvalidOperation, ValueError) as error:
            raise SpendPreflightError("Existing spend must be a valid dollar amount.") from error
        if not self._spent_usd.is_finite() or self._spent_usd < 0:
            raise SpendPreflightError("Existing spend must not be negative.")
        if self._spent_usd > plan.configured_spend_cap_usd:
            raise SpendPreflightError(
                "Existing completed-call spend already exceeds the configured run cap."
            )
        self._completed_calls = 0
        self._attempted_calls = 0

    @property
    def spent_usd(self) -> Decimal:
        return self._spent_usd

    @property
    def completed_calls(self) -> int:
        return self._completed_calls

    @property
    def attempted_calls(self) -> int:
        return self._attempted_calls

    def assert_can_call(
        self, model: ModelConfig, prompt: str, max_output_tokens: Optional[int] = None
    ) -> None:
        possible_cost = max_call_cost_usd(model, prompt, max_output_tokens)
        if self._spent_usd + possible_cost > self.plan.configured_spend_cap_usd:
            raise SpendCapExceeded(
                "Live run stopped before the next call: "
                f"${self._spent_usd + possible_cost:.6f} could exceed the "
                f"${self.plan.configured_spend_cap_usd:.2f} spend cap."
            )

    def record_call(self, model: ModelConfig, response: ProviderResponse) -> Decimal:
        if (
            model.input_price_per_million_usd is None
            or model.output_price_per_million_usd is None
        ):
            raise SpendPreflightError(
                f"Cannot record spend: {model.canonical_model_name} has unresolved pricing."
            )
        for field_name in ("input_tokens", "output_tokens"):
            token_count = getattr(response, field_name)
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count < 0
            ):
                raise SpendLimitError(f"Provider returned invalid {field_name}.")
        input_price = Decimal(str(model.input_price_per_million_usd))
        output_price = Decimal(str(model.output_price_per_million_usd))
        if response.provider_reported_cost_usd is not None:
            actual_cost = Decimal(str(response.provider_reported_cost_usd))
            if not actual_cost.is_finite() or actual_cost < 0:
                raise SpendLimitError("Provider returned invalid reported cost.")
        else:
            actual_cost = (
                Decimal(response.input_tokens) * input_price
                + Decimal(response.output_tokens) * output_price
            ) / Decimal(1_000_000)
        next_spend = self._spent_usd + actual_cost
        self._spent_usd = next_spend
        self._completed_calls += 1
        self._attempted_calls += 1
        if next_spend > self.plan.configured_spend_cap_usd:
            raise SpendCapExceeded(
                "Live run stopped after a call reported spend above the configured cap."
            )
        return actual_cost

    def record_failed_attempt(
        self,
        model: ModelConfig,
        prompt: str,
        max_output_tokens: Optional[int] = None,
    ) -> Decimal:
        """Conservatively charge a failed dispatch its maximum possible cost.

        Some providers can bill work even when a connection fails before usage is
        returned. Counting the full bound ensures retries cannot evade the cap.
        """
        possible_cost = max_call_cost_usd(model, prompt, max_output_tokens)
        next_spend = self._spent_usd + possible_cost
        self._spent_usd = next_spend
        self._attempted_calls += 1
        if next_spend > self.plan.configured_spend_cap_usd:
            raise SpendCapExceeded(
                "Live run stopped after a failed attempt consumed its reserved spend."
            )
        return possible_cost
