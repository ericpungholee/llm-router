"""Central registry for every model in the planned comparison."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class GenerationSettings:
    temperature: Optional[float]
    reasoning_setting: str
    max_output_tokens: int

    @property
    def temperature_record(self) -> str:
        return "provider_default" if self.temperature is None else str(self.temperature)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    creator: str
    canonical_model_name: str
    api_model_identifier: Optional[str]
    inference_provider: str
    model_type: str
    generation: GenerationSettings
    input_price_per_million_usd: Optional[float]
    output_price_per_million_usd: Optional[float]
    enabled: bool
    resolution_note: str

    @property
    def live_ready(self) -> bool:
        return (
            self.api_model_identifier is not None
            and self.input_price_per_million_usd is not None
            and self.output_price_per_million_usd is not None
        )


DEFAULT_MAX_OUTPUT_TOKENS = 4096

MODEL_REGISTRY: Tuple[ModelConfig, ...] = (
    ModelConfig(
        key="openai_gpt_5_6_sol",
        creator="OpenAI",
        canonical_model_name="GPT-5.6 Sol",
        api_model_identifier="gpt-5.6-sol",
        inference_provider="openai",
        model_type="closed",
        generation=GenerationSettings(None, "reasoning_effort=medium", DEFAULT_MAX_OUTPUT_TOKENS),
        input_price_per_million_usd=4.00,
        output_price_per_million_usd=20.00,
        enabled=True,
        resolution_note="Verified in official OpenAI model documentation.",
    ),
    ModelConfig(
        key="anthropic_claude_opus_5",
        creator="Anthropic",
        canonical_model_name="Claude Opus 5",
        api_model_identifier="claude-opus-5",
        inference_provider="anthropic",
        model_type="closed",
        generation=GenerationSettings(None, "adaptive_thinking;effort=high", DEFAULT_MAX_OUTPUT_TOKENS),
        input_price_per_million_usd=5.00,
        output_price_per_million_usd=25.00,
        enabled=True,
        resolution_note="Verified for the first-party Claude API.",
    ),
    ModelConfig(
        key="google_gemini_3_1_pro",
        creator="Google",
        canonical_model_name="Gemini 3.1 Pro",
        api_model_identifier="gemini-3.1-pro-preview",
        inference_provider="google",
        model_type="closed",
        generation=GenerationSettings(None, "thinking_level=high", DEFAULT_MAX_OUTPUT_TOKENS),
        input_price_per_million_usd=2.00,
        output_price_per_million_usd=12.00,
        enabled=True,
        resolution_note="Preview ID and prices apply below 200K input tokens.",
    ),
    ModelConfig(
        key="xai_grok_4_6",
        creator="xAI",
        canonical_model_name="Grok 4.6",
        api_model_identifier="grok-4.6",
        inference_provider="xai",
        model_type="closed",
        generation=GenerationSettings(None, "reasoning_effort=high", DEFAULT_MAX_OUTPUT_TOKENS),
        input_price_per_million_usd=2.00,
        output_price_per_million_usd=6.00,
        enabled=True,
        resolution_note="Standard prices apply below 200K input tokens.",
    ),
    ModelConfig(
        key="deepseek_v4_1_flash",
        creator="DeepSeek",
        canonical_model_name="DeepSeek V4.1 Flash",
        api_model_identifier="deepseek-flash",
        inference_provider="deepseek",
        model_type="open_weight",
        generation=GenerationSettings(None, "thinking=enabled", DEFAULT_MAX_OUTPUT_TOKENS),
        input_price_per_million_usd=None,
        output_price_per_million_usd=None,
        enabled=True,
        resolution_note=(
            "Official alias verified. V4.1 pricing changed on 2026-09-10 and must "
            "be transcribed from the current pricing table before live use."
        ),
    ),
    ModelConfig(
        key="qwen_3_8_2_4t_a95b",
        creator="Qwen / Alibaba",
        canonical_model_name="Qwen3.8 2.4T-A95B",
        api_model_identifier="qwen/qwen3.8-2.4t-a95b",
        inference_provider="openrouter",
        model_type="open_weight",
        generation=GenerationSettings(None, "thinking_only", DEFAULT_MAX_OUTPUT_TOKENS),
        input_price_per_million_usd=2.00,
        output_price_per_million_usd=6.00,
        enabled=True,
        resolution_note="OpenRouter listed pricing: $2/M input and $6/M output.",
    ),
)


def enabled_models() -> Tuple[ModelConfig, ...]:
    return tuple(model for model in MODEL_REGISTRY if model.enabled)


def validate_registry(models: Sequence[ModelConfig] = MODEL_REGISTRY) -> None:
    if not models:
        raise ValueError("Model registry is empty")

    keys = [model.key for model in models]
    names = [model.canonical_model_name for model in models]
    if len(keys) != len(set(keys)):
        raise ValueError("Model registry contains duplicate keys")
    if len(names) != len(set(names)):
        raise ValueError("Model registry contains duplicate canonical model names")

    for model in models:
        for field_name in (
            "key",
            "creator",
            "canonical_model_name",
            "inference_provider",
            "model_type",
            "resolution_note",
        ):
            if not str(getattr(model, field_name)).strip():
                raise ValueError(f"{model.key or '<unknown>'} is missing {field_name}")
        if model.model_type not in {"closed", "open_weight"}:
            raise ValueError(f"{model.key} has invalid model_type: {model.model_type}")
        if model.generation.max_output_tokens <= 0:
            raise ValueError(f"{model.key} has invalid max_output_tokens")
        if not model.generation.reasoning_setting.strip():
            raise ValueError(f"{model.key} is missing reasoning_setting")
        if model.generation.temperature is not None and not 0 <= model.generation.temperature <= 2:
            raise ValueError(f"{model.key} has invalid temperature")

        prices = (
            model.input_price_per_million_usd,
            model.output_price_per_million_usd,
        )
        if any(price is not None and price < 0 for price in prices):
            raise ValueError(f"{model.key} has invalid pricing")
        if (model.api_model_identifier is None or any(price is None for price in prices)) and not model.resolution_note:
            raise ValueError(f"{model.key} has unresolved metadata without a note")
