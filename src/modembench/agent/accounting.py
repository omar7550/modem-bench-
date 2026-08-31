"""Dated five-rate price table and the per-run cost basis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

ACCOUNTING_POLICY_VERSION = "modembench-accounting-v1"

# Date this table was transcribed from the published price list.
PRICE_TABLE_DATE = date(2026, 8, 7)

# Component multiples relative to base input rate; provenance only, the table stores products.
CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.00
CACHE_READ_MULTIPLIER = 0.10
# Batches API discount, applied to every component.
BATCH_MULTIPLIER = 0.50

TOKENS_PER_MTOK = 1_000_000


class PricingError(RuntimeError):
    """An unpriceable model, or a date outside every published price period."""


@dataclass(frozen=True)
class Rates:
    """Five USD-per-million-token rates. Never derived at call time from a multiplier."""

    input_usd_per_mtok: float
    cache_write_5m_usd_per_mtok: float
    cache_write_1h_usd_per_mtok: float
    cache_read_usd_per_mtok: float
    output_usd_per_mtok: float

    def as_dict(self) -> dict[str, float]:
        return {
            "input_usd_per_mtok": self.input_usd_per_mtok,
            "cache_write_5m_usd_per_mtok": self.cache_write_5m_usd_per_mtok,
            "cache_write_1h_usd_per_mtok": self.cache_write_1h_usd_per_mtok,
            "cache_read_usd_per_mtok": self.cache_read_usd_per_mtok,
            "output_usd_per_mtok": self.output_usd_per_mtok,
        }


@dataclass(frozen=True)
class PricePeriod:
    """One model's rates over a closed or open-ended date range."""

    effective_from: date
    effective_through: date | None
    rates: Rates
    note: str

    def covers(self, on: date) -> bool:
        if on < self.effective_from:
            return False
        return self.effective_through is None or on <= self.effective_through


# The introductory Sonnet 5 rate expires 2026-08-31; both sealed accesses land after it.
SONNET_5_INTRO_THROUGH = date(2026, 8, 31)

PRICE_TABLE: dict[str, tuple[PricePeriod, ...]] = {
    "claude-sonnet-5": (
        PricePeriod(
            effective_from=date(2026, 1, 1),
            effective_through=SONNET_5_INTRO_THROUGH,
            rates=Rates(
                input_usd_per_mtok=2.00,
                cache_write_5m_usd_per_mtok=2.50,
                cache_write_1h_usd_per_mtok=4.00,
                cache_read_usd_per_mtok=0.20,
                output_usd_per_mtok=10.00,
            ),
            note="introductory rate; expires 2026-08-31",
        ),
        PricePeriod(
            effective_from=date(2026, 9, 1),
            effective_through=None,
            rates=Rates(
                input_usd_per_mtok=3.00,
                cache_write_5m_usd_per_mtok=3.75,
                cache_write_1h_usd_per_mtok=6.00,
                cache_read_usd_per_mtok=0.30,
                output_usd_per_mtok=15.00,
            ),
            note="list rate",
        ),
    ),
    "claude-opus-5": (
        PricePeriod(
            effective_from=date(2026, 1, 1),
            effective_through=None,
            rates=Rates(
                input_usd_per_mtok=5.00,
                cache_write_5m_usd_per_mtok=6.25,
                cache_write_1h_usd_per_mtok=10.00,
                cache_read_usd_per_mtok=0.50,
                output_usd_per_mtok=25.00,
            ),
            note="list rate",
        ),
    ),
}

PRICED_MODELS = tuple(sorted(PRICE_TABLE))

# Minimum cacheable prefix per model, in tokens. A shorter prefix silently does not cache.
CACHE_MINIMUM_TOKENS: dict[str, int] = {
    "claude-sonnet-5": 1024,
    "claude-opus-5": 512,
}


def cache_minimum_tokens(model: str) -> int:
    """The model's minimum cacheable prefix; unknown model is a hard error."""
    try:
        return CACHE_MINIMUM_TOKENS[model]
    except KeyError:
        raise PricingError(
            f"no minimum cacheable prefix is published for model {model!r}"
        ) from None


def rates_for(model: str, on: date | None = None) -> Rates:
    """Resolve one model's five rates on a date. Unknown model or gap => hard error."""
    periods = PRICE_TABLE.get(model)
    if periods is None:
        raise PricingError(f"no price period is published for model {model!r}")
    when = on or PRICE_TABLE_DATE
    for period in periods:
        if period.covers(when):
            return period.rates
    raise PricingError(f"no price period covers {when.isoformat()} for model {model!r}")


def price_period_for(model: str, on: date | None = None) -> PricePeriod:
    periods = PRICE_TABLE.get(model)
    if periods is None:
        raise PricingError(f"no price period is published for model {model!r}")
    when = on or PRICE_TABLE_DATE
    for period in periods:
        if period.covers(when):
            return period
    raise PricingError(f"no price period covers {when.isoformat()} for model {model!r}")


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PricingError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class TokenUsage:
    """One response's token counts, split along the axes the price table charges on.

    When the SDK omits the cache-write breakdown, the whole figure is attributed to the
    cheaper 5-minute rate and cache_write_breakdown_observed records the assumption.
    """

    input_tokens: int = 0
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    cache_write_breakdown_observed: bool = True

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "cache_creation_5m_tokens",
            "cache_creation_1h_tokens",
            "cache_read_tokens",
            "output_tokens",
        ):
            _count(getattr(self, name), name)

    @property
    def prompt_tokens_total(self) -> int:
        """A size, not a cost basis: the components are charged at different rates."""
        return (
            self.input_tokens
            + self.cache_creation_5m_tokens
            + self.cache_creation_1h_tokens
            + self.cache_read_tokens
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cache_creation_5m_tokens": self.cache_creation_5m_tokens,
            "cache_creation_1h_tokens": self.cache_creation_1h_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_breakdown_observed": self.cache_write_breakdown_observed,
            "prompt_tokens_total_SIZE_NOT_COST": self.prompt_tokens_total,
        }

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_creation_5m_tokens=(
                self.cache_creation_5m_tokens + other.cache_creation_5m_tokens
            ),
            cache_creation_1h_tokens=(
                self.cache_creation_1h_tokens + other.cache_creation_1h_tokens
            ),
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_breakdown_observed=(
                self.cache_write_breakdown_observed and other.cache_write_breakdown_observed
            ),
        )


def _attr(source: Any, name: str, default: int = 0) -> int:
    if isinstance(source, dict):
        value = source.get(name, default)
    else:
        value = getattr(source, name, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise PricingError(f"usage field {name!r} is not an integer")
    return int(value)


def usage_from_response(usage: Any) -> TokenUsage:
    """Read a provider usage object (SDK model or plain mapping) into the cost basis."""
    breakdown = usage.get("cache_creation") if isinstance(usage, dict) else getattr(
        usage, "cache_creation", None
    )
    total_creation = _attr(usage, "cache_creation_input_tokens")
    if breakdown is None:
        five = total_creation
        hour = 0
        observed = total_creation == 0
    else:
        five = _attr(breakdown, "ephemeral_5m_input_tokens")
        hour = _attr(breakdown, "ephemeral_1h_input_tokens")
        observed = True
        if total_creation and five + hour != total_creation:
            # Breakdown and total disagree: trust the total, shortfall to the cheaper rate.
            five = max(0, total_creation - hour)
            observed = False
    return TokenUsage(
        input_tokens=_attr(usage, "input_tokens"),
        cache_creation_5m_tokens=five,
        cache_creation_1h_tokens=hour,
        cache_read_tokens=_attr(usage, "cache_read_input_tokens"),
        output_tokens=_attr(usage, "output_tokens"),
        cache_write_breakdown_observed=observed,
    )


def cost_breakdown(
    usage: TokenUsage, *, model: str, on: date | None = None, batch: bool = False
) -> dict[str, Any]:
    """Per-component dollars for one usage record. The batch multiplier hits every row."""
    rates = rates_for(model, on)
    factor = BATCH_MULTIPLIER if batch else 1.0
    components = {
        "input": (usage.input_tokens, rates.input_usd_per_mtok),
        "cache_write_5m": (usage.cache_creation_5m_tokens, rates.cache_write_5m_usd_per_mtok),
        "cache_write_1h": (usage.cache_creation_1h_tokens, rates.cache_write_1h_usd_per_mtok),
        "cache_read": (usage.cache_read_tokens, rates.cache_read_usd_per_mtok),
        "output": (usage.output_tokens, rates.output_usd_per_mtok),
    }
    dollars = {
        name: tokens * rate * factor / TOKENS_PER_MTOK for name, (tokens, rate) in components.items()
    }
    return {
        "model": model,
        "priced_on": (on or PRICE_TABLE_DATE).isoformat(),
        "price_table_date": PRICE_TABLE_DATE.isoformat(),
        "price_note": price_period_for(model, on).note,
        "batch": bool(batch),
        "batch_multiplier": BATCH_MULTIPLIER if batch else 1.0,
        "rates_usd_per_mtok": rates.as_dict(),
        "tokens": usage.as_dict(),
        "usd": dollars,
        "total_usd": sum(dollars.values()),
    }


def cost_usd(
    usage: TokenUsage, *, model: str, on: date | None = None, batch: bool = False
) -> float:
    """Total USD for one usage record under the five-rate table."""
    return float(cost_breakdown(usage, model=model, on=on, batch=batch)["total_usd"])


# --- campaign projection ------------------------------------------------------------------
# Projected token shape of one no-evaluator-feedback run, derived from measured byte counts
# at harness.BYTES_PER_TOKEN_NOMINAL. The projection test recomputes the input rows from the
# harness and fails on drift.
PROJECTED_TOOL_TURNS = 2
PROJECTED_RUN_USAGE = TokenUsage(
    input_tokens=1_923,
    cache_creation_5m_tokens=1_823,
    cache_read_tokens=3_646,
    output_tokens=12_000,
)
# One attempt-unit per dev capture; at N attempts per unit the sweep issues 40 x N runs.
DEV_SWEEP_RUNS = 40

# The sealed campaign is denominated in attempt-units (one attempt, one signal, one arm),
# not runs: dollars are attempt_units * N * unit cost. arms.ledger allocates in these units.
SEALED_SIGNALS = 60
SEALED_REPLICATES = 3
SEALED_ARMS = 2
SEALED_ATTEMPT_UNITS_PER_ACCESS = SEALED_SIGNALS * SEALED_REPLICATES * SEALED_ARMS

# Output tokens per run swept in the sensitivity table.
OUTPUT_SENSITIVITY = (12_000, 30_000, 60_000, 100_000)


def sealed_campaign_attempts(n_attempts: int, *, accesses: int = 1) -> int:
    """One-shot-shaped attempts a sealed campaign issues at ``n_attempts`` per unit."""
    return (
        SEALED_ATTEMPT_UNITS_PER_ACCESS
        * _count(n_attempts, "n_attempts")
        * _count(accesses, "accesses")
    )


def campaign_cost_usd(
    *,
    model: str,
    runs: int,
    on: date | None = None,
    usage: TokenUsage | None = None,
    output_tokens: int | None = None,
    batch: bool = False,
) -> float:
    """Total USD for ``runs`` identical runs, optionally overriding the output estimate."""
    shape = usage or PROJECTED_RUN_USAGE
    if output_tokens is not None:
        shape = TokenUsage(
            input_tokens=shape.input_tokens,
            cache_creation_5m_tokens=shape.cache_creation_5m_tokens,
            cache_creation_1h_tokens=shape.cache_creation_1h_tokens,
            cache_read_tokens=shape.cache_read_tokens,
            output_tokens=output_tokens,
        )
    return cost_usd(shape, model=model, on=on, batch=batch) * int(runs)


def uncached_input_tokens_for_tool_turns(
    tool_turns: int,
    *,
    task_tokens: int,
    assistant_turn_tokens: int = 30,
    tool_result_tokens: int = 200,
) -> int:
    """Uncached input tokens for a run with ``tool_turns`` tool calls; grows quadratically
    because the conversation tail sits behind the cache breakpoint."""
    requests = int(tool_turns) + 1
    per_exchange = int(assistant_turn_tokens) + int(tool_result_tokens)
    # Request k (1-indexed) carries the task message plus the k-1 exchanges before it.
    return sum(int(task_tokens) + (k - 1) * per_exchange for k in range(1, requests + 1))


def projected_total_cost_usd(
    *,
    input_tokens: int,
    output_estimate_tokens: int,
    model: str,
    on: date | None = None,
    batch: bool = False,
) -> float:
    """Project a run's total cost from a pre-flight input count plus an output estimate."""
    usage = TokenUsage(
        input_tokens=_count(input_tokens, "input_tokens"),
        output_tokens=_count(output_estimate_tokens, "output_estimate_tokens"),
    )
    return cost_usd(usage, model=model, on=on, batch=batch)
