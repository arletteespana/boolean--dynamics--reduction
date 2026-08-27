from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from time import perf_counter

from .attractors import (
    attractor_signature,
    canonical_cycle,
    compile_synchronous_update,
)
from .utils import BNetModel


def _deterministic_seed(
    base_seed: int,
    network: str,
    method: str,
    variant: str,
) -> int:
    payload = (
        f"{base_seed}|{network}|{method}|{variant}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def _wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    """
    if trials <= 0:
        return 0.0, 0.0

    p = successes / trials
    z2 = z * z

    denominator = 1.0 + z2 / trials

    center = (
        p + z2 / (2.0 * trials)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1.0 - p) / trials
            + z2 / (4.0 * trials * trials)
        )
        / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def _assign_path_to_known_attractor(
    path: list[int],
    successor_state: int,
    state_cache: dict[int, tuple[tuple[int, ...], int]],
) -> None:
    attractor, distance = state_cache[
        successor_state
    ]

    for state in reversed(path):
        distance += 1
        state_cache[state] = (
            attractor,
            distance,
        )


def _assign_new_cycle(
    path: list[int],
    cycle_start: int,
    state_cache: dict[int, tuple[tuple[int, ...], int]],
) -> tuple[int, ...]:
    cycle_states = path[cycle_start:]
    attractor = canonical_cycle(cycle_states)

    for state in cycle_states:
        state_cache[state] = (
            attractor,
            0,
        )

    distance = 0

    for state in reversed(path[:cycle_start]):
        distance += 1
        state_cache[state] = (
            attractor,
            distance,
        )

    return attractor


def _trace_state(
    initial_state: int,
    successor,
    state_cache: dict[int, tuple[tuple[int, ...], int]],
) -> tuple[tuple[int, ...], int]:
    """
    Follow one trajectory until it reaches a cached state or closes a cycle.

    Returns:
        (attractor, transient_length)
    """
    if initial_state in state_cache:
        return state_cache[initial_state]

    path: list[int] = []
    local_position: dict[int, int] = {}
    current = initial_state

    while (
        current not in state_cache
        and current not in local_position
    ):
        local_position[current] = len(path)
        path.append(current)
        current = successor(current)

    if current in state_cache:
        _assign_path_to_known_attractor(
            path,
            current,
            state_cache,
        )
    else:
        cycle_start = local_position[current]

        _assign_new_cycle(
            path,
            cycle_start,
            state_cache,
        )

    return state_cache[initial_state]


def _basin_rows(
    network: str,
    method: str,
    variant: str,
    analysis_type: str,
    attractor_counts: Counter,
    total_initial_states: int,
) -> list[dict[str, object]]:
    rows = []

    sorted_attractors = sorted(
        attractor_counts,
        key=lambda cycle: (
            len(cycle),
            cycle,
        ),
    )

    for i, attractor in enumerate(
        sorted_attractors,
        1,
    ):
        count = attractor_counts[attractor]
        fraction = (
            count / total_initial_states
            if total_initial_states
            else 0.0
        )

        if analysis_type == "monte_carlo":
            ci_low, ci_high = _wilson_interval(
                count,
                total_initial_states,
            )
        else:
            ci_low = fraction
            ci_high = fraction

        rows.append(
            {
                "network": network,
                "method": method,
                "variant": variant,
                "analysis_type": analysis_type,
                "attractor_id": f"A{i:03d}",
                "attractor_signature": (
                    attractor_signature(attractor)
                ),
                "period": len(attractor),
                "basin_count_or_sample_count": count,
                "basin_fraction": fraction,
                "basin_fraction_ci95_low": ci_low,
                "basin_fraction_ci95_high": ci_high,
            }
        )

    return rows


def analyze_exact_dynamics(
    model: BNetModel,
    network: str,
    method: str,
    variant: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """
    Exhaustively analyze synchronous dynamics over all 2^n states.
    """
    n = len(model.nodes)
    total_states = 1 << n
    successor = compile_synchronous_update(
        model
    )

    state_cache: dict[
        int,
        tuple[tuple[int, ...], int],
    ] = {}

    basin_counts: Counter = Counter()

    start = perf_counter()

    transient_sum = 0
    transient_max = 0

    for initial_state in range(total_states):
        attractor, transient = _trace_state(
            initial_state,
            successor,
            state_cache,
        )

        basin_counts[attractor] += 1
        transient_sum += transient

        if transient > transient_max:
            transient_max = transient

    elapsed = perf_counter() - start

    periods = sorted(
        len(attractor)
        for attractor in basin_counts
    )

    fixed_observed = sum(
        1
        for period in periods
        if period == 1
    )

    periodic_observed = sum(
        1
        for period in periods
        if period > 1
    )

    metrics = {
        "analysis_type": "exact",
        "attractor_count_is_complete": True,
        "initial_states_analyzed": total_states,
        "state_space_size": str(total_states),
        "coverage_fraction": 1.0,
        "attractors_observed": len(basin_counts),
        "fixed_attractors_observed": fixed_observed,
        "periodic_attractors_observed": periodic_observed,
        "attractor_periods": ";".join(
            str(period)
            for period in periods
        ),
        "mean_transient_length": (
            transient_sum / total_states
            if total_states
            else 0.0
        ),
        "max_transient_length": transient_max,
        "dynamic_analysis_time_seconds": elapsed,
        "monte_carlo_samples_requested": "",
        "random_seed": "",
    }

    basin_rows = _basin_rows(
        network=network,
        method=method,
        variant=variant,
        analysis_type="exact",
        attractor_counts=basin_counts,
        total_initial_states=total_states,
    )

    return metrics, basin_rows


def _sample_unique_states(
    n: int,
    samples: int,
    rng: random.Random,
) -> list[int]:
    selected: set[int] = set()

    while len(selected) < samples:
        selected.add(
            rng.getrandbits(n)
        )

    return list(selected)


def analyze_monte_carlo_dynamics(
    model: BNetModel,
    network: str,
    method: str,
    variant: str,
    samples: int = 10_000,
    base_seed: int = 2026,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """
    Estimate synchronous basin and transient properties from uniformly sampled
    initial states.

    The reported attractor count is the number observed in the sample. It is
    not claimed to be complete.
    """
    if samples <= 0:
        raise ValueError(
            "Monte Carlo sample size must be positive."
        )

    n = len(model.nodes)
    total_states = 1 << n

    if samples >= total_states:
        return analyze_exact_dynamics(
            model,
            network,
            method,
            variant,
        )

    seed = _deterministic_seed(
        base_seed,
        network,
        method,
        variant,
    )

    rng = random.Random(seed)

    initial_states = _sample_unique_states(
        n,
        samples,
        rng,
    )

    successor = compile_synchronous_update(
        model
    )

    state_cache: dict[
        int,
        tuple[tuple[int, ...], int],
    ] = {}

    basin_counts: Counter = Counter()

    transient_sum = 0
    transient_max = 0

    start = perf_counter()

    for initial_state in initial_states:
        attractor, transient = _trace_state(
            initial_state,
            successor,
            state_cache,
        )

        basin_counts[attractor] += 1
        transient_sum += transient

        if transient > transient_max:
            transient_max = transient

    elapsed = perf_counter() - start

    periods = sorted(
        len(attractor)
        for attractor in basin_counts
    )

    fixed_observed = sum(
        1
        for period in periods
        if period == 1
    )

    periodic_observed = sum(
        1
        for period in periods
        if period > 1
    )

    metrics = {
        "analysis_type": "monte_carlo",
        "attractor_count_is_complete": False,
        "initial_states_analyzed": samples,
        "state_space_size": str(total_states),
        "coverage_fraction": (
            samples / total_states
        ),
        "attractors_observed": len(basin_counts),
        "fixed_attractors_observed": fixed_observed,
        "periodic_attractors_observed": periodic_observed,
        "attractor_periods": ";".join(
            str(period)
            for period in periods
        ),
        "mean_transient_length": (
            transient_sum / samples
        ),
        "max_transient_length": transient_max,
        "dynamic_analysis_time_seconds": elapsed,
        "monte_carlo_samples_requested": samples,
        "random_seed": seed,
    }

    basin_rows = _basin_rows(
        network=network,
        method=method,
        variant=variant,
        analysis_type="monte_carlo",
        attractor_counts=basin_counts,
        total_initial_states=samples,
    )

    return metrics, basin_rows


def analyze_synchronous_dynamics(
    model: BNetModel,
    network: str,
    method: str,
    variant: str,
    exact_dimension: int = 20,
    monte_carlo_samples: int = 10_000,
    base_seed: int = 2026,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """
    Select exact enumeration or Monte Carlo automatically from model dimension.
    """
    if exact_dimension < 0:
        raise ValueError(
            "Exact dimension threshold cannot be negative."
        )

    if len(model.nodes) <= exact_dimension:
        return analyze_exact_dynamics(
            model,
            network,
            method,
            variant,
        )

    return analyze_monte_carlo_dynamics(
        model,
        network,
        method,
        variant,
        samples=monte_carlo_samples,
        base_seed=base_seed,
    )
