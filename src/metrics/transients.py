from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from time import perf_counter

import sympy as sp

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


def _state_space_layout(
    model: BNetModel,
) -> tuple[list[int], dict[int, int]]:
    """
    Return free variable positions and prescribed constant values.

    A variable whose Boolean update rule is constant is interpreted as a
    prescribed fixed component and is therefore fixed to that value already
    in the admissible initial conditions.
    """
    free_indices: list[int] = []
    fixed_values: dict[int, int] = {}

    for i, node in enumerate(model.nodes):
        rule = model.rules[node]

        if rule.free_symbols:
            free_indices.append(i)
            continue

        if rule == sp.true:
            fixed_values[i] = 1

        elif rule == sp.false:
            fixed_values[i] = 0

        else:
            raise ValueError(
                f"Constant rule for {node!r} could not be interpreted "
                f"as Boolean 0/1: {rule!r}"
            )

    return free_indices, fixed_values


def count_free_variables(
    model: BNetModel,
) -> int:
    free_indices, _ = _state_space_layout(
        model
    )

    return len(free_indices)


def count_fixed_variables(
    model: BNetModel,
) -> int:
    _, fixed_values = _state_space_layout(
        model
    )

    return len(fixed_values)


def _expand_free_state(
    free_state: int,
    free_indices: list[int],
    fixed_values: dict[int, int],
) -> int:
    """
    Embed a state of the free coordinates into the full serialized state.
    """
    full_state = 0

    for index, value in fixed_values.items():
        if value:
            full_state |= 1 << index

    for free_position, model_position in enumerate(
        free_indices
    ):
        if (free_state >> free_position) & 1:
            full_state |= 1 << model_position

    return full_state


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
    attractor = canonical_cycle(
        cycle_states
    )

    for state in cycle_states:
        state_cache[state] = (
            attractor,
            0,
        )

    distance = 0

    for state in reversed(
        path[:cycle_start]
    ):
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
        return state_cache[
            initial_state
        ]

    path: list[int] = []
    local_position: dict[int, int] = {}

    current = initial_state

    while (
        current not in state_cache
        and current not in local_position
    ):
        local_position[
            current
        ] = len(path)

        path.append(
            current
        )

        current = successor(
            current
        )

    if current in state_cache:
        _assign_path_to_known_attractor(
            path,
            current,
            state_cache,
        )

    else:
        cycle_start = local_position[
            current
        ]

        _assign_new_cycle(
            path,
            cycle_start,
            state_cache,
        )

    return state_cache[
        initial_state
    ]


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
        count = attractor_counts[
            attractor
        ]

        fraction = (
            count / total_initial_states
            if total_initial_states
            else 0.0
        )

        if analysis_type == "monte_carlo":
            ci_low, ci_high = (
                _wilson_interval(
                    count,
                    total_initial_states,
                )
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
                    attractor_signature(
                        attractor
                    )
                ),
                "period": len(
                    attractor
                ),
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
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """
    Exhaustively analyze synchronous dynamics over all admissible states.

    Constant-rule variables are fixed to their prescribed values in the
    initial conditions. Enumeration is therefore performed over the free
    Boolean coordinates only.
    """
    serialized_dimension = len(
        model.nodes
    )

    free_indices, fixed_values = (
        _state_space_layout(
            model
        )
    )

    free_variables = len(
        free_indices
    )

    fixed_variables = len(
        fixed_values
    )

    total_states = (
        1 << free_variables
    )

    serialized_state_space_size = (
        1 << serialized_dimension
    )

    successor = (
        compile_synchronous_update(
            model
        )
    )

    state_cache: dict[
        int,
        tuple[tuple[int, ...], int],
    ] = {}

    basin_counts: Counter = Counter()

    start = perf_counter()

    transient_sum = 0
    transient_max = 0

    for free_state in range(
        total_states
    ):
        initial_state = (
            _expand_free_state(
                free_state,
                free_indices,
                fixed_values,
            )
        )

        attractor, transient = (
            _trace_state(
                initial_state,
                successor,
                state_cache,
            )
        )

        basin_counts[
            attractor
        ] += 1

        transient_sum += transient

        if transient > transient_max:
            transient_max = transient

    elapsed = (
        perf_counter()
        - start
    )

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

        "serialized_dimension": serialized_dimension,
        "fixed_variables": fixed_variables,
        "free_variables": free_variables,

        "initial_states_analyzed": total_states,

        "state_space_size": str(
            total_states
        ),

        "serialized_state_space_size": str(
            serialized_state_space_size
        ),

        "effective_state_space_size": str(
            total_states
        ),

        "coverage_fraction": 1.0,

        "attractors_observed": len(
            basin_counts
        ),

        "fixed_attractors_observed": (
            fixed_observed
        ),

        "periodic_attractors_observed": (
            periodic_observed
        ),

        "attractor_periods": ";".join(
            str(period)
            for period in periods
        ),

        "mean_transient_length": (
            transient_sum / total_states
            if total_states
            else 0.0
        ),

        "max_transient_length": (
            transient_max
        ),

        "dynamic_analysis_time_seconds": (
            elapsed
        ),

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

    return (
        metrics,
        basin_rows,
    )


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

    return list(
        selected
    )


def analyze_monte_carlo_dynamics(
    model: BNetModel,
    network: str,
    method: str,
    variant: str,
    samples: int = 10_000,
    base_seed: int = 2026,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """
    Estimate synchronous basin and transient properties from uniformly sampled
    admissible initial states.

    Constant-rule variables are fixed to their prescribed Boolean values.
    Monte Carlo sampling is uniform over the remaining free coordinates.
    """
    if samples <= 0:
        raise ValueError(
            "Monte Carlo sample size must be positive."
        )

    serialized_dimension = len(
        model.nodes
    )

    free_indices, fixed_values = (
        _state_space_layout(
            model
        )
    )

    free_variables = len(
        free_indices
    )

    fixed_variables = len(
        fixed_values
    )

    total_states = (
        1 << free_variables
    )

    serialized_state_space_size = (
        1 << serialized_dimension
    )

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

    rng = random.Random(
        seed
    )

    sampled_free_states = (
        _sample_unique_states(
            free_variables,
            samples,
            rng,
        )
    )

    initial_states = [
        _expand_free_state(
            free_state,
            free_indices,
            fixed_values,
        )
        for free_state
        in sampled_free_states
    ]

    successor = (
        compile_synchronous_update(
            model
        )
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
        attractor, transient = (
            _trace_state(
                initial_state,
                successor,
                state_cache,
            )
        )

        basin_counts[
            attractor
        ] += 1

        transient_sum += transient

        if transient > transient_max:
            transient_max = transient

    elapsed = (
        perf_counter()
        - start
    )

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

        "serialized_dimension": serialized_dimension,
        "fixed_variables": fixed_variables,
        "free_variables": free_variables,

        "initial_states_analyzed": samples,

        "state_space_size": str(
            total_states
        ),

        "serialized_state_space_size": str(
            serialized_state_space_size
        ),

        "effective_state_space_size": str(
            total_states
        ),

        "coverage_fraction": (
            samples
            / total_states
        ),

        "attractors_observed": len(
            basin_counts
        ),

        "fixed_attractors_observed": (
            fixed_observed
        ),

        "periodic_attractors_observed": (
            periodic_observed
        ),

        "attractor_periods": ";".join(
            str(period)
            for period in periods
        ),

        "mean_transient_length": (
            transient_sum
            / samples
        ),

        "max_transient_length": (
            transient_max
        ),

        "dynamic_analysis_time_seconds": (
            elapsed
        ),

        "monte_carlo_samples_requested": (
            samples
        ),

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

    return (
        metrics,
        basin_rows,
    )


def analyze_synchronous_dynamics(
    model: BNetModel,
    network: str,
    method: str,
    variant: str,
    exact_dimension: int = 20,
    monte_carlo_samples: int = 10_000,
    base_seed: int = 2026,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """
    Select exact enumeration or Monte Carlo from the number of free variables.
    """
    if exact_dimension < 0:
        raise ValueError(
            "Exact dimension threshold cannot be negative."
        )

    free_variables = (
        count_free_variables(
            model
        )
    )

    if free_variables <= exact_dimension:
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
