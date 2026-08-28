from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import mpbn
import sympy as sp

from .utils import BNetModel, ModelRecord, load_bnet


def count_fixed_points(
    path: str | Path,
) -> int:
    """
    Count fixed points exactly using mpbn.

    A fixed point satisfies x_i = f_i(x) for every variable i.
    Fixed points are properties of the Boolean map itself and therefore
    do not depend on the update scheme.
    """
    model = mpbn.MPBooleanNetwork(
        str(path)
    )

    return int(
        model.count_fixedpoints()
    )


def compile_synchronous_update(model: BNetModel):
    """
    Compile the synchronous Boolean update map.

    States are represented as integers. Bit i stores the state of the i-th
    variable in model.nodes.
    """
    symbols = [
        model.symbols[node]
        for node in model.nodes
    ]

    expressions = [
        model.rules[node]
        for node in model.nodes
    ]

    update_function = sp.lambdify(
        symbols,
        expressions,
        modules="math",
    )

    n = len(symbols)

    if n == 0:
        def successor(state: int) -> int:
            return 0

        return successor

    def successor(state: int) -> int:
        values = [
            bool((state >> i) & 1)
            for i in range(n)
        ]

        updated = update_function(*values)

        if n == 1 and not isinstance(
            updated,
            (list, tuple),
        ):
            updated = [updated]

        result = 0

        for i, value in enumerate(updated):
            if bool(value):
                result |= 1 << i

        return result

    return successor


def canonical_cycle(cycle: list[int]) -> tuple[int, ...]:
    """
    Return a rotation-independent representation of a deterministic cycle.
    """
    if not cycle:
        raise ValueError("A cycle cannot be empty.")

    if len(cycle) == 1:
        return (cycle[0],)

    start = min(
        range(len(cycle)),
        key=cycle.__getitem__,
    )

    return tuple(
        cycle[start:] + cycle[:start]
    )


def attractor_signature(
    cycle: tuple[int, ...],
) -> str:
    """
    Stable compact identifier for an attractor cycle.
    """
    payload = ",".join(
        str(state)
        for state in cycle
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()[:16]


def analyze_fixed_points(
    record: ModelRecord,
) -> dict[str, object]:
    """
    Compute the exact fixed-point count of a Boolean-network model.

    mpbn is used only for the fixed-point computation. The synchronous
    attractor, basin, and transient analyses are performed separately.
    """
    model = load_bnet(
        record.path
    )

    start = perf_counter()

    fixed_points = count_fixed_points(
        record.path
    )

    elapsed = (
        perf_counter() - start
    )

    return {
        "network": record.network,
        "method": record.method,
        "variant": record.variant,
        "model_file": str(
            record.path
        ),
        "state_dimension": len(
            model.nodes
        ),
        "fixed_points": fixed_points,
        "fixed_point_time_seconds": elapsed,
    }
