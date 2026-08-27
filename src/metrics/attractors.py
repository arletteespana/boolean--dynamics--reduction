from __future__ import annotations

import hashlib
from time import perf_counter

import sympy as sp
from sympy.logic.boolalg import Xor
from sympy.logic.inference import satisfiable

from .utils import BNetModel, ModelRecord, load_bnet


def count_fixed_points(model: BNetModel) -> int:
    """
    Count fixed points exactly using Boolean satisfiability.

    A fixed point satisfies x_i = f_i(x) for every variable i.
    SAT models may omit variables that are unconstrained by the resulting
    formula; each omitted variable contributes a factor of two.
    """
    if not model.nodes:
        return 1

    variables = [
        model.symbols[node]
        for node in model.nodes
    ]

    constraints = [
        ~Xor(
            model.symbols[node],
            model.rules[node],
        )
        for node in model.nodes
    ]

    formula = sp.And(*constraints)
    total = 0

    for solution in satisfiable(
        formula,
        all_models=True,
    ):
        if solution is False:
            return 0

        assigned = sum(
            1
            for variable in variables
            if variable in solution
        )

        total += 2 ** (
            len(variables) - assigned
        )

    return total


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

    start = min(range(len(cycle)), key=cycle.__getitem__)

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

    return hashlib.sha256(payload).hexdigest()[:16]


def analyze_fixed_points(
    record: ModelRecord,
) -> dict[str, object]:
    model = load_bnet(record.path)

    start = perf_counter()
    fixed_points = count_fixed_points(model)
    elapsed = perf_counter() - start

    return {
        "network": record.network,
        "method": record.method,
        "variant": record.variant,
        "model_file": str(record.path),
        "state_dimension": len(model.nodes),
        "fixed_points": fixed_points,
        "fixed_point_time_seconds": elapsed,
    }
