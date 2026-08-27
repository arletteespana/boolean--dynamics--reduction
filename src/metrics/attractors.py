from __future__ import annotations

from time import perf_counter

import sympy as sp
from sympy.logic.boolalg import Xor
from sympy.logic.inference import satisfiable

from .utils import ModelRecord, BNetModel, load_bnet


def count_fixed_points(model: BNetModel) -> int:
    """
    Count fixed points exactly using Boolean satisfiability.

    A fixed point satisfies x_i = f_i(x) for every variable i.
    SAT models may omit variables that are unconstrained by the resulting
    formula; each omitted variable contributes a factor of two.
    """
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


def _compile_synchronous_update(model: BNetModel):
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

    def successor(state: int) -> int:
        values = [
            bool((state >> i) & 1)
            for i in range(n)
        ]

        updated = update_function(*values)

        result = 0

        for i, value in enumerate(updated):
            if bool(value):
                result |= 1 << i

        return result

    return successor


def enumerate_synchronous_attractors(
    model: BNetModel,
) -> dict[str, object]:
    """
    Enumerate every attractor of the synchronous deterministic dynamics.

    This does not explicitly build a NetworkX STG. It traverses the functional
    graph directly, which is more memory efficient but still requires visiting
    all 2^n states.
    """
    n = len(model.nodes)
    total_states = 1 << n
    successor = _compile_synchronous_update(model)

    done = bytearray(total_states)
    periods: list[int] = []

    for start in range(total_states):
        if done[start]:
            continue

        path: list[int] = []
        local_position: dict[int, int] = {}
        current = start

        while not done[current] and current not in local_position:
            local_position[current] = len(path)
            path.append(current)
            current = successor(current)

        if current in local_position:
            cycle_start = local_position[current]
            periods.append(
                len(path) - cycle_start
            )

        for state in path:
            done[state] = 1

    periods.sort()

    fixed_points = sum(
        1
        for period in periods
        if period == 1
    )

    periodic_periods = [
        period
        for period in periods
        if period > 1
    ]

    return {
        "number_of_attractors": len(periods),
        "fixed_points_from_enumeration": fixed_points,
        "periodic_attractors": len(periodic_periods),
        "all_attractor_periods": periods,
        "periodic_attractor_periods": periodic_periods,
    }


def analyze_attractors(
    record: ModelRecord,
    max_exact_dimension: int = 20,
) -> dict[str, object]:
    model = load_bnet(record.path)
    dimension = len(model.nodes)

    fixed_start = perf_counter()
    fixed_points = count_fixed_points(model)
    fixed_time = perf_counter() - fixed_start

    row: dict[str, object] = {
        "network": record.network,
        "method": record.method,
        "variant": record.variant,
        "model_file": str(record.path),
        "state_dimension": dimension,
        "fixed_points": fixed_points,
        "fixed_point_time_seconds": fixed_time,
        "synchronous_analysis_status": "",
        "number_of_attractors": "",
        "periodic_attractors": "",
        "all_attractor_periods": "",
        "periodic_attractor_periods": "",
        "synchronous_attractor_time_seconds": "",
        "fixed_point_enumeration_check": "",
    }

    if dimension > max_exact_dimension:
        row["synchronous_analysis_status"] = (
            f"skipped: dimension>{max_exact_dimension}"
        )
        return row

    sync_start = perf_counter()
    dynamics = enumerate_synchronous_attractors(
        model
    )
    sync_time = perf_counter() - sync_start

    row.update(
        {
            "synchronous_analysis_status": "computed",
            "number_of_attractors": dynamics[
                "number_of_attractors"
            ],
            "periodic_attractors": dynamics[
                "periodic_attractors"
            ],
            "all_attractor_periods": ";".join(
                str(value)
                for value in dynamics[
                    "all_attractor_periods"
                ]
            ),
            "periodic_attractor_periods": ";".join(
                str(value)
                for value in dynamics[
                    "periodic_attractor_periods"
                ]
            ),
            "synchronous_attractor_time_seconds": (
                sync_time
            ),
            "fixed_point_enumeration_check": (
                dynamics[
                    "fixed_points_from_enumeration"
                ]
                == fixed_points
            ),
        }
    )

    return row
