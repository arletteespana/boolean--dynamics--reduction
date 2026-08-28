from __future__ import annotations

import argparse
import csv
import itertools
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import networkx as nx
import sympy as sp
from sympy.logic.boolalg import And, Not, Or


@dataclass
class BNetModel:
    nodes: list[str]
    rules: dict[str, sp.Basic]
    symbols: dict[str, sp.Symbol]
    has_header: bool = False


# -----------------------------------------------------------------------------
# .bnet I/O
# -----------------------------------------------------------------------------

def _tokenize_rule(rule: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(rule):
        ch = rule[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()!~&|":
            tokens.append(ch)
            i += 1
            continue

        j = i
        while j < len(rule) and (not rule[j].isspace()) and rule[j] not in "()!~&|":
            j += 1
        tokens.append(rule[i:j])
        i = j
    return tokens


def _parse_rule(rule: str, symbols: dict[str, sp.Symbol]) -> sp.Basic:
    converted = []

    for token in _tokenize_rule(rule):
        low = token.lower()

        if token in ("!", "~") or low == "not":
            converted.append("~")
        elif token == "&" or low == "and":
            converted.append("&")
        elif token == "|" or low == "or":
            converted.append("|")
        elif token == "(":
            converted.append("(")
        elif token == ")":
            converted.append(")")
        elif low in ("1", "true"):
            converted.append("_TRUE")
        elif low in ("0", "false"):
            converted.append("_FALSE")
        elif token in symbols:
            converted.append(symbols[token].name)
        else:
            raise ValueError(
                f"Unknown token in Boolean rule: {token!r}\n"
                f"Rule: {rule}"
            )

    local_dict = {
        symbol.name: symbol
        for symbol in symbols.values()
    }

    local_dict["_TRUE"] = sp.true
    local_dict["_FALSE"] = sp.false

    return sp.sympify(
        " ".join(converted),
        locals=local_dict,
        evaluate=False,
    )


def load_bnet(path: str | Path) -> BNetModel:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    if path.suffix.lower() != ".bnet":
        raise ValueError(
            f"Input file must be a .bnet file: {path}"
        )

    raw_rules: list[tuple[str, str]] = []
    has_header = False

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.lower().replace(" ", "") in {
            "targets,factors",
            "target,factor",
        }:
            has_header = True
            continue

        if "," not in line:
            raise ValueError(
                f"Invalid .bnet line {line_number}: "
                f"{raw_line}"
            )

        node, rule = line.split(",", 1)

        node = node.strip()
        rule = rule.strip()

        if not node or not rule:
            raise ValueError(
                f"Invalid .bnet line {line_number}: "
                f"{raw_line}"
            )

        raw_rules.append(
            (node, rule)
        )

    nodes = [
        node
        for node, _ in raw_rules
    ]

    if len(nodes) != len(set(nodes)):
        raise ValueError(
            "Duplicated target variables were found "
            "in the .bnet file."
        )

    symbols = {
        node: sp.Symbol(f"_x{i}")
        for i, node in enumerate(nodes)
    }

    rules = {
        node: _parse_rule(
            rule,
            symbols,
        )
        for node, rule in raw_rules
    }

    return BNetModel(
        nodes=nodes,
        rules=rules,
        symbols=symbols,
        has_header=has_header,
    )


def _expr_to_bnet(
    expr: sp.Basic,
    symbol_names: dict[sp.Symbol, str],
) -> str:

    if expr is sp.true or expr == True:
        return "1"

    if expr is sp.false or expr == False:
        return "0"

    if isinstance(expr, sp.Symbol):
        return symbol_names[expr]

    if isinstance(expr, Not):
        arg = expr.args[0]

        if isinstance(arg, sp.Symbol):
            return f"!{_expr_to_bnet(arg, symbol_names)}"

        return (
            f"!({_expr_to_bnet(arg, symbol_names)})"
        )

    if isinstance(expr, And):
        return (
            "("
            + " & ".join(
                _expr_to_bnet(
                    arg,
                    symbol_names,
                )
                for arg in expr.args
            )
            + ")"
        )

    if isinstance(expr, Or):
        return (
            "("
            + " | ".join(
                _expr_to_bnet(
                    arg,
                    symbol_names,
                )
                for arg in expr.args
            )
            + ")"
        )

    raise TypeError(
        f"Unsupported Boolean expression: {expr!r}"
    )


def write_bnet(
    path: str | Path,
    output_rules: list[tuple[str, sp.Basic]],
    symbol_names: dict[sp.Symbol, str],
    has_header: bool,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = []

    if has_header:
        lines.append(
            "targets, factors"
        )

    for node, expr in output_rules:
        lines.append(
            f"{node}, "
            f"{_expr_to_bnet(expr, symbol_names)}"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Interaction graph
# -----------------------------------------------------------------------------

def interaction_graph(
    model: BNetModel,
) -> nx.DiGraph:

    graph = nx.DiGraph()

    graph.add_nodes_from(
        model.nodes
    )

    symbol_to_node = {
        symbol: node
        for node, symbol
        in model.symbols.items()
    }

    for target in model.nodes:
        for symbol in model.rules[
            target
        ].free_symbols:

            regulator = symbol_to_node[
                symbol
            ]

            graph.add_edge(
                regulator,
                target,
            )

    return graph


# -----------------------------------------------------------------------------
# Exact minimum-cardinality dominant sets
# -----------------------------------------------------------------------------

def _cycle_nodes(
    graph: nx.DiGraph,
) -> list[str] | None:

    try:
        edges = nx.find_cycle(
            graph,
            orientation="original",
        )

    except nx.NetworkXNoCycle:
        return None

    nodes = []
    seen = set()

    for edge in edges:
        u, v = edge[0], edge[1]

        if u not in seen:
            nodes.append(u)
            seen.add(u)

        if v not in seen:
            nodes.append(v)
            seen.add(v)

    return nodes


def _disjoint_cycle_lower_bound(
    graph: nx.DiGraph,
) -> int:

    work = graph.copy()
    count = 0

    while True:
        cycle = _cycle_nodes(work)

        if cycle is None:
            return count

        count += 1

        work.remove_nodes_from(
            cycle
        )


def _minimum_fvs_component(
    graph: nx.DiGraph,
) -> list[frozenset[str]]:

    best_size = float("inf")
    best_sets: set[
        frozenset[str]
    ] = set()

    def search(
        work: nx.DiGraph,
        chosen: frozenset[str],
    ) -> None:

        nonlocal best_size
        nonlocal best_sets

        if len(chosen) > best_size:
            return

        cycle = _cycle_nodes(work)

        if cycle is None:

            if len(chosen) < best_size:
                best_size = len(chosen)
                best_sets = {chosen}

            elif len(chosen) == best_size:
                best_sets.add(chosen)

            return

        lower_bound = (
            _disjoint_cycle_lower_bound(
                work
            )
        )

        if (
            len(chosen)
            + lower_bound
            > best_size
        ):
            return

        cycle = sorted(
            cycle,
            key=lambda node: (
                work.in_degree(node)
                + work.out_degree(node)
            ),
            reverse=True,
        )

        for node in cycle:

            child = work.copy()
            child.remove_node(node)

            search(
                child,
                chosen
                | frozenset([node]),
            )

    search(
        graph.copy(),
        frozenset(),
    )

    return sorted(
        best_sets,
        key=lambda s: tuple(
            sorted(s)
        ),
    )


def minimum_dominant_sets(
    graph: nx.DiGraph,
) -> list[tuple[str, ...]]:

    # Vertices without incoming regulators are mandatory.
    sources = {
        node
        for node in graph.nodes
        if graph.in_degree(node) == 0
    }

    # Every self-loop must also be hit.
    self_loops = {
        node
        for node in graph.nodes
        if graph.has_edge(
            node,
            node,
        )
    }

    forced = (
        sources
        | self_loops
    )

    reduced = graph.copy()

    reduced.remove_nodes_from(
        forced
    )

    cyclic_components = []

    for component in (
        nx.strongly_connected_components(
            reduced
        )
    ):
        subgraph = reduced.subgraph(
            component
        ).copy()

        if (
            len(component) > 1
            and not nx.is_directed_acyclic_graph(
                subgraph
            )
        ):
            cyclic_components.append(
                subgraph
            )

    component_solutions: list[
        list[frozenset[str]]
    ] = []

    for component in cyclic_components:
        component_solutions.append(
            _minimum_fvs_component(
                component
            )
        )

    if not component_solutions:
        return [
            tuple(
                sorted(forced)
            )
        ]

    results = set()

    for combination in itertools.product(
        *component_solutions
    ):

        combined = set(
            forced
        )

        for solution in combination:
            combined.update(
                solution
            )

        results.add(
            tuple(
                sorted(combined)
            )
        )

    return sorted(
        results
    )


# -----------------------------------------------------------------------------
# Dominant-set structural quantities
# -----------------------------------------------------------------------------

def dominant_depth(
    graph: nx.DiGraph,
    dominant_set: set[str],
) -> int:

    known = set(
        dominant_set
    )

    depth = 0

    while (
        len(known)
        < graph.number_of_nodes()
    ):

        new_nodes = {
            node
            for node in graph.nodes
            if (
                node not in known
                and set(
                    graph.predecessors(
                        node
                    )
                ).issubset(
                    known
                )
            )
        }

        if not new_nodes:
            raise ValueError(
                "The supplied set is not dominant "
                "for this interaction graph."
            )

        known.update(
            new_nodes
        )

        depth += 1

    return depth


def recurrence_length(
    graph: nx.DiGraph,
    dominant_set: set[str],
) -> int:

    U = set(
        dominant_set
    )

    non_u = [
        node
        for node in graph.nodes
        if node not in U
    ]

    dag = graph.subgraph(
        non_u
    ).copy()

    if not nx.is_directed_acyclic_graph(
        dag
    ):
        raise ValueError(
            "The supplied set is not dominant: "
            "G - U contains a directed cycle."
        )

    topo = list(
        nx.topological_sort(
            dag
        )
    )

    ell = 1

    for source in U:

        # Direct U -> U edges,
        # including self-loops.
        for target in graph.successors(
            source
        ):
            if target in U:
                ell = max(
                    ell,
                    1,
                )

        dist: dict[
            str,
            int,
        ] = {}

        for target in graph.successors(
            source
        ):

            if target not in U:
                dist[target] = max(
                    dist.get(
                        target,
                        0,
                    ),
                    1,
                )

        for node in topo:

            if node not in dist:
                continue

            current = dist[
                node
            ]

            for target in graph.successors(
                node
            ):

                candidate = (
                    current + 1
                )

                if target in U:
                    ell = max(
                        ell,
                        candidate,
                    )

                else:
                    dist[target] = max(
                        dist.get(
                            target,
                            0,
                        ),
                        candidate,
                    )

    return ell


# -----------------------------------------------------------------------------
# Induced Boolean realization on |U| * ell binary memory coordinates
# -----------------------------------------------------------------------------

def build_induced_bnet(
    model: BNetModel,
    graph: nx.DiGraph,
    dominant_set: tuple[str, ...],
) -> tuple[
    list[tuple[str, sp.Basic]],
    dict[sp.Symbol, str],
    int,
    int,
]:

    U = set(
        dominant_set
    )

    ell = recurrence_length(
        graph,
        U,
    )

    depth = dominant_depth(
        graph,
        U,
    )

    # __t0 is the most recent stored value of a dominant variable,
    # __t1 is one step older, etc.
    memory_symbols: dict[
        tuple[str, int],
        sp.Symbol,
    ] = {}

    symbol_names: dict[
        sp.Symbol,
        str,
    ] = {}

    def output_name(
        node: str,
        age: int,
    ) -> str:

        if ell == 1:
            return node

        return (
            f"{node}__t{age}"
        )

    for node in dominant_set:

        for absolute_time in range(
            ell
        ):

            age = (
                ell
                - 1
                - absolute_time
            )

            symbol = sp.Symbol(
                f"_m{len(memory_symbols)}"
            )

            memory_symbols[
                (
                    node,
                    absolute_time,
                )
            ] = symbol

            symbol_names[
                symbol
            ] = output_name(
                node,
                age,
            )

    @lru_cache(maxsize=None)
    def expression_at(
        node: str,
        absolute_time: int,
    ) -> sp.Basic:

        if node in U:

            if (
                0
                <= absolute_time
                < ell
            ):
                return memory_symbols[
                    (
                        node,
                        absolute_time,
                    )
                ]

            raise ValueError(
                "Insufficient dominant-state history while "
                "constructing the induced dynamics. "
                f"Reached dominant variable {node!r} "
                "before the available history."
            )

        rule = model.rules[
            node
        ]

        # Constant rules do not require any state history.
        if not rule.free_symbols:
            return rule

        substitutions = {}

        for regulator in graph.predecessors(
            node
        ):

            substitutions[
                model.symbols[
                    regulator
                ]
            ] = expression_at(
                regulator,
                absolute_time - 1,
            )

        return rule.xreplace(
            substitutions
        )

    output_rules: list[
        tuple[str, sp.Basic]
    ] = []

    for node in dominant_set:

        substitutions = {}

        for regulator in graph.predecessors(
            node
        ):

            substitutions[
                model.symbols[
                    regulator
                ]
            ] = expression_at(
                regulator,
                ell - 1,
            )

        next_expr = model.rules[
            node
        ].xreplace(
            substitutions
        )

        unknown = (
            next_expr.free_symbols
            .difference(
                symbol_names
            )
        )

        if unknown:
            raise ValueError(
                f"The induced rule for "
                f"{node!r} still contains "
                "non-memory symbols: "
                f"{sorted(map(str, unknown))}"
            )

        output_rules.append(
            (
                output_name(
                    node,
                    0,
                ),
                next_expr,
            )
        )

    if ell > 1:

        for node in dominant_set:

            for age in range(
                1,
                ell,
            ):

                newer_absolute_time = (
                    ell - age
                )

                newer_symbol = (
                    memory_symbols[
                        (
                            node,
                            newer_absolute_time,
                        )
                    ]
                )

                output_rules.append(
                    (
                        output_name(
                            node,
                            age,
                        ),
                        newer_symbol,
                    )
                )

    return (
        output_rules,
        symbol_names,
        depth,
        ell,
    )


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def generate_dominant_vertex_models(
    input_bnet: str | Path,
    output_dir: str | Path,
) -> list[dict[str, object]]:

    input_bnet = Path(
        input_bnet
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_bnet(
        input_bnet
    )

    graph = interaction_graph(
        model
    )

    dominant_sets = (
        minimum_dominant_sets(
            graph
        )
    )

    rows = []

    for index, dominant_set in enumerate(
        dominant_sets,
        1,
    ):

        (
            rules,
            symbol_names,
            depth,
            ell,
        ) = build_induced_bnet(
            model,
            graph,
            dominant_set,
        )

        filename = (
            f"dominant_vertices_"
            f"{index:02d}.bnet"
        )

        output_path = (
            output_dir
            / filename
        )

        write_bnet(
            output_path,
            output_rules=rules,
            symbol_names=symbol_names,
            has_header=model.has_header,
        )

        rows.append(
            {
                "id": index,
                "file": filename,
                "dominant_set_size": len(
                    dominant_set
                ),
                "dominant_set": ";".join(
                    dominant_set
                ),
                "depth": depth,
                "recurrence_length": ell,
                "state_dimension": (
                    len(dominant_set)
                    * ell
                ),
            }
        )

    summary_path = (
        output_dir
        / "dominant_vertices_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "file",
                "dominant_set_size",
                "dominant_set",
                "depth",
                "recurrence_length",
                "state_dimension",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return rows


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Find all minimum-cardinality "
            "dominant sets and generate one "
            "induced .bnet model for each "
            "of them."
        )
    )

    parser.add_argument(
        "input",
        help=(
            "Path to the original "
            ".bnet file."
        ),
    )

    parser.add_argument(
        "output_dir",
        help=(
            "Directory where induced "
            ".bnet files will be written."
        ),
    )

    args = parser.parse_args()

    rows = (
        generate_dominant_vertex_models(
            args.input,
            args.output_dir,
        )
    )

    if not rows:
        print(
            "No dominant-set models "
            "were generated."
        )
        return

    print(
        "Minimum dominant-set size: "
        f"{rows[0]['dominant_set_size']}"
    )

    print(
        "Number of minimum dominant sets: "
        f"{len(rows)}"
    )

    for row in rows:

        print(
            f"DV {row['id']:02d}: "
            f"|U|="
            f"{row['dominant_set_size']}, "
            f"d={row['depth']}, "
            f"ell="
            f"{row['recurrence_length']}, "
            f"dimension="
            f"{row['state_dimension']} "
            f"-> {row['file']}"
        )


if __name__ == "__main__":
    main()
