from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import sympy as sp
from sympy.logic.boolalg import And, Not, Or, Xor


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

    local_dict = {symbol.name: symbol for symbol in symbols.values()}
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
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() != ".bnet":
        raise ValueError(f"Input file must be a .bnet file: {path}")

    raw_rules: list[tuple[str, str]] = []
    has_header = False

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
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
                f"Invalid .bnet line {line_number}: {raw_line}"
            )

        node, rule = line.split(",", 1)
        node = node.strip()
        rule = rule.strip()

        if not node or not rule:
            raise ValueError(
                f"Invalid .bnet line {line_number}: {raw_line}"
            )

        raw_rules.append((node, rule))

    nodes = [node for node, _ in raw_rules]

    if len(nodes) != len(set(nodes)):
        raise ValueError(
            "Duplicated target variables were found in the .bnet file."
        )

    symbols = {
        node: sp.Symbol(f"_x{i}")
        for i, node in enumerate(nodes)
    }

    rules = {
        node: _parse_rule(rule, symbols)
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

        return f"!({_expr_to_bnet(arg, symbol_names)})"

    if isinstance(expr, And):
        return "(" + " & ".join(
            _expr_to_bnet(arg, symbol_names)
            for arg in expr.args
        ) + ")"

    if isinstance(expr, Or):
        return "(" + " | ".join(
            _expr_to_bnet(arg, symbol_names)
            for arg in expr.args
        ) + ")"

    raise TypeError(
        f"Unsupported Boolean expression: {expr!r}"
    )


def write_bnet(
    path: str | Path,
    model: BNetModel,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    symbol_names = {
        symbol: node
        for node, symbol in model.symbols.items()
    }

    lines = []

    if model.has_header:
        lines.append("targets, factors")

    for node in model.nodes:
        lines.append(
            f"{node}, "
            f"{_expr_to_bnet(model.rules[node], symbol_names)}"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Boolean-function utilities
# -----------------------------------------------------------------------------

def simplify_rule(expr: sp.Basic) -> sp.Basic:
    """
    Simplify a Boolean rule after substitutions.

    The reduction described by Naldi et al. updates the logical functions of
    the targets of every removed component. Simplification is useful because
    substitutions can make some regulatory dependencies ineffective.
    """
    return sp.simplify_logic(expr, force=False)


def effectively_depends_on(
    expr: sp.Basic,
    symbol: sp.Symbol,
) -> bool:
    """
    Return True iff the Boolean function effectively depends on `symbol`.

    This follows the notion of effective interaction used by Naldi et al.:
    changing the value of the regulator must be able to change the target
    value for at least one assignment of the other variables.
    """
    if symbol not in expr.free_symbols:
        return False

    expr_0 = expr.subs(symbol, sp.false)
    expr_1 = expr.subs(symbol, sp.true)

    difference = Xor(expr_0, expr_1)
    witness = sp.satisfiable(difference, all_models=False)

    return witness is not False


def is_autoregulated(
    model: BNetModel,
    node: str,
) -> bool:
    return effectively_depends_on(
        model.rules[node],
        model.symbols[node],
    )


# -----------------------------------------------------------------------------
# Naldi et al. node elimination
# -----------------------------------------------------------------------------

def removable_nodes(model: BNetModel) -> list[str]:
    """
    Nodes eligible for a one-node reduction.

    In the Boolean case, a node can be removed when it is not effectively
    autoregulated. Effective Boolean autoregulations are functional, so the
    restriction agrees with the reduction rule of Naldi et al. (2011).
    """
    return [
        node
        for node in model.nodes
        if not is_autoregulated(model, node)
    ]


def eliminate_node(
    model: BNetModel,
    node: str,
) -> BNetModel:
    """
    Remove one non-autoregulated component.

    For every surviving target i, the reduced Boolean function is obtained by
    replacing the removed variable r with its target function K_r:

        K_i^r(z) = K_i(s_r(z))

    which, in the Boolean setting, is exactly the symbolic substitution

        r <- K_r.

    The removed node must not be effectively autoregulated.
    """
    if node not in model.rules:
        raise KeyError(f"Unknown node: {node}")

    if is_autoregulated(model, node):
        raise ValueError(
            f"Node {node!r} is effectively autoregulated and "
            "cannot be removed by the Naldi reduction."
        )

    removed_symbol = model.symbols[node]
    removed_rule = model.rules[node]

    new_nodes = [
        current
        for current in model.nodes
        if current != node
    ]

    new_rules: dict[str, sp.Basic] = {}

    for target in new_nodes:
        rule = model.rules[target]

        if removed_symbol in rule.free_symbols:
            rule = rule.xreplace(
                {removed_symbol: removed_rule}
            )
            rule = simplify_rule(rule)

        new_rules[target] = rule

    new_symbols = {
        current: model.symbols[current]
        for current in new_nodes
    }

    return BNetModel(
        nodes=new_nodes,
        rules=new_rules,
        symbols=new_symbols,
        has_header=model.has_header,
    )


def reduce_node_elimination(
    model: BNetModel,
) -> tuple[BNetModel, list[str], int]:
    """
    Iteratively apply one-node reductions until no removable node remains.

    The scan follows the current .bnet node order and restarts after every
    successful elimination. This makes the implementation deterministic.

    Important:
    Naldi et al. note that, for multiple reductions, the order may affect
    which nodes can subsequently be removed. Therefore this procedure
    computes a deterministic maximal reduction, not a guaranteed globally
    minimum-size reduction for arbitrary networks.
    """
    reduced = model
    removal_order: list[str] = []
    iterations = 0

    while True:
        iterations += 1

        candidate = None

        for node in reduced.nodes:
            if not is_autoregulated(reduced, node):
                candidate = node
                break

        if candidate is None:
            break

        reduced = eliminate_node(
            reduced,
            candidate,
        )

        removal_order.append(candidate)

    return reduced, removal_order, iterations


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def generate_node_elimination_model(
    input_bnet: str | Path,
    output_bnet: str | Path,
) -> dict[str, object]:
    input_bnet = Path(input_bnet)
    output_bnet = Path(output_bnet)

    original = load_bnet(input_bnet)

    reduced, removal_order, iterations = reduce_node_elimination(
        original
    )

    write_bnet(
        output_bnet,
        reduced,
    )

    return {
        "method": "node_elimination",
        "input_file": str(input_bnet),
        "output_file": str(output_bnet),
        "original_size": len(original.nodes),
        "reduced_size": len(reduced.nodes),
        "removed_nodes": len(removal_order),
        "iterations": iterations,
        "removal_order": removal_order,
        "retained_nodes": reduced.nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the dynamically consistent node-elimination reduction "
            "of Naldi et al. (2011) to a Boolean .bnet model."
        )
    )

    parser.add_argument(
        "input",
        help="Path to the original .bnet file.",
    )

    parser.add_argument(
        "output",
        help="Path where the reduced .bnet file will be written.",
    )

    args = parser.parse_args()

    result = generate_node_elimination_model(
        args.input,
        args.output,
    )

    print(
        f"Original size: {result['original_size']}"
    )
    print(
        f"Reduced size: {result['reduced_size']}"
    )
    print(
        f"Removed nodes: {result['removed_nodes']}"
    )
    print(
        f"Iterations: {result['iterations']}"
    )

    if result["removal_order"]:
        print(
            "Removal order: "
            + " -> ".join(result["removal_order"])
        )
    else:
        print(
            "No node could be removed."
        )

    print(
        "Retained nodes: "
        + ", ".join(result["retained_nodes"])
    )

    print(
        f"Reduced model written to: {result['output_file']}"
    )


if __name__ == "__main__":
    main()
