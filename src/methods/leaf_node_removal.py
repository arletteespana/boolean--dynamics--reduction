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
# Effective regulatory dependencies
# -----------------------------------------------------------------------------

def effectively_depends_on(
    expr: sp.Basic,
    symbol: sp.Symbol,
) -> bool:
    """
    Return True iff changing `symbol` can change the Boolean function for
    at least one assignment of the remaining variables.
    """
    if symbol not in expr.free_symbols:
        return False

    expr_0 = expr.subs(symbol, sp.false)
    expr_1 = expr.subs(symbol, sp.true)

    difference = Xor(expr_0, expr_1)
    witness = sp.satisfiable(difference, all_models=False)

    return witness is not False


def targets(
    model: BNetModel,
    node: str,
) -> list[str]:
    """
    Return the effective targets of `node`.

    A self-loop counts as an outgoing interaction. Therefore a node whose
    only target is itself is not a leaf.
    """
    symbol = model.symbols[node]

    return [
        target
        for target in model.nodes
        if effectively_depends_on(
            model.rules[target],
            symbol,
        )
    ]


def leaf_nodes(model: BNetModel) -> list[str]:
    """
    Return all nodes with effective out-degree equal to zero.
    """
    return [
        node
        for node in model.nodes
        if len(targets(model, node)) == 0
    ]


# -----------------------------------------------------------------------------
# Leaf-node removal
# -----------------------------------------------------------------------------

def remove_leaf_nodes_once(
    model: BNetModel,
) -> tuple[BNetModel, list[str]]:
    """
    Remove all current leaf nodes in one round.

    Since a leaf has no effective targets, none of the surviving Boolean
    functions depends on it. Therefore no substitution or rule rewriting
    is required.
    """
    leaves = leaf_nodes(model)

    if not leaves:
        return model, []

    leaf_set = set(leaves)

    new_nodes = [
        node
        for node in model.nodes
        if node not in leaf_set
    ]

    new_rules = {
        node: model.rules[node]
        for node in new_nodes
    }

    new_symbols = {
        node: model.symbols[node]
        for node in new_nodes
    }

    reduced = BNetModel(
        nodes=new_nodes,
        rules=new_rules,
        symbols=new_symbols,
        has_header=model.has_header,
    )

    return reduced, leaves


def reduce_leaf_nodes(
    model: BNetModel,
) -> tuple[BNetModel, list[list[str]]]:
    """
    Iteratively remove zero-out-degree nodes until no leaf remains.

    Leaves are removed in layers. Removing one layer can expose a new set
    of leaves in the remaining network.
    """
    reduced = model
    layers: list[list[str]] = []

    while True:
        reduced, removed = remove_leaf_nodes_once(
            reduced
        )

        if not removed:
            break

        layers.append(removed)

    return reduced, layers


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def generate_leaf_node_removal_model(
    input_bnet: str | Path,
    output_bnet: str | Path,
) -> dict[str, object]:
    input_bnet = Path(input_bnet)
    output_bnet = Path(output_bnet)

    original = load_bnet(input_bnet)

    reduced, layers = reduce_leaf_nodes(
        original
    )

    write_bnet(
        output_bnet,
        reduced,
    )

    removed_nodes = [
        node
        for layer in layers
        for node in layer
    ]

    return {
        "method": "leaf_node_removal",
        "input_file": str(input_bnet),
        "output_file": str(output_bnet),
        "original_size": len(original.nodes),
        "reduced_size": len(reduced.nodes),
        "removed_nodes": len(removed_nodes),
        "removal_rounds": len(layers),
        "removal_layers": layers,
        "removal_order": removed_nodes,
        "retained_nodes": reduced.nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Iteratively remove leaf nodes (nodes with effective "
            "out-degree zero) from a Boolean .bnet model."
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

    result = generate_leaf_node_removal_model(
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
        f"Removal rounds: {result['removal_rounds']}"
    )

    layers = result["removal_layers"]

    if layers:
        print("Removal layers:")

        for i, layer in enumerate(layers, 1):
            print(
                f"  Round {i}: "
                + ", ".join(layer)
            )
    else:
        print("No leaf nodes were found.")

    print(
        "Retained nodes: "
        + ", ".join(result["retained_nodes"])
    )

    print(
        f"Reduced model written to: {result['output_file']}"
    )


if __name__ == "__main__":
    main()
