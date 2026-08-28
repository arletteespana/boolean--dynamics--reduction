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
    return sp.simplify_logic(expr, force=False)


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


def constant_value(expr: sp.Basic) -> int | None:
    """
    Return 0 or 1 if the Boolean function is constant, otherwise None.
    """
    simplified = simplify_rule(expr)

    if simplified is sp.true or simplified == True:
        return 1

    if simplified is sp.false or simplified == False:
        return 0

    return None


def regulators(
    model: BNetModel,
    node: str,
) -> list[str]:
    expr = model.rules[node]

    return [
        regulator
        for regulator in model.nodes
        if effectively_depends_on(
            expr,
            model.symbols[regulator],
        )
    ]


def targets(
    model: BNetModel,
    node: str,
) -> list[str]:
    symbol = model.symbols[node]

    return [
        target
        for target in model.nodes
        if effectively_depends_on(
            model.rules[target],
            symbol,
        )
    ]


def _remove_node_from_model(
    model: BNetModel,
    node: str,
    new_rules: dict[str, sp.Basic],
) -> BNetModel:
    new_nodes = [
        current
        for current in model.nodes
        if current != node
    ]

    new_symbols = {
        current: model.symbols[current]
        for current in new_nodes
    }

    return BNetModel(
        nodes=new_nodes,
        rules={
            current: new_rules[current]
            for current in new_nodes
        },
        symbols=new_symbols,
        has_header=model.has_header,
    )


# -----------------------------------------------------------------------------
# Algorithm 1: stabilized nodes
# -----------------------------------------------------------------------------

def eliminate_stabilized_node(
    model: BNetModel,
    node: str,
) -> tuple[BNetModel, int]:
    """
    Eliminate one node whose Boolean function is constant.

    The constant value is inserted into every Boolean rule depending on the
    stabilized node, and the resulting rules are simplified.
    """
    value = constant_value(model.rules[node])

    if value is None:
        raise ValueError(
            f"Node {node!r} does not have a constant Boolean function."
        )

    symbol = model.symbols[node]
    replacement = sp.true if value == 1 else sp.false

    new_rules: dict[str, sp.Basic] = {}

    for target in model.nodes:
        if target == node:
            continue

        rule = model.rules[target]

        if effectively_depends_on(rule, symbol):
            rule = rule.xreplace(
                {symbol: replacement}
            )
            rule = simplify_rule(rule)

        new_rules[target] = rule

    reduced = _remove_node_from_model(
        model,
        node,
        new_rules,
    )

    return reduced, value


def reduce_stabilized_nodes(
    model: BNetModel,
) -> tuple[BNetModel, list[tuple[str, int]]]:
    """
    Algorithm 1 of Saadatpour et al. (2013).

    Repeatedly remove constant Boolean functions until no additional
    constant function is produced.
    """
    reduced = model
    removed: list[tuple[str, int]] = []

    while True:
        candidate = None

        for node in reduced.nodes:
            value = constant_value(reduced.rules[node])

            if value is not None:
                candidate = (node, value)
                break

        if candidate is None:
            break

        node, value = candidate

        reduced, _ = eliminate_stabilized_node(
            reduced,
            node,
        )

        removed.append((node, value))

    return reduced, removed


# -----------------------------------------------------------------------------
# Algorithm 2: simple mediator nodes
# -----------------------------------------------------------------------------

def simple_mediator_data(
    model: BNetModel,
    node: str,
) -> tuple[str, str] | None:
    """
    Check whether `node` can be removed according to Algorithm 2.

    We implement the strict rule stated in Saadatpour et al. (2013):

      * v has one effective regulator u;
      * v has one effective target w;
      * B_w depends only on v;
      * B_u does not depend on w;
      * B_w does not depend on u.

    The explicit one-target condition enforces the paper's description of v
    as a simple mediator with in-degree and out-degree equal to one.
    """
    regs_v = regulators(model, node)

    if len(regs_v) != 1:
        return None

    u = regs_v[0]

    if u == node:
        return None

    targets_v = targets(model, node)

    if len(targets_v) != 1:
        return None

    w = targets_v[0]

    if w == node or w == u:
        return None

    regs_w = regulators(model, w)

    if regs_w != [node]:
        return None

    if effectively_depends_on(
        model.rules[u],
        model.symbols[w],
    ):
        return None

    if effectively_depends_on(
        model.rules[w],
        model.symbols[u],
    ):
        return None

    return u, w


def eliminate_simple_mediator(
    model: BNetModel,
    node: str,
) -> tuple[BNetModel, str, str]:
    """
    Remove one simple mediator v in a chain u -> v -> w.

    The downstream rule is replaced according to

        B_w(v) -> B_w(B_v(u)).
    """
    data = simple_mediator_data(
        model,
        node,
    )

    if data is None:
        raise ValueError(
            f"Node {node!r} is not an eligible simple mediator."
        )

    u, w = data

    mediator_symbol = model.symbols[node]
    mediator_rule = model.rules[node]

    new_rules: dict[str, sp.Basic] = {}

    for target in model.nodes:
        if target == node:
            continue

        rule = model.rules[target]

        if target == w:
            rule = rule.xreplace(
                {mediator_symbol: mediator_rule}
            )
            rule = simplify_rule(rule)

        new_rules[target] = rule

    reduced = _remove_node_from_model(
        model,
        node,
        new_rules,
    )

    return reduced, u, w


def reduce_simple_mediators(
    model: BNetModel,
) -> tuple[BNetModel, list[tuple[str, str, str]]]:
    """
    Algorithm 2 of Saadatpour et al. (2013).

    Repeatedly merge eligible simple mediator nodes. The current node order
    is used to select the first eligible mediator, making the implementation
    deterministic.
    """
    reduced = model
    removed: list[tuple[str, str, str]] = []

    while True:
        candidate = None

        for node in reduced.nodes:
            data = simple_mediator_data(
                reduced,
                node,
            )

            if data is not None:
                candidate = (node, data[0], data[1])
                break

        if candidate is None:
            break

        node, u, w = candidate

        reduced, _, _ = eliminate_simple_mediator(
            reduced,
            node,
        )

        removed.append((node, u, w))

    return reduced, removed


# -----------------------------------------------------------------------------
# Complete two-step reduction
# -----------------------------------------------------------------------------

def reduce_two_step(
    model: BNetModel,
) -> tuple[
    BNetModel,
    list[tuple[str, int]],
    list[tuple[str, str, str]],
]:
    """
    Apply the two algorithms in the order presented in the paper:

      Step 1: eliminate stabilized nodes.
      Step 2: merge simple mediator nodes.
    """
    after_step_1, stabilized = reduce_stabilized_nodes(
        model
    )

    reduced, mediators = reduce_simple_mediators(
        after_step_1
    )

    return reduced, stabilized, mediators


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def generate_two_step_model(
    input_bnet: str | Path,
    output_bnet: str | Path,
) -> dict[str, object]:
    input_bnet = Path(input_bnet)
    output_bnet = Path(output_bnet)

    original = load_bnet(input_bnet)

    reduced, stabilized, mediators = reduce_two_step(
        original
    )

    write_bnet(
        output_bnet,
        reduced,
    )

    return {
        "method": "two_step",
        "input_file": str(input_bnet),
        "output_file": str(output_bnet),
        "original_size": len(original.nodes),
        "after_step_1_size": (
            len(original.nodes) - len(stabilized)
        ),
        "reduced_size": len(reduced.nodes),
        "stabilized_nodes": stabilized,
        "mediator_nodes": mediators,
        "removed_nodes": (
            len(stabilized) + len(mediators)
        ),
        "retained_nodes": reduced.nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the two-step Boolean-network reduction of "
            "Saadatpour, Albert, and Reluga (2013)."
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

    result = generate_two_step_model(
        args.input,
        args.output,
    )

    print(
        f"Original size: {result['original_size']}"
    )
    print(
        f"After stabilized-node removal: "
        f"{result['after_step_1_size']}"
    )
    print(
        f"Reduced size: {result['reduced_size']}"
    )
    print(
        f"Total removed nodes: {result['removed_nodes']}"
    )

    stabilized = result["stabilized_nodes"]

    if stabilized:
        print("Step 1 - stabilized nodes:")
        for node, value in stabilized:
            print(f"  {node} = {value}")
    else:
        print("Step 1 - no stabilized nodes removed.")

    mediators = result["mediator_nodes"]

    if mediators:
        print("Step 2 - simple mediators:")
        for node, u, w in mediators:
            print(f"  {node}: {u} -> {node} -> {w}")
    else:
        print("Step 2 - no simple mediators removed.")

    print(
        "Retained nodes: "
        + ", ".join(result["retained_nodes"])
    )

    print(
        f"Reduced model written to: {result['output_file']}"
    )


if __name__ == "__main__":
    main()
