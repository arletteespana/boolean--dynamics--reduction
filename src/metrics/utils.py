from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import sympy as sp


@dataclass
class BNetModel:
    nodes: list[str]
    rules: dict[str, sp.Basic]
    symbols: dict[str, sp.Symbol]
    has_header: bool = False


@dataclass
class ModelRecord:
    network: str
    method: str
    variant: str
    path: Path
    metadata: dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# .bnet parser
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
        while (
            j < len(rule)
            and (not rule[j].isspace())
            and rule[j] not in "()!~&|"
        ):
            j += 1

        tokens.append(rule[i:j])
        i = j

    return tokens


def _parse_rule(
    rule: str,
    symbols: dict[str, sp.Symbol],
) -> sp.Basic:
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
            converted.append(
                symbols[token].name
            )

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


def load_bnet(
    path: str | Path,
) -> BNetModel:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    raw_rules: list[
        tuple[str, str]
    ] = []

    has_header = False

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        1,
    ):
        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
        ):
            continue

        if line.lower().replace(
            " ",
            "",
        ) in {
            "targets,factors",
            "target,factor",
        }:
            has_header = True
            continue

        if "," not in line:
            raise ValueError(
                f"Invalid .bnet line "
                f"{line_number}: {raw_line}"
            )

        node, rule = line.split(
            ",",
            1,
        )

        node = node.strip()
        rule = rule.strip()

        if not node or not rule:
            raise ValueError(
                f"Invalid .bnet line "
                f"{line_number}: {raw_line}"
            )

        raw_rules.append(
            (
                node,
                rule,
            )
        )

    nodes = [
        node
        for node, _ in raw_rules
    ]

    if len(nodes) != len(set(nodes)):
        raise ValueError(
            f"Duplicated target variables "
            f"found in {path}"
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


# -----------------------------------------------------------------------------
# Result discovery
# -----------------------------------------------------------------------------

def _read_dominant_summary(
    path: Path,
) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}

    rows: dict[
        str,
        dict[str, object],
    ] = {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            filename = row[
                "file"
            ]

            rows[
                filename
            ] = {
                "id": int(
                    row["id"]
                ),
                "dominant_set_size": int(
                    row[
                        "dominant_set_size"
                    ]
                ),
                "dominant_set": row[
                    "dominant_set"
                ],
                "depth": int(
                    row["depth"]
                ),
                "recurrence_length": int(
                    row[
                        "recurrence_length"
                    ]
                ),
                "state_dimension": int(
                    row[
                        "state_dimension"
                    ]
                ),
            }

    return rows


def discover_models(
    network_dir: str | Path,
) -> list[ModelRecord]:
    network_dir = Path(
        network_dir
    )

    network = (
        network_dir.name
    )

    records: list[
        ModelRecord
    ] = []

    single_models = [
        (
            "original",
            "original",
            "original.bnet",
        ),
        (
            "node_elimination",
            "node_elimination",
            "node_elimination.bnet",
        ),
        (
            "two_step",
            "two_step",
            "two_step.bnet",
        ),
        (
            "leaf_node_removal",
            "leaf_node_removal",
            "leaf_node_removal.bnet",
        ),
    ]

    for (
        method,
        variant,
        filename,
    ) in single_models:
        path = (
            network_dir
            / filename
        )

        if path.exists():
            records.append(
                ModelRecord(
                    network=network,
                    method=method,
                    variant=variant,
                    path=path,
                )
            )

    dominant_dir = (
        network_dir
        / "dominant_vertices"
    )

    if dominant_dir.exists():
        summary = (
            _read_dominant_summary(
                dominant_dir
                / "dominant_vertices_summary.csv"
            )
        )

        for path in sorted(
            dominant_dir.glob(
                "dominant_vertices_*.bnet"
            )
        ):
            metadata = summary.get(
                path.name,
                {},
            )

            dv_id = metadata.get(
                "id"
            )

            if dv_id is None:
                variant = path.stem

            else:
                variant = (
                    f"dv_{int(dv_id):02d}"
                )

            records.append(
                ModelRecord(
                    network=network,
                    method="dominant_vertices",
                    variant=variant,
                    path=path,
                    metadata=metadata,
                )
            )

    return records
