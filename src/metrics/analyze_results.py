from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.metrics.attractors import analyze_attractors
from src.metrics.structural import compute_structural_metrics
from src.metrics.utils import discover_models, load_bnet


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def merge_rows(
    structural_rows: list[dict[str, object]],
    attractor_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    attractor_index = {
        (
            row["network"],
            row["method"],
            row["variant"],
        ): row
        for row in attractor_rows
    }

    merged = []

    for structural in structural_rows:
        key = (
            structural["network"],
            structural["method"],
            structural["variant"],
        )

        row = dict(structural)
        attractor = attractor_index.get(
            key,
            {},
        )

        for name, value in attractor.items():
            if name not in {
                "network",
                "method",
                "variant",
                "model_file",
                "state_dimension",
            }:
                row[name] = value

        merged.append(row)

    return merged


def add_count_agreement(
    rows: list[dict[str, object]],
) -> None:
    original_counts: dict[str, object] = {}

    for row in rows:
        if row["method"] == "original":
            original_counts[row["network"]] = row[
                "fixed_points"
            ]

    for row in rows:
        original = original_counts.get(
            row["network"]
        )

        if original is None:
            row["fixed_point_count_match"] = ""
        else:
            row["fixed_point_count_match"] = (
                row["fixed_points"] == original
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute structural and attractor metrics for all "
            "Boolean-network models generated under results/models."
        )
    )

    parser.add_argument(
        "--results-dir",
        default="results/models",
        help="Directory containing one subdirectory per network.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/analysis",
        help="Directory where metric CSV files will be written.",
    )

    parser.add_argument(
        "--max-exact-dimension",
        type=int,
        default=20,
        help=(
            "Maximum Boolean dimension for exhaustive synchronous "
            "attractor enumeration. Fixed points are still computed "
            "symbolically above this threshold."
        ),
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}"
        )

    network_dirs = sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir()
    )

    structural_rows: list[dict[str, object]] = []
    attractor_rows: list[dict[str, object]] = []

    for network_dir in network_dirs:
        original_path = network_dir / "original.bnet"

        if not original_path.exists():
            print(
                f"[SKIP] {network_dir.name}: "
                "original.bnet not found"
            )
            continue

        original_variables = len(
            load_bnet(original_path).nodes
        )

        records = discover_models(
            network_dir
        )

        print()
        print("=" * 70)
        print(f"Network: {network_dir.name}")
        print(f"Models found: {len(records)}")
        print("=" * 70)

        for record in records:
            print(
                f"[ANALYZE] {record.method} / "
                f"{record.variant}"
            )

            structural_rows.append(
                compute_structural_metrics(
                    record,
                    original_variables,
                )
            )

            attractor_rows.append(
                analyze_attractors(
                    record,
                    max_exact_dimension=(
                        args.max_exact_dimension
                    ),
                )
            )

    add_count_agreement(
        attractor_rows
    )

    summary_rows = merge_rows(
        structural_rows,
        attractor_rows,
    )

    structural_path = (
        output_dir / "structural_metrics.csv"
    )
    attractor_path = (
        output_dir / "attractor_metrics.csv"
    )
    summary_path = (
        output_dir / "summary.csv"
    )

    write_csv(
        structural_path,
        structural_rows,
    )
    write_csv(
        attractor_path,
        attractor_rows,
    )
    write_csv(
        summary_path,
        summary_rows,
    )

    print()
    print("=" * 70)
    print("Analysis finished")
    print("=" * 70)
    print(f"Structural metrics: {structural_path}")
    print(f"Attractor metrics:  {attractor_path}")
    print(f"Combined summary:   {summary_path}")


if __name__ == "__main__":
    main()
