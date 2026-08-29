from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.metrics.attractors import analyze_fixed_points
from src.metrics.structural import compute_structural_metrics
from src.metrics.transients import (
    analyze_synchronous_dynamics,
    count_fixed_variables,
    count_free_variables,
)
from src.metrics.utils import (
    discover_models,
    load_bnet,
)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
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


def merge_model_rows(
    structural_rows: list[dict[str, object]],
    fixed_point_rows: list[dict[str, object]],
    dynamic_rows: list[dict[str, object]],
) -> list[dict[str, object]]:

    def make_index(rows):
        return {
            (
                row["network"],
                row["method"],
                row["variant"],
            ): row
            for row in rows
        }

    fixed_index = make_index(
        fixed_point_rows
    )

    dynamic_index = make_index(
        dynamic_rows
    )

    merged = []

    for structural in structural_rows:
        key = (
            structural["network"],
            structural["method"],
            structural["variant"],
        )

        row = dict(
            structural
        )

        for source in (
            fixed_index.get(
                key,
                {},
            ),
            dynamic_index.get(
                key,
                {},
            ),
        ):
            for name, value in source.items():

                if name not in {
                    "network",
                    "method",
                    "variant",
                    "model_file",
                    "state_dimension",
                    "state_space_size",
                    "fixed_variables",
                    "free_variables",
                    "effective_state_space_size",
                }:
                    row[name] = value

        merged.append(
            row
        )

    return merged


def add_fixed_point_agreement(
    rows: list[dict[str, object]],
) -> None:
    original_counts: dict[
        str,
        object,
    ] = {}

    for row in rows:

        if row["method"] == "original":
            original_counts[
                row["network"]
            ] = row[
                "fixed_points"
            ]

    for row in rows:

        original = original_counts.get(
            row["network"]
        )

        row[
            "fixed_point_count_match"
        ] = (
            ""
            if original is None
            else (
                row["fixed_points"]
                == original
            )
        )


def add_total_attractor_agreement(
    rows: list[dict[str, object]],
) -> None:
    originals: dict[
        str,
        tuple[bool, object],
    ] = {}

    for row in rows:

        if row["method"] == "original":
            originals[
                row["network"]
            ] = (
                bool(
                    row[
                        "attractor_count_is_complete"
                    ]
                ),
                row[
                    "attractors_observed"
                ],
            )

    for row in rows:

        original = originals.get(
            row["network"]
        )

        if original is None:
            row[
                "total_attractor_count_match"
            ] = ""
            continue

        (
            original_complete,
            original_count,
        ) = original

        current_complete = bool(
            row[
                "attractor_count_is_complete"
            ]
        )

        if not (
            original_complete
            and current_complete
        ):
            row[
                "total_attractor_count_match"
            ] = ""

        else:
            row[
                "total_attractor_count_match"
            ] = (
                row[
                    "attractors_observed"
                ]
                == original_count
            )


def theoretical_dv_fixed_point_row(
    record,
    model,
    original_fixed_points: int,
) -> dict[str, object]:

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
        "fixed_points": (
            original_fixed_points
        ),
        "fixed_point_time_seconds": "",
        "fixed_point_source": (
            "theoretical_preservation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute structural, fixed-point, basin, attractor, and "
            "transient metrics for all generated Boolean-network models."
        )
    )

    parser.add_argument(
        "--results-dir",
        default="results/models",
        help=(
            "Directory containing one "
            "subdirectory per network."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="results/analysis",
        help=(
            "Directory where metric CSV "
            "files will be written."
        ),
    )

    parser.add_argument(
        "--exact-dimension",
        "--max-exact-dimension",
        dest="exact_dimension",
        type=int,
        default=20,
        help=(
            "Models with free dimension <= this value are analyzed "
            "exhaustively; larger models use Monte Carlo."
        ),
    )

    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=10_000,
        help=(
            "Number of uniformly sampled admissible initial states for "
            "models above the exact-dimension threshold."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Base random seed. A deterministic model-specific seed is "
            "derived from it for reproducibility."
        ),
    )

    args = parser.parse_args()

    results_dir = Path(
        args.results_dir
    )

    output_dir = Path(
        args.output_dir
    )

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: "
            f"{results_dir}"
        )

    network_dirs = sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir()
    )

    structural_rows: list[
        dict[str, object]
    ] = []

    fixed_point_rows: list[
        dict[str, object]
    ] = []

    dynamic_rows: list[
        dict[str, object]
    ] = []

    basin_rows: list[
        dict[str, object]
    ] = []

    for network_dir in network_dirs:

        original_path = (
            network_dir
            / "original.bnet"
        )

        if not original_path.exists():
            print(
                f"[SKIP] {network_dir.name}: "
                "original.bnet not found"
            )
            continue

        original_model = load_bnet(
            original_path
        )

        original_variables = len(
            original_model.nodes
        )

        original_fixed_variables = (
            count_fixed_variables(
                original_model
            )
        )

        original_free_variables = (
            count_free_variables(
                original_model
            )
        )

        records = discover_models(
            network_dir
        )

        original_record = next(
            (
                record
                for record in records
                if record.method
                == "original"
            ),
            None,
        )

        if original_record is None:
            print(
                f"[SKIP] {network_dir.name}: "
                "original model record not found"
            )
            continue

        print()
        print("=" * 70)

        print(
            f"Network: "
            f"{network_dir.name}"
        )

        print(
            f"Models found: "
            f"{len(records)}"
        )

        print(
            f"Original variables: "
            f"{original_variables}"
        )

        print(
            f"Original fixed variables: "
            f"{original_fixed_variables}"
        )

        print(
            f"Original free variables: "
            f"{original_free_variables}"
        )

        print("=" * 70)

        print(
            "[FIXED POINTS] original / original "
            "(exact mpbn)"
        )

        original_fixed_row = (
            analyze_fixed_points(
                original_record
            )
        )

        original_fixed_row[
            "fixed_point_source"
        ] = "mpbn_exact"

        original_fixed_points = int(
            original_fixed_row[
                "fixed_points"
            ]
        )

        for record in records:

            model = load_bnet(
                record.path
            )

            fixed_variables = (
                count_fixed_variables(
                    model
                )
            )

            free_variables = (
                count_free_variables(
                    model
                )
            )

            analysis_type = (
                "exact"
                if free_variables
                <= args.exact_dimension
                else "monte_carlo"
            )

            print(
                f"[ANALYZE] "
                f"{record.method} / "
                f"{record.variant} "
                f"(D={len(model.nodes)}, "
                f"fixed={fixed_variables}, "
                f"free={free_variables}, "
                f"{analysis_type})"
            )

            structural_rows.append(
                compute_structural_metrics(
                    record,
                    original_variables,
                    original_fixed_variables,
                )
            )

            if record.method == "original":

                fixed_row = dict(
                    original_fixed_row
                )

            elif (
                record.method
                == "dominant_vertices"
            ):

                print(
                    "  [FIXED POINTS] inherited from original "
                    "(theoretical preservation)"
                )

                fixed_row = (
                    theoretical_dv_fixed_point_row(
                        record=record,
                        model=model,
                        original_fixed_points=(
                            original_fixed_points
                        ),
                    )
                )

            else:

                print(
                    "  [FIXED POINTS] exact mpbn"
                )

                fixed_row = (
                    analyze_fixed_points(
                        record
                    )
                )

                fixed_row[
                    "fixed_point_source"
                ] = "mpbn_exact"

            fixed_point_rows.append(
                fixed_row
            )

            print(
                f"  [DYNAMICS] "
                f"{analysis_type}"
            )

            dynamics, basins = (
                analyze_synchronous_dynamics(
                    model=model,
                    network=record.network,
                    method=record.method,
                    variant=record.variant,
                    exact_dimension=(
                        args.exact_dimension
                    ),
                    monte_carlo_samples=(
                        args.monte_carlo_samples
                    ),
                    base_seed=args.seed,
                )
            )

            dynamic_rows.append(
                {
                    "network": record.network,
                    "method": record.method,
                    "variant": record.variant,
                    "model_file": str(
                        record.path
                    ),
                    "state_dimension": len(
                        model.nodes
                    ),
                    **dynamics,
                }
            )

            basin_rows.extend(
                basins
            )

    add_fixed_point_agreement(
        fixed_point_rows
    )

    add_total_attractor_agreement(
        dynamic_rows
    )

    summary_rows = (
        merge_model_rows(
            structural_rows,
            fixed_point_rows,
            dynamic_rows,
        )
    )

    write_csv(
        output_dir
        / "structural_metrics.csv",
        structural_rows,
    )

    write_csv(
        output_dir
        / "fixed_point_metrics.csv",
        fixed_point_rows,
    )

    write_csv(
        output_dir
        / "dynamic_metrics.csv",
        dynamic_rows,
    )

    write_csv(
        output_dir
        / "basin_metrics.csv",
        basin_rows,
    )

    write_csv(
        output_dir
        / "summary.csv",
        summary_rows,
    )

    print()
    print("=" * 70)
    print("Analysis finished")
    print("=" * 70)

    print(
        f"Exact threshold: "
        f"free D <= "
        f"{args.exact_dimension}"
    )

    print(
        f"Monte Carlo samples: "
        f"{args.monte_carlo_samples}"
    )

    print(
        f"Base seed: "
        f"{args.seed}"
    )

    print(
        f"Results written to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
