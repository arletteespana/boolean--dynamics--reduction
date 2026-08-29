from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "original",
    "dominant_vertices",
    "node_elimination",
    "two_step",
    "leaf_node_removal",
]

REDUCED_METHODS = [
    "dominant_vertices",
    "node_elimination",
    "two_step",
    "leaf_node_removal",
]


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def validate_summary(df: pd.DataFrame) -> None:
    required = {
        "network",
        "method",
        "original_variables",
        "original_fixed_variables",
        "original_free_variables",
        "retained_variables",
        "fixed_variables",
        "free_variables",
        "free_eliminated_fraction",
        "effective_state_space_size",
        "log2_effective_state_space_ratio",
        "fixed_points",
        "fixed_point_count_match",
        "analysis_type",
        "attractor_count_is_complete",
        "attractors_observed",
        "mean_transient_length",
        "max_transient_length",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if not (
        df["original_variables"]
        - df["original_fixed_variables"]
        == df["original_free_variables"]
    ).all():
        raise ValueError(
            "Inconsistent original free-variable counts."
        )

    if not (
        df["state_dimension"]
        - df["fixed_variables"]
        == df["free_variables"]
    ).all():
        raise ValueError(
            "Inconsistent reduced free-variable counts."
        )

    expected_effective = df["free_variables"].apply(
        lambda n: str(2 ** int(n))
    )

    observed_effective = (
        df["effective_state_space_size"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    if not (
        expected_effective == observed_effective
    ).all():
        raise ValueError(
            "Inconsistent effective state-space sizes."
        )


def make_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    original = df[
        df["method"] == "original"
    ].copy()

    total_original_variables = int(
        original["original_variables"].sum()
    )

    total_original_fixed = int(
        original["original_fixed_variables"].sum()
    )

    total_original_free = int(
        original["original_free_variables"].sum()
    )

    rows = []

    for method in METHOD_ORDER:
        part = df[
            df["method"] == method
        ].copy()

        if part.empty:
            continue

        total_retained = int(
            part["retained_variables"].sum()
        )

        total_fixed = int(
            part["fixed_variables"].sum()
        )

        total_free = int(
            part["free_variables"].sum()
        )

        fixed_match = as_bool(
            part["fixed_point_count_match"]
        )

        complete = as_bool(
            part["attractor_count_is_complete"]
        )

        rows.append(
            {
                "method": method,
                "networks": int(
                    part["network"].nunique()
                ),
                "original_variables_total": (
                    total_original_variables
                ),
                "original_fixed_variables_total": (
                    total_original_fixed
                ),
                "original_free_variables_total": (
                    total_original_free
                ),
                "retained_variables_total": (
                    total_retained
                ),
                "fixed_variables_total": (
                    total_fixed
                ),
                "free_variables_total": (
                    total_free
                ),
                "mean_retained_variables": (
                    part["retained_variables"].mean()
                ),
                "mean_free_variables": (
                    part["free_variables"].mean()
                ),
                "median_free_variables": (
                    part["free_variables"].median()
                ),
                "mean_free_reduction_fraction": (
                    part[
                        "free_eliminated_fraction"
                    ].mean()
                ),
                "median_free_reduction_fraction": (
                    part[
                        "free_eliminated_fraction"
                    ].median()
                ),
                "global_free_reduction_fraction": (
                    0.0
                    if method == "original"
                    else (
                        1.0
                        - total_free
                        / total_original_free
                    )
                ),
                "networks_with_zero_free_variables": int(
                    (
                        part["free_variables"] == 0
                    ).sum()
                ),
                "exact_models": int(
                    (
                        part["analysis_type"]
                        == "exact"
                    ).sum()
                ),
                "monte_carlo_models": int(
                    (
                        part["analysis_type"]
                        == "monte_carlo"
                    ).sum()
                ),
                "fixed_point_matches": int(
                    fixed_match.sum()
                ),
                "fixed_point_match_rate": float(
                    fixed_match.mean()
                ),
                "mean_log2_effective_state_space_ratio": (
                    part[
                        "log2_effective_state_space_ratio"
                    ].mean()
                ),
                "mean_transient_length": (
                    part[
                        "mean_transient_length"
                    ].mean()
                ),
                "maximum_transient_length": (
                    part[
                        "max_transient_length"
                    ].max()
                ),
                "complete_attractor_analyses": int(
                    complete.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def make_network_long(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "network",
        "method",
        "original_variables",
        "original_fixed_variables",
        "original_free_variables",
        "retained_variables",
        "fixed_variables",
        "free_variables",
        "free_eliminated_fraction",
        "effective_state_space_size",
        "log2_effective_state_space_ratio",
        "fixed_points",
        "fixed_point_count_match",
        "analysis_type",
        "attractor_count_is_complete",
        "attractors_observed",
        "fixed_attractors_observed",
        "periodic_attractors_observed",
        "mean_transient_length",
        "max_transient_length",
        "coverage_fraction",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    result = df[available].copy()

    result["free_reduction_percent"] = (
        100.0
        * result["free_eliminated_fraction"]
    )

    method_rank = {
        method: i
        for i, method in enumerate(
            METHOD_ORDER
        )
    }

    result["_method_rank"] = (
        result["method"]
        .map(method_rank)
        .fillna(999)
    )

    result = (
        result.sort_values(
            [
                "network",
                "_method_rank",
            ]
        )
        .drop(
            columns="_method_rank"
        )
        .reset_index(
            drop=True
        )
    )

    return result


def make_network_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    original = (
        df[
            df["method"] == "original"
        ][
            [
                "network",
                "original_variables",
                "original_fixed_variables",
                "original_free_variables",
                "fixed_points",
            ]
        ]
        .drop_duplicates(
            subset="network"
        )
        .set_index(
            "network"
        )
    )

    result = original.copy()

    for method in REDUCED_METHODS:
        part = (
            df[
                df["method"] == method
            ]
            .set_index(
                "network"
            )
        )

        result[
            f"{method}_retained"
        ] = part[
            "retained_variables"
        ]

        result[
            f"{method}_fixed"
        ] = part[
            "fixed_variables"
        ]

        result[
            f"{method}_free"
        ] = part[
            "free_variables"
        ]

        result[
            f"{method}_free_reduction_percent"
        ] = (
            100.0
            * part[
                "free_eliminated_fraction"
            ]
        )

        result[
            f"{method}_log2_effective_ratio"
        ] = part[
            "log2_effective_state_space_ratio"
        ]

        result[
            f"{method}_analysis_type"
        ] = part[
            "analysis_type"
        ]

        result[
            f"{method}_attractors_observed"
        ] = part[
            "attractors_observed"
        ]

        result[
            f"{method}_mean_transient_length"
        ] = part[
            "mean_transient_length"
        ]

        result[
            f"{method}_max_transient_length"
        ] = part[
            "max_transient_length"
        ]

    return (
        result.reset_index()
        .sort_values(
            "network"
        )
        .reset_index(
            drop=True
        )
    )


def make_complete_attractor_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:
    original = (
        df[
            df["method"] == "original"
        ][
            [
                "network",
                "attractor_count_is_complete",
                "attractors_observed",
            ]
        ]
        .rename(
            columns={
                "attractor_count_is_complete": (
                    "original_complete"
                ),
                "attractors_observed": (
                    "original_attractors"
                ),
            }
        )
    )

    reduced = df[
        df["method"].isin(
            REDUCED_METHODS
        )
    ].copy()

    comparison = reduced.merge(
        original,
        on="network",
        how="left",
    )

    comparison[
        "current_complete"
    ] = as_bool(
        comparison[
            "attractor_count_is_complete"
        ]
    )

    comparison[
        "original_complete_bool"
    ] = as_bool(
        comparison[
            "original_complete"
        ]
    )

    comparison = comparison[
        comparison[
            "current_complete"
        ]
        & comparison[
            "original_complete_bool"
        ]
    ].copy()

    comparison[
        "attractor_count_match_complete"
    ] = (
        comparison[
            "attractors_observed"
        ]
        == comparison[
            "original_attractors"
        ]
    )

    return comparison[
        [
            "network",
            "method",
            "free_variables",
            "original_attractors",
            "attractors_observed",
            "attractor_count_match_complete",
        ]
    ].sort_values(
        [
            "network",
            "method",
        ]
    )


def make_dv_ne_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:
    dv = (
        df[
            df["method"]
            == "dominant_vertices"
        ][
            [
                "network",
                "retained_variables",
                "fixed_variables",
                "free_variables",
                "free_eliminated_fraction",
            ]
        ]
        .rename(
            columns={
                "retained_variables": (
                    "DV_retained"
                ),
                "fixed_variables": (
                    "DV_fixed"
                ),
                "free_variables": (
                    "DV_free"
                ),
                "free_eliminated_fraction": (
                    "DV_free_elimination"
                ),
            }
        )
    )

    ne = (
        df[
            df["method"]
            == "node_elimination"
        ][
            [
                "network",
                "retained_variables",
                "fixed_variables",
                "free_variables",
                "free_eliminated_fraction",
            ]
        ]
        .rename(
            columns={
                "retained_variables": (
                    "NE_retained"
                ),
                "fixed_variables": (
                    "NE_fixed"
                ),
                "free_variables": (
                    "NE_free"
                ),
                "free_eliminated_fraction": (
                    "NE_free_elimination"
                ),
            }
        )
    )

    comparison = dv.merge(
        ne,
        on="network",
        how="inner",
    )

    comparison[
        "same_free_dimension"
    ] = (
        comparison[
            "DV_free"
        ]
        == comparison[
            "NE_free"
        ]
    )

    return comparison.sort_values(
        "network"
    )


def plot_free_variables_by_network(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    reduced = (
        df[
            df["method"].isin(
                REDUCED_METHODS
            )
        ]
        .pivot(
            index="network",
            columns="method",
            values="free_variables",
        )
        .reindex(
            columns=REDUCED_METHODS
        )
    )

    ax = reduced.plot(
        kind="bar",
        figsize=(13, 6),
    )

    ax.set_xlabel(
        "Network"
    )
    ax.set_ylabel(
        "Free variables retained"
    )
    ax.set_title(
        "Dynamic degrees of freedom retained by reduction method"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "free_variables_by_network.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_free_reduction_by_method(
    method_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    data = method_summary[
        method_summary[
            "method"
        ].isin(
            REDUCED_METHODS
        )
    ].copy()

    values = (
        100.0
        * data[
            "mean_free_reduction_fraction"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        data["method"],
        values,
    )

    ax.set_ylabel(
        "Mean free-variable reduction (%)"
    )
    ax.set_xlabel(
        "Method"
    )
    ax.set_title(
        "Mean dynamic reduction across the 14 networks"
    )

    ax.set_ylim(
        0,
        105,
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "mean_free_reduction_by_method.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def print_key_findings(
    df: pd.DataFrame,
    method_summary: pd.DataFrame,
    dv_ne: pd.DataFrame,
) -> None:
    original = df[
        df["method"] == "original"
    ]

    print()
    print("=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)

    print(
        "Original totals:",
        f"N={int(original['original_variables'].sum())},",
        f"fixed={int(original['original_fixed_variables'].sum())},",
        f"free={int(original['original_free_variables'].sum())}",
    )

    reduced_summary = method_summary[
        method_summary["method"].isin(
            REDUCED_METHODS
        )
    ]

    print()
    print(
        reduced_summary[
            [
                "method",
                "retained_variables_total",
                "fixed_variables_total",
                "free_variables_total",
                "mean_free_reduction_fraction",
                "global_free_reduction_fraction",
                "networks_with_zero_free_variables",
                "exact_models",
                "monte_carlo_models",
                "fixed_point_match_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "DV and NE have the same free dimension in",
        f"{int(dv_ne['same_free_dimension'].sum())}/"
        f"{len(dv_ne)} networks."
    )

    nonzero = (
        df[
            (
                df["method"]
                == "dominant_vertices"
            )
            & (
                df["free_variables"]
                > 0
            )
        ][
            [
                "network",
                "free_variables",
            ]
        ]
        .sort_values(
            "network"
        )
    )

    print()
    print(
        "Networks with nonzero DV dynamic core:"
    )

    if nonzero.empty:
        print(
            "None."
        )
    else:
        print(
            nonzero.to_string(
                index=False
            )
        )

    fixed_match_all = as_bool(
        df[
            "fixed_point_count_match"
        ]
    ).all()

    print()
    print(
        "Fixed-point counts preserved in all rows:",
        fixed_match_all,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible comparison of Boolean-network "
            "reduction methods from summary.csv."
        )
    )

    parser.add_argument(
        "--summary",
        default="results/analysis/summary.csv",
        help="Path to summary.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="results/analysis/comparison",
        help="Directory for derived tables and figures",
    )

    args = parser.parse_args()

    summary_path = Path(
        args.summary
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        summary_path
    )

    validate_summary(
        df
    )

    method_summary = (
        make_method_summary(
            df
        )
    )

    network_long = (
        make_network_long(
            df
        )
    )

    network_summary = (
        make_network_summary(
            df
        )
    )

    complete_attractors = (
        make_complete_attractor_comparison(
            df
        )
    )

    dv_ne = (
        make_dv_ne_comparison(
            df
        )
    )

    method_summary.to_csv(
        output_dir
        / "method_summary.csv",
        index=False,
    )

    network_long.to_csv(
        output_dir
        / "network_method_long.csv",
        index=False,
    )

    network_summary.to_csv(
        output_dir
        / "network_summary.csv",
        index=False,
    )

    complete_attractors.to_csv(
        output_dir
        / "complete_attractor_comparison.csv",
        index=False,
    )

    dv_ne.to_csv(
        output_dir
        / "dv_vs_node_elimination.csv",
        index=False,
    )

    plot_free_variables_by_network(
        df,
        output_dir,
    )

    plot_free_reduction_by_method(
        method_summary,
        output_dir,
    )

    print_key_findings(
        df,
        method_summary,
        dv_ne,
    )

    print()
    print(
        f"Outputs written to: {output_dir}"
    )


if __name__ == "__main__":
    main()
