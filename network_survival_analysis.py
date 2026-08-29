from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.metrics.utils import load_bnet


SURVIVAL = {
    "acute_myeloid_leukemia": 23.6,
    "basal_cell_carcinoma": 91.4,
    "bladder_cancer": 78.1,
    "chronic_myeloid_leukemia": 55.2,
    "colorectal_cancer": 63.6,
    "endometrial_cancer": 68.6,
    "glioma": 33.4,
    "melanoma": 91.2,
    "non_small_cell_lung_cancer": 18.0,
    "pancreatic_cancer": 5.5,
    "prostate_cancer": 99.4,
    "renal_cell_carcinoma": 69.5,
    "small_cell_lung_cancer": 6.2,
    "thyroid_cancer": 97.2,
}


DISPLAY_NAMES = {
    "acute_myeloid_leukemia": "AML",
    "basal_cell_carcinoma": "Basal",
    "bladder_cancer": "Bladder",
    "chronic_myeloid_leukemia": "CML",
    "colorectal_cancer": "Colorectal",
    "endometrial_cancer": "Endometrial",
    "glioma": "Glioma",
    "melanoma": "Melanoma",
    "non_small_cell_lung_cancer": "NSCLC",
    "pancreatic_cancer": "Pancreatic",
    "prostate_cancer": "Prostate",
    "renal_cell_carcinoma": "RCC",
    "small_cell_lung_cancer": "SCLC",
    "thyroid_cancer": "Thyroid",
}


def interaction_graph(model) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes)

    symbol_to_node = {
        symbol: node
        for node, symbol in model.symbols.items()
    }

    for target in model.nodes:
        for symbol in model.rules[target].free_symbols:
            source = symbol_to_node[symbol]
            graph.add_edge(source, target)

    return graph


def degree_entropy(values) -> float:
    """
    Shannon entropy of a degree-frequency distribution, using natural logs.

    If p_k is the fraction of nodes with degree k, then
        H = - sum_k p_k log(p_k).
    """
    values = list(values)

    if not values:
        return 0.0

    counts = pd.Series(values).value_counts()
    probabilities = counts / counts.sum()

    return float(
        -sum(
            p * math.log(p)
            for p in probabilities
            if p > 0
        )
    )


def cyclic_sccs(graph: nx.DiGraph) -> list[set[str]]:
    result = []

    for component in nx.strongly_connected_components(graph):
        component = set(component)

        if len(component) > 1:
            result.append(component)
            continue

        node = next(iter(component))

        if graph.has_edge(node, node):
            result.append(component)

    return result


def dominant_metadata(
    results_dir: Path,
    network: str,
) -> dict[str, object]:
    path = (
        results_dir
        / network
        / "dominant_vertices"
        / "dominant_vertices_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"DV summary not found: {path}"
        )

    summary = pd.read_csv(path)

    if summary.empty:
        raise ValueError(
            f"DV summary is empty: {path}"
        )

    # Use the first generated minimum dominant-set variant.
    row = summary.iloc[0]

    dominant_set = [
        item.strip()
        for item in str(row["dominant_set"]).split(";")
        if item.strip()
    ]

    return {
        "dominant_set": dominant_set,
        "dominant_set_size": int(row["dominant_set_size"]),
        "depth": int(row["depth"]),
        "recurrence_length": int(row["recurrence_length"]),
        "state_dimension": int(row["state_dimension"]),
    }


def centrality_ratio(
    centrality: dict[str, float],
    selected: set[str],
) -> float:
    all_values = np.array(
        list(centrality.values()),
        dtype=float,
    )

    selected_values = np.array(
        [
            centrality[node]
            for node in selected
            if node in centrality
        ],
        dtype=float,
    )

    if (
        len(all_values) == 0
        or len(selected_values) == 0
    ):
        return np.nan

    denominator = float(
        np.mean(all_values)
    )

    numerator = float(
        np.mean(selected_values)
    )

    if denominator == 0:
        return np.nan

    return numerator / denominator


def top_fraction_overlap(
    centrality: dict[str, float],
    selected: set[str],
    fraction: float = 0.10,
) -> float:
    if not centrality or not selected:
        return np.nan

    n_top = max(
        1,
        math.ceil(
            fraction
            * len(centrality)
        ),
    )

    top_nodes = {
        node
        for node, _
        in sorted(
            centrality.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:n_top]
    }

    return (
        len(selected & top_nodes)
        / len(selected)
    )


def network_metrics(
    network_file: Path,
    results_dir: Path,
) -> dict[str, object]:
    network = network_file.stem

    model = load_bnet(
        network_file
    )

    graph = interaction_graph(
        model
    )

    n = graph.number_of_nodes()
    e = graph.number_of_edges()

    fixed_nodes = {
        node
        for node in model.nodes
        if not model.rules[node].free_symbols
    }

    free_nodes = (
        set(model.nodes)
        - fixed_nodes
    )

    dv = dominant_metadata(
        results_dir,
        network,
    )

    dominant_set = set(
        dv["dominant_set"]
    )

    dominant_fixed = (
        dominant_set
        & fixed_nodes
    )

    dominant_dynamic = (
        dominant_set
        - fixed_nodes
    )

    undirected = nx.Graph()
    undirected.add_nodes_from(
        graph.nodes
    )

    undirected.add_edges_from(
        (
            u,
            v,
        )
        for u, v in graph.edges
        if u != v
    )

    total_degree = dict(
        undirected.degree()
    )

    in_degree = dict(
        graph.in_degree()
    )

    out_degree = dict(
        graph.out_degree()
    )

    cyclic_components = (
        cyclic_sccs(
            graph
        )
    )

    cyclic_nodes = set().union(
        *cyclic_components
    ) if cyclic_components else set()

    sccs = list(
        nx.strongly_connected_components(
            graph
        )
    )

    largest_scc = max(
        (
            len(component)
            for component in sccs
        ),
        default=0,
    )

    degree_centrality = (
        nx.degree_centrality(
            undirected
        )
        if n > 1
        else {
            node: 0.0
            for node in graph.nodes
        }
    )

    betweenness = (
        nx.betweenness_centrality(
            graph,
            normalized=True,
        )
        if n > 1
        else {
            node: 0.0
            for node in graph.nodes
        }
    )

    pagerank = (
        nx.pagerank(
            graph,
        )
        if n > 0
        else {}
    )

    clustering = (
        nx.average_clustering(
            undirected
        )
        if n > 1
        else 0.0
    )

    density = (
        nx.density(
            graph
        )
        if n > 1
        else 0.0
    )

    return {
        "network": network,
        "label": DISPLAY_NAMES.get(
            network,
            network,
        ),
        "survival_5y_percent": SURVIVAL[
            network
        ],

        "nodes": n,
        "edges": e,
        "fixed_inputs": len(
            fixed_nodes
        ),
        "free_variables_original": len(
            free_nodes
        ),

        # Original-network descriptors
        "degree_entropy_H": degree_entropy(
            total_degree.values()
        ),
        "in_degree_entropy": degree_entropy(
            in_degree.values()
        ),
        "out_degree_entropy": degree_entropy(
            out_degree.values()
        ),
        "edges_per_node": (
            e / n
            if n
            else np.nan
        ),
        "density": density,
        "average_clustering": clustering,
        "largest_scc_fraction": (
            largest_scc / n
            if n
            else np.nan
        ),
        "cyclic_scc_count": len(
            cyclic_components
        ),
        "cyclic_node_fraction": (
            len(cyclic_nodes) / n
            if n
            else np.nan
        ),
        "self_loops": nx.number_of_selfloops(
            graph
        ),

        # Dominant-vertices descriptors
        "dominant_set_size": len(
            dominant_set
        ),
        "dominant_fraction": (
            len(dominant_set) / n
            if n
            else np.nan
        ),
        "dominant_fixed_count": len(
            dominant_fixed
        ),
        "dominant_dynamic_count": len(
            dominant_dynamic
        ),
        "dominant_fixed_fraction": (
            len(dominant_fixed)
            / len(dominant_set)
            if dominant_set
            else np.nan
        ),
        "dominant_depth": dv[
            "depth"
        ],
        "recurrence_length": dv[
            "recurrence_length"
        ],

        # Are dominant vertices structurally central?
        "dominant_degree_enrichment": (
            centrality_ratio(
                degree_centrality,
                dominant_set,
            )
        ),
        "dominant_betweenness_enrichment": (
            centrality_ratio(
                betweenness,
                dominant_set,
            )
        ),
        "dominant_pagerank_enrichment": (
            centrality_ratio(
                pagerank,
                dominant_set,
            )
        ),
        "dominant_top10_degree_fraction": (
            top_fraction_overlap(
                degree_centrality,
                dominant_set,
                fraction=0.10,
            )
        ),
        "dominant_top10_betweenness_fraction": (
            top_fraction_overlap(
                betweenness,
                dominant_set,
                fraction=0.10,
            )
        ),
        "dominant_top10_pagerank_fraction": (
            top_fraction_overlap(
                pagerank,
                dominant_set,
                fraction=0.10,
            )
        ),
    }


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    """
    Benjamini-Hochberg FDR adjusted p-values.
    """
    p = p_values.astype(float).to_numpy()
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = np.empty(
        n,
        dtype=float,
    )

    running = 1.0

    for i in range(
        n - 1,
        -1,
        -1,
    ):
        rank = i + 1

        value = (
            ranked[i]
            * n
            / rank
        )

        running = min(
            running,
            value,
        )

        adjusted[
            order[i]
        ] = min(
            1.0,
            running,
        )

    return pd.Series(
        adjusted,
        index=p_values.index,
    )


def correlation_table(
    metrics: pd.DataFrame,
    exclude_prostate: bool = False,
) -> pd.DataFrame:
    data = metrics.copy()

    if exclude_prostate:
        data = data[
            data["network"]
            != "prostate_cancer"
        ].copy()

    excluded = {
        "network",
        "label",
        "survival_5y_percent",
    }

    candidate_columns = [
        column
        for column in data.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]

    rows = []

    survival = (
        data["survival_5y_percent"]
        .astype(float)
    )

    for metric in candidate_columns:
        x = pd.to_numeric(
            data[metric],
            errors="coerce",
        )

        valid = (
            x.notna()
            & survival.notna()
        )

        xv = x[valid]
        yv = survival[valid]

        if len(xv) < 4:
            continue

        if xv.nunique() < 2:
            continue

        pearson_r, pearson_p = (
            pearsonr(
                xv,
                yv,
            )
        )

        spearman_rho, spearman_p = (
            spearmanr(
                xv,
                yv,
            )
        )

        rows.append(
            {
                "metric": metric,
                "n": int(
                    len(xv)
                ),
                "pearson_r": pearson_r,
                "pearson_r2": pearson_r ** 2,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result[
        "pearson_fdr"
    ] = benjamini_hochberg(
        result["pearson_p"]
    )

    result[
        "spearman_fdr"
    ] = benjamini_hochberg(
        result["spearman_p"]
    )

    result[
        "abs_spearman_rho"
    ] = result[
        "spearman_rho"
    ].abs()

    result = result.sort_values(
        [
            "spearman_fdr",
            "spearman_p",
            "abs_spearman_rho",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return result


def dv_vs_structure_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    dv_metrics = [
        "dominant_set_size",
        "dominant_fraction",
        "dominant_depth",
        "dominant_degree_enrichment",
        "dominant_betweenness_enrichment",
        "dominant_pagerank_enrichment",
    ]

    structure_metrics = [
        "degree_entropy_H",
        "in_degree_entropy",
        "out_degree_entropy",
        "edges_per_node",
        "density",
        "average_clustering",
        "largest_scc_fraction",
        "cyclic_scc_count",
        "cyclic_node_fraction",
    ]

    rows = []

    for dv_metric in dv_metrics:
        for structure_metric in structure_metrics:
            x = pd.to_numeric(
                metrics[dv_metric],
                errors="coerce",
            )

            y = pd.to_numeric(
                metrics[structure_metric],
                errors="coerce",
            )

            valid = (
                x.notna()
                & y.notna()
            )

            xv = x[valid]
            yv = y[valid]

            if (
                len(xv) < 4
                or xv.nunique() < 2
                or yv.nunique() < 2
            ):
                continue

            pearson_r, pearson_p = (
                pearsonr(
                    xv,
                    yv,
                )
            )

            spearman_rho, spearman_p = (
                spearmanr(
                    xv,
                    yv,
                )
            )

            rows.append(
                {
                    "dv_metric": dv_metric,
                    "network_metric": structure_metric,
                    "n": int(
                        len(xv)
                    ),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result[
        "pearson_fdr"
    ] = benjamini_hochberg(
        result["pearson_p"]
    )

    result[
        "spearman_fdr"
    ] = benjamini_hochberg(
        result["spearman_p"]
    )

    result[
        "abs_spearman_rho"
    ] = result[
        "spearman_rho"
    ].abs()

    return result.sort_values(
        [
            "spearman_fdr",
            "spearman_p",
            "abs_spearman_rho",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


def scatter_plot(
    data: pd.DataFrame,
    metric: str,
    output: Path,
) -> None:
    subset = data[
        [
            "label",
            "survival_5y_percent",
            metric,
        ]
    ].dropna()

    if (
        len(subset) < 4
        or subset[metric].nunique() < 2
    ):
        return

    x = subset[
        metric
    ].astype(float)

    y = subset[
        "survival_5y_percent"
    ].astype(float)

    pearson_r, pearson_p = (
        pearsonr(
            x,
            y,
        )
    )

    spearman_rho, spearman_p = (
        spearmanr(
            x,
            y,
        )
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 4.6)
    )

    ax.scatter(
        x,
        y,
        s=38,
    )

    if x.nunique() >= 2:
        coefficients = np.polyfit(
            x,
            y,
            1,
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            200,
        )

        y_line = (
            coefficients[0]
            * x_line
            + coefficients[1]
        )

        ax.plot(
            x_line,
            y_line,
            linewidth=1.2,
        )

    for _, row in subset.iterrows():
        ax.annotate(
            row["label"],
            (
                row[metric],
                row[
                    "survival_5y_percent"
                ],
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel(
        metric.replace(
            "_",
            " ",
        )
    )

    ax.set_ylabel(
        "5-year survival (%)"
    )

    ax.set_title(
        (
            f"Pearson r={pearson_r:.3f}, p={pearson_p:.3g}; "
            f"Spearman rho={spearman_rho:.3f}, p={spearman_p:.3g}"
        ),
        fontsize=9,
    )

    ax.set_ylim(
        0,
        105,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Explore associations between 5-year cancer survival, "
            "network structure, and dominant-vertices descriptors."
        )
    )

    parser.add_argument(
        "--networks-dir",
        default="networks",
    )

    parser.add_argument(
        "--results-dir",
        default="results/models",
    )

    parser.add_argument(
        "--output-dir",
        default="results/analysis/survival",
    )

    args = parser.parse_args()

    networks_dir = Path(
        args.networks_dir
    )

    results_dir = Path(
        args.results_dir
    )

    output_dir = Path(
        args.output_dir
    )

    figures_dir = (
        output_dir
        / "figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        networks_dir.glob(
            "*.bnet"
        )
    )

    networks_found = {
        path.stem
        for path in files
    }

    missing_survival = (
        networks_found
        - set(SURVIVAL)
    )

    missing_networks = (
        set(SURVIVAL)
        - networks_found
    )

    if missing_survival:
        raise ValueError(
            "Survival values missing for: "
            + ", ".join(
                sorted(
                    missing_survival
                )
            )
        )

    if missing_networks:
        raise ValueError(
            "Network files missing for: "
            + ", ".join(
                sorted(
                    missing_networks
                )
            )
        )

    rows = [
        network_metrics(
            path,
            results_dir,
        )
        for path in files
    ]

    metrics = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "network"
        )
        .reset_index(
            drop=True
        )
    )

    metrics.to_csv(
        output_dir
        / "network_survival_metrics.csv",
        index=False,
    )

    correlations_all = (
        correlation_table(
            metrics,
            exclude_prostate=False,
        )
    )

    correlations_all.to_csv(
        output_dir
        / "survival_correlations_all14.csv",
        index=False,
    )

    correlations_no_prostate = (
        correlation_table(
            metrics,
            exclude_prostate=True,
        )
    )

    correlations_no_prostate.to_csv(
        output_dir
        / "survival_correlations_without_prostate.csv",
        index=False,
    )

    dv_structure = (
        dv_vs_structure_table(
            metrics
        )
    )

    dv_structure.to_csv(
        output_dir
        / "dv_vs_network_structure.csv",
        index=False,
    )

    plot_metrics = [
        "degree_entropy_H",
        "dominant_set_size",
        "dominant_fraction",
        "dominant_depth",
        "largest_scc_fraction",
        "cyclic_node_fraction",
        "dominant_degree_enrichment",
        "dominant_betweenness_enrichment",
        "dominant_pagerank_enrichment",
    ]

    for metric in plot_metrics:
        scatter_plot(
            metrics,
            metric,
            figures_dir
            / f"survival_vs_{metric}.png",
        )

    print()
    print("=" * 78)
    print("NETWORK + DV + SURVIVAL ANALYSIS")
    print("=" * 78)
    print(
        f"Networks analyzed: {len(metrics)}"
    )
    print(
        f"Outputs: {output_dir}"
    )

    print()
    print("Top survival associations (all 14):")
    print(
        correlations_all[
            [
                "metric",
                "pearson_r",
                "pearson_p",
                "pearson_fdr",
                "spearman_rho",
                "spearman_p",
                "spearman_fdr",
            ]
        ]
        .head(12)
        .to_string(
            index=False
        )
    )

    print()
    print("Top survival associations (without prostate):")
    print(
        correlations_no_prostate[
            [
                "metric",
                "pearson_r",
                "pearson_p",
                "pearson_fdr",
                "spearman_rho",
                "spearman_p",
                "spearman_fdr",
            ]
        ]
        .head(12)
        .to_string(
            index=False
        )
    )

    print()
    print("Top DV-vs-network-structure associations:")
    print(
        dv_structure[
            [
                "dv_metric",
                "network_metric",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
