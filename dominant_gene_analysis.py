from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.metrics.utils import load_bnet


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


NETWORK_ORDER = list(DISPLAY_NAMES)


def normalize_hsa_id(node: str) -> tuple[str, str]:
    """
    Return (normalized_node_id, entrez_id).

    Examples:
        hsa_5914 -> (hsa_5914, 5914)
        hsa:5914 -> (hsa_5914, 5914)
    """
    node = str(node).strip()

    match = re.search(r"hsa[_:](\d+)$", node)

    if not match:
        return node, ""

    entrez = match.group(1)

    return f"hsa_{entrez}", entrez


def find_bnet(
    networks_dir: Path,
    network: str,
) -> Path:
    direct = networks_dir / f"{network}.bnet"

    if direct.exists():
        return direct

    matches = list(
        networks_dir.rglob(
            f"{network}.bnet"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Original .bnet not found for {network}"
        )

    raise RuntimeError(
        f"More than one .bnet found for {network}: {matches}"
    )


def discover_dv_summaries(
    results_dir: Path,
) -> dict[str, Path]:
    summaries = {}

    for path in results_dir.rglob(
        "dominant_vertices_summary.csv"
    ):
        # Expected structure:
        # results/models/<network>/dominant_vertices/dominant_vertices_summary.csv
        network = path.parent.parent.name
        summaries[network] = path

    return summaries


def batch_gene_annotations(
    entrez_ids: list[str],
    cache_path: Path,
) -> pd.DataFrame:
    entrez_ids = sorted(
        {
            str(x)
            for x in entrez_ids
            if str(x).strip()
        },
        key=lambda x: int(x),
    )

    cached = pd.DataFrame(
        columns=[
            "entrez_id",
            "gene",
            "gene_name",
            "taxid",
        ]
    )

    if cache_path.exists():
        cached = pd.read_csv(
            cache_path,
            dtype={"entrez_id": str},
        )

    known = set(
        cached["entrez_id"].astype(str)
    )

    missing = [
        gene_id
        for gene_id in entrez_ids
        if gene_id not in known
    ]

    new_rows = []

    if missing:
        try:
            response = requests.post(
                "https://mygene.info/v3/gene",
                data={
                    "ids": ",".join(missing),
                    "fields": "symbol,name,taxid",
                },
                timeout=60,
            )

            response.raise_for_status()

            payload = response.json()

            if isinstance(payload, dict):
                payload = [payload]

            for item in payload:
                gene_id = str(
                    item.get(
                        "_id",
                        item.get(
                            "entrezgene",
                            "",
                        ),
                    )
                )

                if not gene_id:
                    continue

                taxid = item.get(
                    "taxid",
                    "",
                )

                # KEGG hsa identifiers are human genes.
                # Keep non-human or unresolved records visible rather than
                # silently assigning a symbol.
                if taxid not in (
                    9606,
                    "9606",
                    "",
                    None,
                ):
                    symbol = ""
                    name = ""
                else:
                    symbol = str(
                        item.get(
                            "symbol",
                            "",
                        )
                    )
                    name = str(
                        item.get(
                            "name",
                            "",
                        )
                    )

                new_rows.append(
                    {
                        "entrez_id": gene_id,
                        "gene": symbol,
                        "gene_name": name,
                        "taxid": taxid,
                    }
                )

        except Exception as exc:
            print(
                "WARNING: MyGene.info mapping failed."
            )
            print(
                f"         {type(exc).__name__}: {exc}"
            )
            print(
                "         Unresolved genes will be shown by KEGG/Entrez ID."
            )

    if new_rows:
        cached = pd.concat(
            [
                cached,
                pd.DataFrame(
                    new_rows
                ),
            ],
            ignore_index=True,
        )

    cached = (
        cached
        .drop_duplicates(
            subset=[
                "entrez_id"
            ],
            keep="last",
        )
        .sort_values(
            "entrez_id",
            key=lambda series: pd.to_numeric(
                series,
                errors="coerce",
            ),
        )
        .reset_index(
            drop=True
        )
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cached.to_csv(
        cache_path,
        index=False,
    )

    return cached


def latex_escape(text: str) -> str:
    text = str(text)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    return "".join(
        replacements.get(
            char,
            char,
        )
        for char in text
    )


def gene_label(
    row: pd.Series,
) -> str:
    gene = str(
        row["gene"]
    ).strip()

    if gene:
        return gene

    entrez = str(
        row["entrez_id"]
    ).strip()

    if entrez:
        return f"hsa:{entrez}"

    return str(
        row["node_id"]
    )


def build_long_table(
    networks_dir: Path,
    summaries: dict[str, Path],
) -> pd.DataFrame:
    rows = []

    for network, summary_path in sorted(
        summaries.items()
    ):
        bnet_path = find_bnet(
            networks_dir,
            network,
        )

        model = load_bnet(
            bnet_path
        )

        fixed_nodes = {
            node
            for node in model.nodes
            if not model.rules[node].free_symbols
        }

        summary = pd.read_csv(
            summary_path
        )

        if summary.empty:
            continue

        n_variants = len(
            summary
        )

        for _, variant_row in summary.iterrows():
            variant_id = int(
                variant_row.get(
                    "id",
                    1,
                )
            )

            dominant_set = [
                item.strip()
                for item
                in str(
                    variant_row[
                        "dominant_set"
                    ]
                ).split(";")
                if item.strip()
            ]

            for node in dominant_set:
                normalized, entrez = (
                    normalize_hsa_id(
                        node
                    )
                )

                rows.append(
                    {
                        "network": network,
                        "network_label": DISPLAY_NAMES.get(
                            network,
                            network,
                        ),
                        "variant": variant_id,
                        "n_variants": n_variants,
                        "node_id": normalized,
                        "entrez_id": entrez,
                        "dv_type": (
                            "fixed"
                            if node in fixed_nodes
                            else "dynamic"
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def add_gene_annotations(
    long_df: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    result = long_df.merge(
        annotations[
            [
                "entrez_id",
                "gene",
                "gene_name",
            ]
        ],
        on="entrez_id",
        how="left",
    )

    result["gene"] = (
        result["gene"]
        .fillna("")
        .astype(str)
    )

    result["gene_name"] = (
        result["gene_name"]
        .fillna("")
        .astype(str)
    )

    result["label"] = result.apply(
        gene_label,
        axis=1,
    )

    return result


def membership_table(
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per network + gene, even if there are multiple minimum DV variants.
    """
    rows = []

    for (
        network,
        node_id,
    ), group in long_df.groupby(
        [
            "network",
            "node_id",
        ],
        sort=False,
    ):
        n_variants = int(
            group["n_variants"].max()
        )

        variants_present = sorted(
            group[
                "variant"
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        first = group.iloc[0]

        rows.append(
            {
                "network": network,
                "network_label": first[
                    "network_label"
                ],
                "node_id": node_id,
                "entrez_id": first[
                    "entrez_id"
                ],
                "gene": first[
                    "gene"
                ],
                "gene_name": first[
                    "gene_name"
                ],
                "label": first[
                    "label"
                ],
                "dv_type": first[
                    "dv_type"
                ],
                "variants_present": ";".join(
                    map(
                        str,
                        variants_present,
                    )
                ),
                "variants_present_count": len(
                    variants_present
                ),
                "variants_total": n_variants,
                "variant_fraction": (
                    len(
                        variants_present
                    )
                    / n_variants
                ),
                "mandatory_in_all_minimum_sets": (
                    len(
                        variants_present
                    )
                    == n_variants
                ),
            }
        )

    membership = pd.DataFrame(
        rows
    )

    if membership.empty:
        return membership

    # Recurrence is defined across different cancer networks, not across variants.
    recurrence = (
        membership.groupby(
            "node_id"
        )["network"]
        .nunique()
        .rename(
            "networks_count"
        )
    )

    network_lists = (
        membership.groupby(
            "node_id"
        )["network_label"]
        .apply(
            lambda values: ";".join(
                sorted(
                    set(
                        values
                    )
                )
            )
        )
        .rename(
            "networks"
        )
    )

    membership = membership.join(
        recurrence,
        on="node_id",
    )

    membership = membership.join(
        network_lists,
        on="node_id",
    )

    membership[
        "recurrent"
    ] = (
        membership[
            "networks_count"
        ]
        > 1
    )

    return membership


def recurrence_table(
    membership: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for node_id, group in membership.groupby(
        "node_id"
    ):
        first = group.iloc[0]

        fixed_networks = sorted(
            group.loc[
                group[
                    "dv_type"
                ]
                == "fixed",
                "network_label",
            ]
            .unique()
            .tolist()
        )

        dynamic_networks = sorted(
            group.loc[
                group[
                    "dv_type"
                ]
                == "dynamic",
                "network_label",
            ]
            .unique()
            .tolist()
        )

        rows.append(
            {
                "node_id": node_id,
                "entrez_id": first[
                    "entrez_id"
                ],
                "gene": first[
                    "gene"
                ],
                "gene_name": first[
                    "gene_name"
                ],
                "label": first[
                    "label"
                ],
                "networks_count": int(
                    group[
                        "network"
                    ].nunique()
                ),
                "networks": ";".join(
                    sorted(
                        group[
                            "network_label"
                        ]
                        .unique()
                        .tolist()
                    )
                ),
                "fixed_networks_count": len(
                    fixed_networks
                ),
                "fixed_networks": ";".join(
                    fixed_networks
                ),
                "dynamic_networks_count": len(
                    dynamic_networks
                ),
                "dynamic_networks": ";".join(
                    dynamic_networks
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "networks_count",
            "label",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def by_network_table(
    membership: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    order = {
        network: i
        for i, network in enumerate(
            NETWORK_ORDER
        )
    }

    for network, group in membership.groupby(
        "network"
    ):
        fixed = (
            group[
                group[
                    "dv_type"
                ]
                == "fixed"
            ]
            .sort_values(
                [
                    "networks_count",
                    "label",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        dynamic = (
            group[
                group[
                    "dv_type"
                ]
                == "dynamic"
            ]
            .sort_values(
                [
                    "networks_count",
                    "label",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        rows.append(
            {
                "network": network,
                "network_label": DISPLAY_NAMES.get(
                    network,
                    network,
                ),
                "n_minimum_dv_variants": int(
                    group[
                        "variants_total"
                    ].max()
                ),
                "fixed_dominant_count_union": len(
                    fixed
                ),
                "dynamic_dominant_count_union": len(
                    dynamic
                ),
                "dominant_count_union": len(
                    group
                ),
                "fixed_dominant_genes": "; ".join(
                    fixed[
                        "label"
                    ].tolist()
                ),
                "dynamic_dominant_genes": "; ".join(
                    dynamic[
                        "label"
                    ].tolist()
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "_order"
    ] = result[
        "network"
    ].map(
        order
    )

    return (
        result
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )


def matrix_table(
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """
    0 = absent
    1 = fixed dominant
    2 = dynamic dominant
    """
    labels = (
        membership[
            [
                "node_id",
                "label",
                "networks_count",
            ]
        ]
        .drop_duplicates(
            "node_id"
        )
        .sort_values(
            [
                "networks_count",
                "label",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    matrix = pd.DataFrame(
        0,
        index=labels[
            "node_id"
        ],
        columns=NETWORK_ORDER,
        dtype=int,
    )

    for _, row in membership.iterrows():
        value = (
            1
            if row[
                "dv_type"
            ]
            == "fixed"
            else 2
        )

        matrix.loc[
            row[
                "node_id"
            ],
            row[
                "network"
            ],
        ] = value

    matrix.insert(
        0,
        "gene",
        labels.set_index(
            "node_id"
        )[
            "label"
        ],
    )

    matrix.insert(
        1,
        "networks_count",
        labels.set_index(
            "node_id"
        )[
            "networks_count"
        ],
    )

    matrix = matrix.rename(
        columns=DISPLAY_NAMES
    )

    return matrix.reset_index(
        names="node_id"
    )


def plot_recurrence_matrix(
    matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    recurrent = matrix[
        matrix[
            "networks_count"
        ]
        > 1
    ].copy()

    if recurrent.empty:
        print(
            "No recurrent dominant genes were found; recurrence matrix not generated."
        )
        return

    network_columns = [
        DISPLAY_NAMES[
            network
        ]
        for network in NETWORK_ORDER
    ]

    values = recurrent[
        network_columns
    ].to_numpy(
        dtype=int
    )

    n_genes = len(
        recurrent
    )

    fig_height = max(
        4.5,
        0.32 * n_genes + 1.8,
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            fig_height,
        )
    )

    cmap = ListedColormap(
        [
            "#FFFFFF",
            "#DCEAF7",
            "#F2D27A",
        ]
    )

    ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=2,
    )

    ax.set_xticks(
        range(
            len(
                network_columns
            )
        )
    )

    ax.set_xticklabels(
        network_columns,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    labels = [
        (
            f"{row.gene} ({int(row.networks_count)})"
        )
        for row in recurrent.itertuples()
    ]

    ax.set_yticks(
        range(
            n_genes
        )
    )

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    ax.set_xlabel(
        "Cancer network"
    )

    ax.set_ylabel(
        "Recurrent dominant gene (number of networks)"
    )

    ax.set_title(
        "Dominant-gene recurrence across cancer networks"
    )

    for x in np.arange(
        -0.5,
        len(
            network_columns
        ),
        1,
    ):
        ax.axvline(
            x,
            linewidth=0.3,
            color="0.8",
        )

    for y in np.arange(
        -0.5,
        n_genes,
        1,
    ):
        ax.axhline(
            y,
            linewidth=0.3,
            color="0.8",
        )

    ax.legend(
        handles=[
            Patch(
                facecolor="#DCEAF7",
                edgecolor="0.5",
                label="Fixed dominant",
            ),
            Patch(
                facecolor="#F2D27A",
                edgecolor="0.5",
                label="Dynamic dominant",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.12,
        ),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def latex_gene_token(
    row: pd.Series,
) -> str:
    label = latex_escape(
        row[
            "label"
        ]
    )

    if bool(
        row[
            "recurrent"
        ]
    ):
        label = (
            r"\textbf{"
            + label
            + "}"
            + r"$^{("
            + str(
                int(
                    row[
                        "networks_count"
                    ]
                )
            )
            + r")}$"
        )

    else:
        label = (
            r"\textit{"
            + label
            + "}"
        )

    if row[
        "dv_type"
    ] == "dynamic":
        return (
            r"\colorbox{DVDynamic}{"
            + label
            + "}"
        )

    return (
        r"\colorbox{DVFixed}{"
        + label
        + "}"
    )


def generate_latex_table(
    membership: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        r"% Requires: \usepackage[table]{xcolor}",
        r"% Requires: \usepackage{longtable,array,booktabs}",
        r"\definecolor{DVFixed}{RGB}{220,234,247}",
        r"\definecolor{DVDynamic}{RGB}{245,214,132}",
        r"\setlength{\fboxsep}{1.2pt}",
        "",
        r"\begin{longtable}{p{0.22\textwidth} p{0.70\textwidth}}",
        r"\caption{Dominant genes identified across the cancer-related Boolean networks. "
        r"Blue denotes prescribed fixed dominant variables and gold denotes dynamic "
        r"dominant variables. Genes occurring in more than one cancer network are shown "
        r"in bold, with the superscript indicating the number of networks in which they occur.}"
        r"\label{tab:dominant-genes-all}\\",
        r"\toprule",
        r"\textbf{Network} & \textbf{Dominant genes} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Network} & \textbf{Dominant genes} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    order = {
        network: i
        for i, network in enumerate(
            NETWORK_ORDER
        )
    }

    networks = sorted(
        membership[
            "network"
        ].unique(),
        key=lambda network: order.get(
            network,
            999,
        ),
    )

    for network in networks:
        group = membership[
            membership[
                "network"
            ]
            == network
        ].copy()

        group = group.sort_values(
            [
                "dv_type",
                "networks_count",
                "label",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )

        tokens = [
            latex_gene_token(
                row
            )
            for _, row in group.iterrows()
        ]

        network_name = latex_escape(
            DISPLAY_NAMES.get(
                network,
                network,
            )
        )

        lines.append(
            network_name
            + " & "
            + ", ".join(
                tokens
            )
            + r" \\[2pt]"
        )

    lines.append(
        r"\end{longtable}"
    )

    output_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all dominant vertices, map KEGG/Entrez IDs to human gene "
            "symbols, quantify recurrence across cancer networks, and generate "
            "tables and a recurrence matrix."
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
        default="results/analysis/dominant_genes",
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = discover_dv_summaries(
        results_dir
    )

    if not summaries:
        raise FileNotFoundError(
            f"No dominant_vertices_summary.csv files found under {results_dir}"
        )

    long_df = build_long_table(
        networks_dir,
        summaries,
    )

    annotations = batch_gene_annotations(
        long_df[
            "entrez_id"
        ].tolist(),
        output_dir
        / "gene_annotation_cache.csv",
    )

    long_df = add_gene_annotations(
        long_df,
        annotations,
    )

    membership = membership_table(
        long_df
    )

    recurrence = recurrence_table(
        membership
    )

    by_network = by_network_table(
        membership
    )

    matrix = matrix_table(
        membership
    )

    long_df.to_csv(
        output_dir
        / "dominant_genes_variants_long.csv",
        index=False,
    )

    membership.to_csv(
        output_dir
        / "dominant_genes_membership.csv",
        index=False,
    )

    recurrence.to_csv(
        output_dir
        / "dominant_gene_recurrence.csv",
        index=False,
    )

    by_network.to_csv(
        output_dir
        / "dominant_genes_by_network.csv",
        index=False,
    )

    matrix.to_csv(
        output_dir
        / "dominant_gene_matrix.csv",
        index=False,
    )

    generate_latex_table(
        membership,
        output_dir
        / "dominant_genes_table.tex",
    )

    plot_recurrence_matrix(
        matrix,
        output_dir
        / "dominant_gene_recurrence_matrix.png",
    )

    recurrent = recurrence[
        recurrence[
            "networks_count"
        ]
        > 1
    ]

    print()
    print("=" * 78)
    print("DOMINANT GENE ANALYSIS")
    print("=" * 78)
    print(
        f"Networks with DV results: {membership['network'].nunique()}"
    )
    print(
        f"Unique dominant nodes: {membership['node_id'].nunique()}"
    )
    print(
        f"Network-gene memberships: {len(membership)}"
    )
    print(
        f"Recurrent dominant genes (>1 network): {len(recurrent)}"
    )

    print()
    print("Most recurrent dominant genes:")
    print(
        recurrence[
            [
                "label",
                "networks_count",
                "fixed_networks_count",
                "dynamic_networks_count",
                "networks",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print("Outputs:")
    for path in sorted(
        output_dir.iterdir()
    ):
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()
