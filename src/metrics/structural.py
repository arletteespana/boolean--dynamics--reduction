from __future__ import annotations

from .utils import ModelRecord, load_bnet


def compute_structural_metrics(
    record: ModelRecord,
    original_variables: int,
) -> dict[str, object]:
    model = load_bnet(record.path)
    serialized_dimension = len(model.nodes)

    if record.method == "dominant_vertices":
        retained_variables = int(
            record.metadata.get(
                "dominant_set_size",
                serialized_dimension,
            )
        )

        state_dimension = int(
            record.metadata.get(
                "state_dimension",
                serialized_dimension,
            )
        )

        recurrence_length = record.metadata.get(
            "recurrence_length",
            "",
        )
        depth = record.metadata.get("depth", "")
        dominant_set = record.metadata.get(
            "dominant_set",
            "",
        )

        if state_dimension != serialized_dimension:
            raise ValueError(
                f"DV metadata mismatch for {record.path}: "
                f"summary state_dimension={state_dimension}, "
                f".bnet variables={serialized_dimension}"
            )
    else:
        retained_variables = serialized_dimension
        state_dimension = serialized_dimension
        recurrence_length = ""
        depth = ""
        dominant_set = ""

    retained_fraction = (
        retained_variables / original_variables
        if original_variables
        else 0.0
    )

    eliminated_fraction = 1.0 - retained_fraction

    state_space_size = 2 ** state_dimension
    original_state_space_size = 2 ** original_variables

    state_space_ratio = 2.0 ** (
        state_dimension - original_variables
    )

    return {
        "network": record.network,
        "method": record.method,
        "variant": record.variant,
        "model_file": str(record.path),
        "original_variables": original_variables,
        "retained_variables": retained_variables,
        "state_dimension": state_dimension,
        "retained_fraction": retained_fraction,
        "eliminated_fraction": eliminated_fraction,
        "state_space_size": str(state_space_size),
        "original_state_space_size": str(original_state_space_size),
        "state_space_ratio": state_space_ratio,
        "log2_state_space_ratio": (
            state_dimension - original_variables
        ),
        "effective_state_space_reduction": (
            state_dimension < original_variables
        ),
        "recurrence_length": recurrence_length,
        "depth": depth,
        "dominant_set": dominant_set,
    }
