from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


METHODS = {
    "original": {
        "script": "original.py",
        "output": "original.bnet",
        "output_is_directory": False,
    },
    "dominant_vertices": {
        "script": "dominant_vertices.py",
        "output": "dominant_vertices",
        "output_is_directory": True,
    },
    "node_elimination": {
        "script": "node_elimination.py",
        "output": "node_elimination.bnet",
        "output_is_directory": False,
    },
    "two_step": {
        "script": "two_step.py",
        "output": "two_step.bnet",
        "output_is_directory": False,
    },
    "leaf_node_removal": {
        "script": "leaf_node_removal.py",
        "output": "leaf_node_removal.bnet",
        "output_is_directory": False,
    },
}


def run_command(command: list[str], log_path: Path) -> None:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{' '.join(command)}\n\n"
            f"See log: {log_path}"
        )


def run_network(
    network_path: Path,
    methods_dir: Path,
    results_dir: Path,
) -> None:
    network_name = network_path.stem
    network_results = results_dir / network_name
    logs_dir = network_results / "logs"

    network_results.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print(f"Network: {network_name}")
    print("=" * 70)

    for method_name, config in METHODS.items():
        script_path = methods_dir / config["script"]

        if not script_path.exists():
            raise FileNotFoundError(
                f"Method script not found: {script_path}"
            )

        output_path = network_results / config["output"]

        if config["output_is_directory"]:
            output_path.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(script_path),
            str(network_path),
            str(output_path),
        ]

        log_path = logs_dir / f"{method_name}.log"

        print(f"[RUN] {method_name}")

        run_command(
            command,
            log_path,
        )

        print(f"[ OK] {method_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all Boolean-network reduction methods for every .bnet "
            "file in the networks directory."
        )
    )

    parser.add_argument(
        "--networks-dir",
        default="networks",
        help="Directory containing the input .bnet files.",
    )

    parser.add_argument(
        "--methods-dir",
        default="src/methods",
        help="Directory containing the method scripts.",
    )

    parser.add_argument(
        "--results-dir",
        default="results/models",
        help="Directory where generated models will be stored.",
    )

    args = parser.parse_args()

    networks_dir = Path(args.networks_dir)
    methods_dir = Path(args.methods_dir)
    results_dir = Path(args.results_dir)

    if not networks_dir.exists():
        raise FileNotFoundError(
            f"Networks directory not found: {networks_dir}"
        )

    bnet_files = sorted(networks_dir.glob("*.bnet"))

    if not bnet_files:
        raise FileNotFoundError(
            f"No .bnet files found in: {networks_dir}"
        )

    print(f"Networks found: {len(bnet_files)}")
    print(f"Methods per network: {len(METHODS)}")
    print(f"Results directory: {results_dir}")

    completed = 0

    for network_path in bnet_files:
        run_network(
            network_path,
            methods_dir,
            results_dir,
        )
        completed += 1

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Networks processed: {completed}")
    print(f"Results written to: {results_dir}")


if __name__ == "__main__":
    main()
