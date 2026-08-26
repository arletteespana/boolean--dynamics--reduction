from pathlib import Path
import shutil
import argparse


def generate_original(input_bnet, output_bnet):
    input_path = Path(input_bnet)
    output_path = Path(output_bnet)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() != ".bnet":
        raise ValueError(f"Input file must be a .bnet file: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(input_path, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate the original Boolean network used as baseline."
    )

    parser.add_argument(
        "input",
        help="Path to the original .bnet file."
    )

    parser.add_argument(
        "output",
        help="Path where the baseline .bnet file will be written."
    )

    args = parser.parse_args()

    output_path = generate_original(args.input, args.output)

    print(f"Original model written to: {output_path}")


if __name__ == "__main__":
    main()
