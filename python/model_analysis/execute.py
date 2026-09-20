import argparse
import json
from pathlib import Path

from modelload import analyze_huggingface_model


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a Hugging Face model using PyTorch FX."
    )

    parser.add_argument(
        "model",
        help="Hugging Face model ID or local model directory",
    )

    parser.add_argument(
        "--text",
        default="This is an example input.",
        help="Representative input text",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda or cuda:0",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=32,
        help="Maximum number of input tokens",
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally downloaded files",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow code from trusted model repositories",
    )

    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Fail when torch.compile encounters a graph break",
    )

    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="Optional JSON output path",
    )

    return parser


def main() -> None:
    parser = create_argument_parser()
    arguments = parser.parse_args()

    try:
        report = analyze_huggingface_model(
            model_name_or_path=arguments.model,
            text=arguments.text,
            device=arguments.device,
            max_length=arguments.max_length,
            trust_remote_code=arguments.trust_remote_code,
            local_files_only=arguments.local_files_only,
            fullgraph=arguments.fullgraph,
        )

    except Exception as error:
        print("\nAnalysis failed:")
        print(error)
        raise SystemExit(1) from error

    print()
    print("=" * 70)
    print("MODEL ARCHITECTURE REPORT")
    print("=" * 70)
    print(report)

    if arguments.json_path is not None:
        arguments.json_path.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"\nJSON report saved to: {arguments.json_path}")


if __name__ == "__main__":
    main()