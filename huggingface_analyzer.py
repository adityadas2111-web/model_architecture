"""Load a text model from Hugging Face and analyze its compiled FX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fx_model_analyzer import ModelReport, analyze_model


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def analyze_huggingface_model(
    model_name_or_path: str,
    *,
    text: str = "This is an example input for model architecture analysis.",
    revision: str | None = None,
    device: str = "auto",
    max_length: int = 32,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    fullgraph: bool = False,
) -> ModelReport:
    """Download/load a Hugging Face text model and analyze one inference graph.

    ``model_name_or_path`` may be a Hub ID such as ``distilbert-base-uncased``
    or a local directory previously created with ``save_pretrained``.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face support requires transformers. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    load_options: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        load_options["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_options)
    model = AutoModel.from_pretrained(model_name_or_path, **load_options)
    target_device = _device(device)
    model.to(target_device).eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {
        name: value.to(target_device) if isinstance(value, torch.Tensor) else value
        for name, value in encoded.items()
    }

    return analyze_model(
        model,
        example_args=(),
        example_kwargs=inputs,
        use_compile=True,
        fullgraph=fullgraph,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a Hugging Face text model using a compiled FX graph."
    )
    parser.add_argument(
        "model",
        help="Hugging Face Hub model ID or local save_pretrained directory",
    )
    parser.add_argument(
        "--text",
        default="This is an example input for model architecture analysis.",
        help="Representative text used to capture the executed graph",
    )
    parser.add_argument("--revision", help="Optional Hub branch, tag, or commit")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, and so on"
    )
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow code shipped by the model repository (use only when trusted)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download files; load only from the local cache/path",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Fail on a torch.compile graph break instead of capturing split graphs",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="Also save the report as JSON at this path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze_huggingface_model(
        args.model,
        text=args.text,
        revision=args.revision,
        device=args.device,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        fullgraph=args.fullgraph,
    )
    print(report)
    if args.json_path:
        args.json_path.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"JSON report written to {args.json_path}")


if __name__ == "__main__":
    main()
