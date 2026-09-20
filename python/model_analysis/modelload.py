from typing import Any

import torch

from analyze import analyze_model


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    selected_device = torch.device(device_name)

    if (
        selected_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    return selected_device


def load_huggingface_model(
    model_name_or_path: str,
    revision: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
):
    try:
        from transformers import AutoModel, AutoTokenizer

    except ImportError as error:
        raise RuntimeError(
            "Run: python -m pip install torch transformers safetensors"
        ) from error

    load_options: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }

    if revision is not None:
        load_options["revision"] = revision

    print(f"Loading tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        **load_options,
    )

    print(f"Loading model: {model_name_or_path}")

    model = AutoModel.from_pretrained(
        model_name_or_path,
        **load_options,
    )

    return tokenizer, model


def prepare_text_inputs(
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 32,
) -> dict[str, Any]:
    encoded_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    model_inputs = {}

    for name, value in encoded_inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[name] = value.to(device)
        else:
            model_inputs[name] = value

    return model_inputs


def analyze_huggingface_model(
    model_name_or_path: str,
    text: str = "This is an example input.",
    revision: str | None = None,
    device: str = "auto",
    max_length: int = 32,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    fullgraph: bool = False,
):
    tokenizer, model = load_huggingface_model(
        model_name_or_path=model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    target_device = select_device(device)

    print(f"Using device: {target_device}")

    model.to(target_device)
    model.eval()

    model_inputs = prepare_text_inputs(
        tokenizer=tokenizer,
        text=text,
        device=target_device,
        max_length=max_length,
    )

    print("Capturing compiled FX graphs...")

    return analyze_model(
        model=model,
        example_args=(),
        example_kwargs=model_inputs,
        use_compile=True,
        fullgraph=fullgraph,
    )