from torch import nn


ACTIVATION_NAMES = (
    "CELU",
    "ELU",
    "GELU",
    "GLU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "LeakyReLU",
    "LogSigmoid",
    "LogSoftmax",
    "Mish",
    "PReLU",
    "ReLU",
    "ReLU6",
    "RReLU",
    "SELU",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "Threshold",
)


ACTIVATION_MODULES = tuple(
    getattr(nn, name)
    for name in ACTIVATION_NAMES
    if hasattr(nn, name)
)


ACTIVATION_FUNCTIONS = {
    name.lower(): name
    for name in ACTIVATION_NAMES
}


ACTIVATION_FUNCTIONS.update(
    {
        "leaky_relu": "LeakyReLU",
        "log_softmax": "LogSoftmax",
        "relu6": "ReLU6",
        "hardshrink": "Hardshrink",
        "hardsigmoid": "Hardsigmoid",
        "hardswish": "Hardswish",
        "hardtanh": "Hardtanh",
        "logsigmoid": "LogSigmoid",
        "softmax": "Softmax",
        "softmin": "Softmin",
        "softplus": "Softplus",
        "softshrink": "Softshrink",
        "tanhshrink": "Tanhshrink",
    }
)


def identify_activation(name: str) -> str | None:
    normalized = name.lower()

    normalized = normalized.replace(
        "torch.nn.functional.",
        "",
    )

    normalized = normalized.replace(
        "torch.",
        "",
    )

    parts = normalized.split(".")

    for part in reversed(parts):
        if part in ACTIVATION_FUNCTIONS:
            return ACTIVATION_FUNCTIONS[part]

    return ACTIVATION_FUNCTIONS.get(normalized)