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
    }
)


def identify_activation(name: str) -> str | None:
    normalized = (
        name.lower()
        .replace("torch.nn.functional.", "")
        .replace("torch.", "")
    )

    for part in reversed(normalized.split(".")):
        activation = ACTIVATION_FUNCTIONS.get(part)

        if activation is not None:
            return activation

    return None