# FX Model Architecture Analyzer

Analyze a PyTorch model and report how many layers, linear operations,
function calls, activation functions and parameters it uses. Counts come from
the model's executable FX graph, so the analyzer sees both `nn.Module` layers
and functional calls such as `torch.nn.functional.relu`.

## What it reports

- Total executable graph operations
- Total module layers
- Module, function and method call counts
- Linear layers/operations
- Every detected activation type and its count
- Layer and function breakdowns
- Total and trainable parameter counts

## Run the demo

```bash
python -m pip install -r requirements.txt
python demo.py
```

## Analyze a Hugging Face model

Pass either a model ID from the Hugging Face Hub or a downloaded local model
directory:

```bash
python huggingface_analyzer.py distilbert-base-uncased
```

Use representative text and optionally save a machine-readable report:

```bash
python huggingface_analyzer.py distilbert-base-uncased \
  --text "FX graphs make model structure inspectable." \
  --json report.json
```

To analyze a downloaded model without network access:

```bash
python huggingface_analyzer.py ./my_downloaded_model --local-files-only
```

The command uses `AutoTokenizer` and `AutoModel`, runs one representative
inference through `torch.compile`, and analyzes the captured FX graph. Remote
model code is disabled by default. Only add `--trust-remote-code` for a model
repository whose code you have reviewed and trust.

The current automatic loader targets text models. Image, audio, multimodal, or
custom models can still use `analyze_model` directly by supplying their real
example tensors through `example_args` or `example_kwargs`.

The same functionality is importable from Python:

```python
from huggingface_analyzer import analyze_huggingface_model

report = analyze_huggingface_model("distilbert-base-uncased")
print(report)
```

## Use it with any PyTorch model

For models that can be symbolically traced:

```python
from fx_model_analyzer import analyze_model

report = analyze_model(model)
print(report)
print(report.to_dict())
```

For input-dependent models, provide representative inputs and capture the FX
graph used by `torch.compile`:

```python
report = analyze_model(
    model.eval(),
    example_args=(sample_input,),
    use_compile=True,
)
print(report)
```

`use_compile=True` runs one inference call under `torch.no_grad()` to capture
the graph. The supplied inputs must therefore have the shapes and dtypes the
model expects. Set `fullgraph=True` if graph breaks should be treated as an
error instead of producing multiple graphs.

## Meaning of "layer"

The report counts a layer when an FX `call_module` node executes it. Total graph
operations also include `call_function` and `call_method` nodes. Input
placeholders, stored attributes and the output node are excluded. This avoids
counting registered modules that are never used in `forward`.

No graph system can statically describe every possible Python program. Models
with data-dependent branches, unsupported custom operators or graph breaks may
need representative inputs and compiled capture; the report states how many FX
graphs were captured so split graphs remain visible.
