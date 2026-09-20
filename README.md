# FX Model Architecture Analyzer

Analyze PyTorch and Hugging Face text models through FX graphs captured by `torch.compile`.

## Features

- Counts module layers, function calls, and method calls
- Counts linear layers/operations
- Detects module and functional activation functions
- Reports total and trainable parameters
- Loads Hugging Face Hub models or local `save_pretrained` directories
- Optionally saves the report as JSON

## Project structure

```text
python/model_analysis/
├── execute.py
├── modelload.py
├── analyze.py
├── activations.py
├── fx_graph.py
└── model_report.py
```

The original RNN proof-of-concept remains in `rnn_model.py` and `trial.py`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

```powershell
python .\python\model_analysis\execute.py distilbert-base-uncased
```

With representative input text:

```powershell
python .\python\model_analysis\execute.py distilbert-base-uncased --text "This movie was excellent."
```

Save a JSON report:

```powershell
python .\python\model_analysis\execute.py distilbert-base-uncased --json report.json
```

Analyze a downloaded model:

```powershell
python .\python\model_analysis\execute.py "C:\models\my_model" --local-files-only
```

Remote model code is disabled by default. Use `--trust-remote-code` only for repositories you trust.

## Notes

Compiled graphs can lower modules into function calls, so a compiled report may show fewer `call_module` layers and more `call_function` operations. The report describes the execution path exercised by the supplied example input.
