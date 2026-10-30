# SkinDiseaseAI

TensorFlow/Keras EfficientNetB0 research prototype with a Streamlit interface. Outputs are experimental and **are not medical diagnoses**. The model only predicts among seven HAM10000 categories and is not a substitute for clinical assessment.

## Data sources and label mapping

HAM10000 remains the existing source dataset. Set its location with `HAM_DATASET_ROOT` or `dataset_root` in `config.json`. Set the ISIC 2019 folder with `ISIC_DATASET_ROOT` or the `--isic-root` argument; no machine-specific ISIC path is required in source code or configuration.

The ISIC 2019 mapping uses the one-hot diagnosis columns in `ISIC_2019_Training_GroundTruth.csv`:

| ISIC diagnosis | HAM10000 class | Action |
| --- | --- | --- |
| `AK` | `akiec` | Include |
| `BCC` | `bcc` | Include |
| `BKL` | `bkl` | Include |
| `DF` | `df` | Include |
| `MEL` | `mel` | Include |
| `NV` | `nv` | Include |
| `VASC` | `vasc` | Include |
| `SCC` | — | Exclude; no compatible class |
| `UNK`, missing, malformed, or multi-hot labels | — | Exclude |

ISIC 2019 metadata provides `lesion_id` but this collection has no patient ID column. Splits group by patient if a patient ID field is present, otherwise by lesion ID. The inspector reports the metadata fields and label counts before training.

`AK` maps from the source diagnosis column named `AK` to HAM10000's broader `akiec` category. ISIC examples therefore represent the actinic-keratosis part of that category; no additional diagnosis is inferred.

## Inspection before training

Run the inspection command first. It verifies metadata/image coverage, decodes images, hashes files for exact duplicates, computes perceptual hashes for near-duplicate candidates, and checks those against every image in the existing HAM split manifest. It writes `outputs/isic_dataset_report.json`. Review missing and invalid files, exclusions, duplicate candidates, and test overlap before training. ISIC examples matching HAM images or other ISIC examples are excluded from the added training data. The original HAM test rows are kept in the test partition; non-test HAM duplicates are omitted from the fine-tuning run where needed.

The collection currently identified in the workspace is ISIC 2019 Training: 25,331 ground-truth rows and metadata rows, with 25,331 images in the eight diagnosis folders. Source counts are `NV` 12,875, `MEL` 4,522, `BCC` 3,323, `BKL` 2,624, `AK` 867, `DF` 239, `VASC` 253, and `SCC` 628. These are metadata counts; use the generated inspection report for verified image validity and duplicate counts.

## Setup (PowerShell)

Run commands from the project directory. Python 3.10–3.12 and the dependencies in `requirements.txt` are required.

```powershell
Set-Location 'D:\DOWNLOADS D\SkinDiseaseAI\SkinDiseaseAI'
py -3.12 -m venv .venv-isic
.\.venv-isic\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -c "import tensorflow as tf, numpy, pandas, PIL, sklearn; print('TensorFlow', tf.__version__, '| GPUs', tf.config.list_physical_devices('GPU'))"
```

If PowerShell blocks environment activation, use `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell window.

## Inspect, train, evaluate, launch

Set the ISIC directory for this PowerShell session, then inspect before training:

```powershell
$env:ISIC_DATASET_ROOT = 'D:\DOWNLOADS D\archive (1)'
Test-Path $env:ISIC_DATASET_ROOT
Test-Path 'D:\DOWNLOADS D\archive\HAM10000_metadata.csv'
python -m src.isic --isic-root $env:ISIC_DATASET_ROOT
Get-Content .\outputs\isic_dataset_report.json
```

Start fine-tuning only after reviewing the report:

```powershell
python -m src.train --isic-root $env:ISIC_DATASET_ROOT --batch-size 16 --epochs 10 --learning-rate 0.00001 --inspection-acknowledged
```

The original `models/best_model.keras` is loaded as the starting checkpoint and is never overwritten. The new checkpoint is `models/isic_finetuned.keras`; the existing HAM `outputs/splits.csv` is retained and its test images are not used for optimization. Training uses streamed image batches, model-side training-only augmentation, class weights, reproducible seeds, early stopping, learning-rate reduction, and best-validation checkpointing. Add `--resume` to continue from the new checkpoint.

Compare the original and fine-tuned checkpoints on the same saved test rows, then start Streamlit:

```powershell
python -m src.evaluate --new-model models/isic_finetuned.keras
streamlit run app.py
```

The app keeps its existing navigation, lets you choose either checkpoint, displays all seven class scores, and labels the output as experimental rather than diagnostic.

## Remote chat model

The workspace includes a separate OpenAI-compatible chat client for the Fable model exposed by the configured router. This is a chat model and does not replace the local seven-class skin lesion classifier. Install the added `openai` dependency, then set the API key issued by the router in `OPENAI_API_KEY` (PowerShell: `$env:OPENAI_API_KEY = 'your-router-key'`). The default base URL is `https://rsjwdt9.abc-tunnel.us/v1` and the default model name is `Fable 5.1`; override them with `OPENAI_BASE_URL` and `OPENAI_MODEL` if your router uses different values. Run `python -m src.remote_model "Hello"` from this project directory.

## Reports and checkpoints

- `outputs/isic_dataset_report.json`: ISIC discovery, metadata, diagnosis counts, invalid files, exact/near duplicate candidates, and overlap with the saved HAM split.
- `outputs/combined_dataset_report.json` and `outputs/combined_splits.csv`: training exclusions and combined split counts; original HAM test membership is checked before training.
- `outputs/evaluation/model_comparison.json`: old/new metrics and deltas on the same untouched test set.
- `outputs/evaluation/*classification_report.csv`, `*confusion_matrix.csv`, and `*.png`: precision, recall, F1, confusion matrices, and plots.
- `models/isic_finetuned.keras`: separate fine-tuned model.

Per-class support below 20 is flagged as low support and should be interpreted cautiously. Do not claim improvement unless the saved test-set comparison demonstrates it.
