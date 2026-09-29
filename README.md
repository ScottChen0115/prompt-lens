# Bilingual request-quality model

Active model: XLM-RoBERTa-base with five issue outputs, **278,047,493 parameters (0.278B)**.
English and Chinese; the user's final model budget is 0.1B–0.3B parameters.
This is an encoder classifier; it does not generate free-form rewrites. Its five provisional
outputs concern unclear goals, missing information, unresolved references, conflicting
constraints and unclear deliverables. Suggestions will require reviewed templates or a
separately evaluated generator. No reliable overall quality score is available yet.

## Use the local window

Build the native app with `bash scripts/build_interface.sh`, then open the generated `Prompt Lens.app`.
Keep it inside this project folder so it can find the existing Python environment and saved model.
Choose English or 中文, enter your request, optionally add context, and click Analyze (or ⌘Return).
The five bars are experimental model signals, not validated quality grades. Expand the self-review
questions for general wording guidance. Clear removes the current input and results.

Everything runs locally; there is no cloud call or saved prompt history. The app loads the saved
model for inference only, with gradients disabled. Combined input is limited to 256 tokens and
is never silently truncated. Close the window to quit the app and its worker.

Original interface verification (2026-09-28): 21 tests plus native English/Chinese interaction and visual checks.
That interface-only phase preserved the original checkpoint. The authorized 2026-09-29 training update is described below.
See the local interface verification report after running the project checks.
Rebuild with `bash scripts/build_interface.sh` (Apple Silicon, macOS 14+).

## Supplied-data training — 2026-09-29

The user authorized resuming local training with the two supplied datasets and updating Prompt Lens.
The 0.278B model completed three passes over 11,999 usable training examples. Another 2,000
examples selected the checkpoint and thresholds; 2,000 fitted calibration temperatures.
Both datasets contain machine-generated labels, not independent human ground truth. One overlong
training example was quarantined. Unknown labels are masked individually, never treated as negatives.

The selected model passed validation replacement gates against the old model and a character TF-IDF
baseline. Final evaluation and application activation evidence are recorded in the
[run report](outputs/runs/supplied-20260929T121031Z/Report.md).
The [dataset audit](outputs/dataset-audit/Audit Report.md)
records substantial template similarity across development splits. Synthetic scores cannot establish
real-world quality or improvements to GPT-6 Sol outcomes. No cloud service or deployment is used.

Reproduction entry points: `scripts/audit_supplied_data.py`, `scripts/train_supplied.py`,
`scripts/evaluate_frozen_supplied.py`. Final tests are reserved until the checkpoint is frozen;
do not tune on their results or repeatedly reuse them as an untouched benchmark.
The application reads `outputs/development/latest-bilingual.json`; the prior pointer is retained
in the new run when activation succeeds.

## Historical bilingual fixture training

```sh
bash scripts/run.sh python scripts/download_bilingual.py
bash scripts/run.sh python scripts/train_bilingual.py --device mps
bash scripts/run.sh python -m pytest -q
```

Use `--device cpu` when MPS is unavailable. No paid services are used.
Settings: `configs/training-bilingual.json`; pinned checkpoint: `configs/checkpoint-bilingual.json`.
The runner enforces 100M–300M **total** parameters, not just trainable adapter weights.
It fine-tunes every parameter using a five-output binary-cross-entropy objective, microbatch 4,
accumulation 4 and learning rate 2e-5. The effective batch size is 16.

Each run writes a model, tokenizer, policy, source hashes, data/split metadata and separate
English/Chinese diagnostics. `outputs/development/latest-bilingual.json` points only to a
successfully completed run. The fixture dataset is deliberately artificial and unreviewed;
translations stay together, calibration/test sets remain unused, and exported quality scores
and thresholds remain disabled. Traditional Chinese and mixed-language tokenizer probes
exercise tokenization only; training fixtures use English and Simplified Chinese.

Research data, bilingual accuracy, useful feedback and downstream GPT-6 Sol improvements
remain unverified. The previous BERT-Tiny executable and artifacts below are historical
infrastructure checks, not the active product architecture.

## Environment

This project uses Python 3.13.15, a local `.venv`, and a dependency lock in `uv.lock`.
Caches are local to `.cache/`. The global Python installation is not modified.

From this project folder, recreate the environment with:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked --python 3.13.15
```

Run the offline infrastructure check:

```sh
bash scripts/run.sh python scripts/check_environment.py
```

The wrapper works from macOS Terminal's default zsh and sets project-local cache paths.
Use `bash scripts/run.sh python ...` for other project Python commands.

The check creates 20 deliberately artificial fixture rows in memory, exercises dataset ingestion,
a TF-IDF baseline, tokenization, a randomly initialized two-layer BERT, optimizer updates,
safe checkpoint save/reload, and plotting. It checks CPU and MPS when available.
It downloads no dataset or model. Fixture labels and losses are infrastructure checks,
not valid misunderstanding examples or evidence of model quality.

Results go to `outputs/environment/environment-report.json` and `plot-smoke.png`.
MPS availability can differ inside restricted execution environments; a successful CPU run
does not establish GPU readiness. Inspect the report's device results.

## Files and progress

- `pyproject.toml`: dependency requirements; `uv.lock`: resolved versions and package hashes.
- `inputs/`: immutable source material. Do not store private material without approval.
- `data/processed/`: normalized supplied data and split manifests.
- `configs/`, `src/`, `tests/`: model configuration, implementation and verification.
- `outputs/`: reports and saved model artifacts.
- Project knowledge is maintained separately from this public source repository.

No research model has been trained. The exact target-assistant configuration, task domain, reviewed quality-label protocol and data
permissions still need to be fixed before research data collection.

Setup references: [Transformers installation](https://huggingface.co/docs/transformers/installation)
and [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Historical English-only development training

A pretrained BERT-Tiny development trainer is now available:

```sh
bash scripts/run.sh python scripts/train_development.py --device cpu --epochs 4
# Use --device mps when the execution environment exposes the Apple GPU.
bash scripts/run.sh python -m pytest -q
```

The downloaded base revision is pinned in `configs/checkpoint.json`, with files in the
project-local Hugging Face cache. Each run saves its own fixture dataset, source/lock hashes,
validation diagnostics, temperature-scaling exercise, and safely serialized model/tokenizer.
`outputs/development/latest.json` points to the latest completed run. Research data is not
accepted by this executable. Research training has not started: target-assistant outcomes and
independent human review remain required. The generated model policy returns no valid risk
probability and has no action threshold. See `outputs/development/Research Readiness.md`.
