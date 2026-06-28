# Evo 1.0 training and export

This workflow trains the small multi-output PyTorch MLP, exports ONNX and
OpenVINO IR, and runs the acceptance checks from the Evo 1.0 brief.

## Colab

Open `Evo_1_0_Training.ipynb` in Colab and run all cells. The notebook clones
the repository when needed, installs the isolated training dependencies, and
writes artifacts to `artifacts/evo1.0/`.

## Command line

```bash
python -m pip install -r model_training/evo1/requirements-colab.txt
python model_training/evo1/train_evo1.py \
  --data data/processed/evacuation_reference.json \
  --output-dir artifacts/evo1.0
```

Add `--strict` in CI to fail if any acceptance threshold is missed.

Generated files:

- `evo1.0.pt`: PyTorch checkpoint
- `evo1.0.onnx`: dynamic-batch ONNX model
- `evo1.0.xml` and `evo1.0.bin`: OpenVINO IR
- `feature_schema.json`: preprocessing and output contract
- `data_audit.json`: duplicate, missing-label, and limitation audit
- `validation_report.json`: model metrics, parity checks, and CPU latency

## Important data limitations

The source workbook's `All Data` sheet duplicates every source row, so the
workflow deduplicates before splitting. Office Building rows have no density or
success target; Stadium rows have no success target. Their time targets still
participate through a masked multi-output loss, but success MAE can only be
validated on Train Station data. Severity and hazard magnitude are neutral
inputs because the reference dataset contains neither value nor a defensible
target relationship for them.
