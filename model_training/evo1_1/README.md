# Evo 1.1 dual-head experiment

Evo 1.1 tests whether the reference data can support genuinely predictive
evacuation-success regression. It uses grouped five-fold cross-validation,
eight hyperparameter combinations, dual time/success towers, masked success
loss, engineered features, and grouped synthetic severity variants.

```bash
python model_training/evo1_1/train_evo1_1.py \
  --data data/processed/evacuation_reference.json \
  --output-dir artifacts/evo1.1
```

Use `--strict` in CI. A failed success R² gate intentionally produces honest
artifacts and a time-only recommendation before returning a non-zero status.
