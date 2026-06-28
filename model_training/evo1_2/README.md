# Evo 1.2 live-hazard experiment

Evo 1.2 compares a dual-head MLP, two LightGBM regression heads, and their OOF
average. Labeled reference rows receive a deterministic category-compatible
FCUSD monitoring-spot proxy and its nearest real hazard within 250 km. The live
hazard seed is pseudo-labeled with production-distance k-NN for training only and
is never counted in outcome metrics.

```bash
python scripts/build_hazard_training_seed.py
python model_training/evo1_2/train_evo1_2.py \
  --data data/processed/evacuation_reference.json \
  --hazard-seed data/processed/hazard_live_seed.json \
  --output-dir artifacts/evo1.2
```

Use `--strict` in CI. Failed gates still produce a complete diagnostic artifact
bundle and return non-zero so the model cannot be promoted accidentally.
