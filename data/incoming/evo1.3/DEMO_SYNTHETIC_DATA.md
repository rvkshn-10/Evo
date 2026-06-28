# Synthetic Phase B demo data (mentor preview)

**Not real FCUSD drill data.** These files exist so mentors can see the Evo 1.3
training pipeline unblock and run end-to-end while real drills are still in
progress.

| File | Purpose |
|------|---------|
| `real_outcomes.json` | Fabricated success % and evacuation times (`_meta.not_for_production: true`) |
| `peoplesense/drill-*.xml` | Drill-timestamp occupancy snapshots aligned to each outcome row |
| `config/monitoring_locations.json` | Demo GGV2 Group IDs + `coords_confirmed: true` on three Pi sites |

## Run preflight

```bash
python model_training/evo1_3/train_evo1_3.py \
  --peoplesense-dir data/incoming/evo1.3/peoplesense \
  --real-outcomes data/incoming/evo1.3/real_outcomes.json \
  --coords-confirmed \
  --preflight-only
```

## Train (demo) — completed 2026-06-28

```bash
python model_training/evo1_3/train_evo1_3.py \
  --peoplesense-dir data/incoming/evo1.3/peoplesense \
  --real-outcomes data/incoming/evo1.3/real_outcomes.json \
  --coords-confirmed \
  --output-dir artifacts/evo1.3

cp -R artifacts/evo1.3/* models/evo1.3/
```

**Result:** `complete_research_demo` · `DATA_CEILING` · `keep_evo1.2_hybrid`  
See `model_training/evo1_3/PHASE_B_STATUS.md` for metrics.

Tell mentors the numbers are illustrative — validation gates may still fail until real labeled drill data replaces the synthetic set.
