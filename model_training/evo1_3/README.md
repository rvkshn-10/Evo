# Evo 1.3 data-first training

Evo 1.3 intentionally refuses to train until timestamped **PeopleSense** occupancy
samples, real non-transit evacuation outcomes, and site coordinate confirmation
are provided. It does not attempt to clear the Evo 1.2 data ceiling by increasing
network size or epoch count.

## Data flow (important)

```
Raspberry Pi / edge  →  PeopleSense database  →  our app (GET occupancy API)
```

We **never** read occupancy directly from Pis. Pis (and other sources) write into
PeopleSense; we consume PeopleSense exports or API snapshots for both the live
dashboard and Evo 1.3 training.

## Required inputs

1. **PeopleSense occupancy samples** for training — one or more files in
   `data/incoming/evo1.3/peoplesense/`:
   - `.xml` OccupancyXML exports from PeopleSense, or
   - `.json` event envelopes containing `OccupancyXML`
   - Each sample: timestamp + Zone `id`, `Count`, `Density`, `Volatility`
   - **Vista del Lago** (`spot_id: vista-del-lago`) is mandatory
   - Live dashboard uses `GET .../v1/occupancy?filter=ALL`; training needs
     **timestamped samples aligned to drill times** (export or snapshot from
     PeopleSense at drill time — not the static GET cache alone)

2. **Real drill outcomes** — `data/incoming/evo1.3/real_outcomes.json` matching
   `real_outcomes.template.json` (measured success % and evacuation time).

3. **Confirmed site coordinates** in `config/monitoring_locations.json` for
   `vista-del-lago`, `folsom-high`, and `cordova-park` (must match how spots are
   registered in PeopleSense). Pass `--coords-confirmed` only after FCUSD verifies.

## Preflight

```bash
python model_training/evo1_3/train_evo1_3.py \
  --peoplesense-dir data/incoming/evo1.3/peoplesense \
  --real-outcomes data/incoming/evo1.3/real_outcomes.json \
  --coords-confirmed \
  --preflight-only
```

Remove `--preflight-only` to run five-fold grouped MLP/LightGBM/ensemble
evaluation and ONNX/OpenVINO export. A failed run preserves the existing Evo 1.2
hybrid production policy.
