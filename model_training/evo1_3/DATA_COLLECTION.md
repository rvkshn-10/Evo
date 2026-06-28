# FCUSD Evo 1.3 data collection

Evo 1.3 is supervised by measured evacuation outcomes. PeopleSense occupancy and
public hazard feeds are features; neither is an evacuation label by itself.

## GGV2 provisioning and spot mapping

Provision GGV2 sensors as **Places** devices. Raspberry Pis push to PeopleSense;
Evo never reads a Pi IP address or `/health` endpoint.

| FCUSD site | Place ID | Group ID rule | Location ID example | Evo `spot_id` |
|---|---|---|---|---|
| Folsom High | `FCUSD` | contains `FHS` | `ClassRoom-01` | `folsom-high` |
| Vista del Lago | `FCUSD` | record assigned Group ID after provisioning | `MainHall-01` | `vista-del-lago` |
| Cordova Park | `FCUSD` | contains `Cordova` | `Platform-A` | `cordova-park` |

After provisioning, update the `peoplesense_match` rule in
`config/monitoring_locations.json` if FCUSD assigns a different Group ID. Verify
the GPS submitted on the GGV2 form against the installed site. Only pass
`--coords-confirmed` after FCUSD confirms all three deployments.

Current coordinates awaiting FCUSD confirmation:

- Vista del Lago: `38.6581, -121.1412`
- Folsom High: `38.6779, -121.1761`
- Cordova Park: `38.5898, -121.3026`

## PeopleSense export at drill time

At drill start, call the PeopleSense GET API through the server environment:

```bash
python scripts/export_peoplesense_occupancy.py
```

The script reads `PEOPLESENSE_API_KEY` from the environment, requests
`occupancy?filter=ALL`, applies the configured Place/Group/GPS rules, and writes
a timestamped XML file under `data/incoming/evo1.3/peoplesense/`. It never
contacts a Raspberry Pi and never writes the API key.

For every drill, retain snapshots at these points when operationally possible:

1. Immediately before evacuation begins.
2. At the recorded drill start time.
3. At one-minute intervals while evacuation is active.
4. At the measured completion time.

Every accepted `OccupancyXML` needs:

- root `generated_at` in ISO-8601 UTC;
- Zone `id` equal to `vista-del-lago`, `folsom-high`, or `cordova-park`;
- `Count`, `Density`, and `Volatility`;
- the matching measured drill timestamp retained by the safety team.

The GET snapshot is an input feature only. Do not copy its occupancy values into
success percentage or evacuation-time columns.

## Measured drill outcomes

Save measured results as `data/incoming/evo1.3/real_outcomes.json`. Start from
`data/incoming/evo1.3/real_outcomes.template.json`, but do not submit the
template itself.

Required per row:

- `spot_id`
- `outcome_timestamp` (ISO-8601, aligned within 24 hours of a PeopleSense sample)
- `scenario`
- `category`
- `evacuation_success_pct`
- `evacuation_time_min`

Recommended egress fields:

- `egress_exit_count`
- `egress_usable_width_m`
- `egress_route_length_m`
- `egress_blockage_fraction` from 0 to 1

The measured outcome file must include at least one Office Building or Stadium
row. Repeated drills across sites, occupancy levels, blocked-route conditions,
and hazard contexts are substantially more valuable than additional synthetic
variants.

## Preflight and training

```bash
python model_training/evo1_3/train_evo1_3.py \
  --peoplesense-dir data/incoming/evo1.3/peoplesense \
  --real-outcomes data/incoming/evo1.3/real_outcomes.json \
  --coords-confirmed \
  --preflight-only
```

Only after preflight returns zero:

```bash
python model_training/evo1_3/train_evo1_3.py \
  --peoplesense-dir data/incoming/evo1.3/peoplesense \
  --real-outcomes data/incoming/evo1.3/real_outcomes.json \
  --coords-confirmed \
  --output-dir artifacts/evo1.3
```

Promotion requires every gate in `validation_report.json`. A failure keeps the
Evo 1.2 hybrid policy unchanged.
