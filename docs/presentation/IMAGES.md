# Image index — `images/`

Use these filenames when building slides. All paths are relative to `docs/presentation/`.

| File | Use on slide | What it shows |
|------|----------------|---------------|
| `04-dashboard-desktop.png` | Dashboard tour, map, value prop | Best full-width UI shot: map, layers, stats, predictions (desktop layout) |
| `01-dashboard-main.png` | Mobile / header / controls | Top bar: History, Model, Notify, run mode, Run Agent, layer checkboxes |
| `02-model-modal.png` | Model panel, loss chart | Evo 1.2 modal: live inference area + train/val loss graph |
| `03-model-live-inference.png` | Model detail (partial) | Cropped view of model modal / header area |
| `05-history-modal.png` | History & export | Run history modal: charts, export buttons, high-risk table |
| `diagram-architecture.png` | Hosting / deployment | Browser → Vercel → Oracle → feeds + DB + local NCS |
| `diagram-data-pipeline.png` | Backend data flow | Feed sync → AlertProcessor → Predictor → Dashboard |
| `diagram-evo-hybrid.png` | Evo 1.2 architecture | Features → MLP + LightGBM → hybrid outputs |
| `diagram-broadcast.png` | 7-agent pipeline | Coordinator through Script Writer chain |
| `diagram-ncs-stack.png` | NCS vs OpenVINO | ONNX/IR → OpenVINO → CPU or USB stick |

## Regenerating diagrams

```bash
python scripts/deck_diagrams.py
# outputs to docs/deck_images/ — copy to docs/presentation/images/
```

## Regenerating screenshots

Capture from https://evac-evo.vercel.app at 1440px width:
- Dashboard (full page)
- Model modal open (◈ Model)
- History modal open (📊 History)

Save into `images/` with same naming convention.
