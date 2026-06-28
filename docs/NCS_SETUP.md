# Intel Neural Compute Stick — setup guide

Evo runs evacuation inference on your **CPU** by default. To use an **Intel Neural Compute Stick (NCS1 or NCS2)** plugged into USB, you need **OpenVINO** — Intel’s inference **driver/runtime**, not a separate product from the stick.

| Term | What it is |
|------|------------|
| **Neural Compute Stick (NCS)** | USB hardware (Movidius VPU) that accelerates neural net inference |
| **OpenVINO** | Intel software that loads the Evo model on **CPU** or routes it to the stick (`MYRIAD` device) |
| **ONNX** | Portable model file (`models/evo1.2/evo1.2.onnx`) — CPU fallback via ONNX Runtime |

> **You do not need OpenVINO on the website.** The Vercel dashboard talks to Oracle cloud (CPU only). NCS works only when `python3 main.py` runs on the **same Mac** as the USB stick and you open **`http://localhost:5173`**.

---

## Prerequisites

1. Evo repo cloned and backend deps installed (`pip install -r requirements.txt`)
2. Model artifacts present:
   ```text
   models/evo1.2/evo1.2.onnx
   models/evo1.2/openvino/evo1.2.xml
   models/evo1.2/openvino/evo1.2.bin
   ```
3. `.env`:
   ```env
   EVO_MODEL_VERSION=evo1.2
   EVO_PREFER_OPENVINO=true
   EVO_ACCELERATOR=auto
   ```

---

## macOS (recommended for NCS)

```bash
cd Evo
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt openvino onnxruntime
```

1. Plug **NCS2** (or NCS1) into USB — use a powered hub if it resets.
2. Start API: `python3 main.py`
3. Start UI: `cd web && npm run dev` → open **http://localhost:5173**
4. Run mode → **Evo 1.2 hybrid**
5. Inference device → **Auto**, **NCS2**, or **NCS1**
6. Green dot = stick active; amber = stick selected but not detected

Verify:
```bash
curl http://localhost:8092/api/evo/runtime | python3 -m json.tool
```
Look for `"device": "MYRIAD.0"` and `"accelerator": "ncs2"`.

---

## Windows / Linux

Same flow: install OpenVINO, plug stick, run API locally, use localhost dashboard.

- **Windows:** Install [OpenVINO](https://docs.openvino.ai/) and USB drivers if needed.
- **Linux:** Add udev rules for Movidius USB; `pip install openvino`.

---

## Inference device dropdown

| Option | Behavior |
|--------|----------|
| **Auto** | Detect NCS2 → NCS1 → CPU |
| **CPU** | ONNX Runtime or OpenVINO on CPU (no USB) |
| **NCS2** | OpenVINO `MYRIAD` on Neural Compute Stick 2 |
| **NCS1** | Older stick (Myriad 2); limited support |

Switch at runtime: `POST /api/evo/accelerator` with `{"accelerator":"ncs2"}` — no API restart.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Vercel site ignores stick | Expected — use **localhost:5173** with local API |
| `MYRIAD` not in device list | Replug stick; install OpenVINO; try powered USB hub |
| Falls back to CPU | Check `models/evo1.2/openvino/` exists; read `/api/evo/runtime` `status_message` |
| Cloud Oracle API | CPU only — no USB on VM |

---

## How it fits the stack

```
Evo ONNX / OpenVINO IR
        ↓
   OpenVINO Runtime
    ↙          ↘
 CPU          MYRIAD (NCS USB)
```

Training exports both ONNX and OpenVINO IR in Colab (`model_training/evo1_2/train_evo1_2.py`). The stick runs the **same** exported graph on edge hardware.

See also: [README.md](../README.md) · [docs/ORACLE_SETUP.md](ORACLE_SETUP.md)
