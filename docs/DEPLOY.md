# Deploy — Vercel + Oracle + Neon

## Repos

| Repo | Purpose |
|------|---------|
| [rvkshn-10/Evo](https://github.com/rvkshn-10/Evo) | Full app + Evo model artifacts |

## Quick deploy checklist

- [ ] Push code to GitHub
- [ ] Neon: create DB, run `db/schema.sql`, copy `DATABASE_URL`
- [ ] Oracle VM: clone repo, `.env`, `systemd` service (see `docs/ORACLE_SETUP.md`)
- [ ] Vercel: import repo, root `web/`, set `VITE_API_BASE`
- [ ] Train Evo 1.0 in Colab → add `models/evo1.0/*.onnx` to repo

## Run modes (dashboard dropdown)

| Mode | LLM | Model |
|------|-----|-------|
| Sync only | None | k-NN |
| External AI | Gemini → OpenAI failover | k-NN |
| Evo 1.0 | None | OpenVINO / ONNX |
| Full broadcast | All GPT agents | k-NN |

## Local dev

```bash
pip install -r requirements.txt
python3 main.py
cd web && npm install && npm run dev
```
