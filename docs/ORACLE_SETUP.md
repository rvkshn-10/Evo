# Oracle Cloud — Backend Setup

Deploy the **Python API** on Oracle Always Free ARM VM.  
Deploy the **dashboard** on Vercel.  
Store disaster history in **Neon** (from Vercel dashboard).

---

## Architecture

```
Vercel (web/)  ──VITE_API_BASE──►  Oracle VM :8092 (FastAPI)
                                        │
                                        ├── NOAA / USGS / PeopleSense
                                        ├── Evo 1.0 (OpenVINO)
                                        └── Neon Postgres (DATABASE_URL)
```

---

## Part 1 — Oracle account (15–30 min)

1. Go to [cloud.oracle.com](https://cloud.oracle.com) → **Start for free**
2. Verify email + add credit card (Always Free tier won't charge if you stay in limits)
3. Pick home region (e.g. `us-phoenix-1`) — **cannot change later**

---

## Part 2 — Create VM (20 min)

1. **Compute → Instances → Create instance**
2. Name: `emergency-api`
3. Image: **Ubuntu 22.04**
4. Shape: **Ampere** → `VM.Standard.A1.Flex` → 2 OCPU, 12 GB RAM (free tier)
5. Networking: assign public IPv4
6. SSH keys: download private key
7. Create

---

## Part 3 — Open firewall ports (10 min)

Oracle blocks ports by default.

1. **Networking → Virtual Cloud Networks** → your VCN
2. **Security Lists → Default Security List → Add Ingress Rules:**
   - Port `22` — SSH (your IP only)
   - Port `8092` — API (or `80`/`443` with nginx)
3. On the VM itself (after SSH):
   ```bash
   sudo iptables -I INPUT -p tcp --dport 8092 -j ACCEPT
   sudo netfilter-persistent save 2>/dev/null || true
   ```

---

## Part 4 — Install app on VM (30–45 min)

```bash
ssh -i ~/Downloads/ssh-key ubuntu@YOUR_VM_IP

sudo apt update && sudo apt install -y git python3-pip python3-venv

git clone https://github.com/rvkshn-10/Evo.git emergency-api
cd emergency-api

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env   # add API keys, DATABASE_URL, CORS_ORIGINS
```

**Required `.env` on Oracle:**
```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
DATABASE_URL=postgresql://...neon.tech/...?sslmode=require
CORS_ORIGINS=https://your-app.vercel.app
HOST=0.0.0.0
PORT=8092
```

Run schema once (Neon SQL editor or from VM):
```bash
psql "$DATABASE_URL" -f db/schema.sql
```

Test:
```bash
python3 main.py
# From laptop: curl http://YOUR_VM_IP:8092/health
```

---

## Part 5 — Keep API running (systemd)

```bash
sudo tee /etc/systemd/system/emergency-api.service << 'EOF'
[Unit]
Description=Emergency Management API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/emergency-api
EnvironmentFile=/home/ubuntu/emergency-api/.env
ExecStart=/home/ubuntu/emergency-api/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable emergency-api
sudo systemctl start emergency-api
sudo systemctl status emergency-api
```

---

## Part 6 — Vercel frontend

1. [vercel.com](https://vercel.com) → Import `rvkshn-10/Evo` (or your repo)
2. **Root directory:** `web`
3. **Environment variable:**
   ```
   VITE_API_BASE=http://YOUR_VM_IP:8092
   ```
4. Deploy

For production, put **nginx + HTTPS** on Oracle or use a domain + Let's Encrypt.

---

## Part 7 — Neon (disaster history)

1. Vercel dashboard → **Storage → Neon** → Create
2. Copy `DATABASE_URL` into Oracle `.env`
3. Run `db/schema.sql` in Neon SQL editor
4. Each **Run Agent** (sync mode) saves snapshots to `disaster_snapshots`

Check: `GET http://YOUR_VM_IP:8092/api/history`

---

## Optional — OpenVINO on Oracle ARM

```bash
pip install openvino
# Place trained files in models/evo1.0/openvino/
```

ARM supports OpenVINO CPU plugin. After Codex trains Evo, copy `evo1.0.xml` + `.bin` to the VM.

---

## Time estimate

| Step | First time | With guide |
|------|------------|------------|
| Oracle account + VM | 30–45 min | 20 min |
| Firewall + SSH | 15 min | 10 min |
| App install + systemd | 45 min | 30 min |
| Vercel + Neon | 20 min | 15 min |
| **Total** | **~2–3 hours** | **~1–1.5 hours** |

Not 2–4 hours if you've used Linux before; could be longer if Oracle networking is confusing.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Can't reach `:8092` | Check Oracle Security List + VM firewall |
| CORS error on Vercel | Set `CORS_ORIGINS` to exact Vercel URL |
| Neon connection failed | Use `?sslmode=require` in DATABASE_URL |
| Evo not loading | Train model in Colab, copy to `models/evo1.0/` |
