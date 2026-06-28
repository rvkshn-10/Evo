# Copy-paste prompt for another AI (Gamma, Canva, Beautiful.ai, etc.)

Paste everything below the line into your slide tool, then attach or upload the `images/` folder.

---

Build a professional PhD-level presentation for **Evo**, an evacuation intelligence platform.

**Audience:** Technical mentors and PhDs (smart audience, but speaker notes should stay in plain English).

**Design:**
- Dark navy background like the live site (#0c1220)
- Cyan accents (#38bdf8), white body text
- Use the screenshots and diagrams from the attached `images/` folder
- Do not overcrowd slides — max 5 bullets per slide
- Add speaker notes under each slide in simple, conversational language

**Source material (read these files):**
1. `SLIDES.md` — 28 slides with titles, bullets, image filenames, and speaker notes (PRIMARY)
2. `REFERENCE.md` — technical depth if you need more detail
3. `GLOSSARY.md` — define jargon simply in notes
4. `IMAGES.md` — which image goes on which slide

**Must include:**
- Live URL: https://evac-evo.vercel.app
- All five run modes explained (sync, evo 1.2, evo 1.3, external AI, broadcast)
- Backend data pipeline diagram (`diagram-data-pipeline.png`)
- Architecture diagram (`diagram-architecture.png`)
- Dashboard screenshot (`04-dashboard-desktop.png`)
- Model modal screenshot + explain train/val loss (`02-model-modal.png`)
- History modal (`05-history-modal.png`)
- Evo 1.2 hybrid diagram (`diagram-evo-hybrid.png`)
- Neural Compute Stick vs OpenVINO (`diagram-ncs-stack.png`) — clarify OpenVINO is driver software, NCS is USB hardware
- Vercel + Oracle deployment
- Google Colab training workflow
- Evo 1.3 synthetic training results (completed, not promoted — keep Evo 1.2 production)
- How the system helps people in emergencies

**Do NOT include:**
- Slides asking FCUSD for data or drill cooperation
- Claims that Evo 1.3 is production-ready (it's research only)
- Overly ugly auto-generated layouts

**Key honest message:** Metrics hit a DATA_CEILING because we lack enough real post-drill outcome labels; the pipeline and deployment work; Evo 1.2 hybrid is production policy.

Generate the deck with speaker notes on every slide.
