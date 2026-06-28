# Evo presentation source pack

Use this folder to build a slide deck in PowerPoint, Google Slides, Canva, Gamma, etc.

**Do not use** `docs/Evo_PhD_Deck.pptx` or `docs/Evo_Mentor_Deck.pptx` — those were auto-generated placeholders.

## Folder contents

| File | Purpose |
|------|---------|
| **`SLIDES.md`** | Slide-by-slide titles, bullets, image paths, **speaker notes in plain English** |
| **`REFERENCE.md`** | Full technical reference (pipelines, APIs, metrics, deployment) |
| **`GLOSSARY.md`** | Simple definitions of jargon (R², k-NN, OpenVINO, RAG, etc.) |
| **`images/`** | All screenshots and diagrams for the deck |

## Quick facts for any slide tool

- **Product name:** Evo
- **Live site:** https://evac-evo.vercel.app
- **Repo:** https://github.com/rvkshn-10/Evo
- **Production model:** Evo 1.2 hybrid
- **Research model:** Evo 1.3 (synthetic demo trained; not promoted)
- **Audience:** PhDs / technical mentors — go deep, but speaker notes stay simple

## Design hints

- Match website: dark navy background (`#0c1220`), cyan accents (`#38bdf8`), white text
- Use screenshots from `images/` — do not stretch low-res mobile shots; prefer `04-dashboard-desktop.png` for full UI
- Diagrams are in `images/diagram-*.png`

## Optional add-on

Drop Colab notebook export in `docs/colab/` and add one appendix slide linking to it.
