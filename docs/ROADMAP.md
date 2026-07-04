# Build Roadmap

This project is built incrementally, module by module, so every piece that
lands is real, runnable, and tested — rather than a large amount of
placeholder scaffolding. This file tracks order and status.

| # | Module | Depends on | Status |
|---|--------|-----------|--------|
| 0 | Project scaffold, config, logging, DB base, FastAPI/Streamlit skeleton | – | ✅ done |
| 1 | Classification (EfficientNetB0) — training + inference + `/predict` | 0 | ⬜ next |
| 2 | Segmentation (U-Net) — training + inference + `/segment` | 0 | ⬜ |
| 3 | Explainability (Grad-CAM, Grad-CAM++, SHAP) + `/explain` | 1 | ⬜ |
| 4 | Auth (JWT, roles, password hashing) | 0 | ⬜ |
| 5 | Medical report generation (LLM + PDF export) + `/report` | 1, 3 | ⬜ |
| 6 | Medical RAG (LangChain + FAISS) + `/ask` | 0 | ⬜ |
| 7 | Multimodal fusion model (MRI + metadata) | 1 | ⬜ |
| 8 | Prognosis (XGBoost, Cox PH, Kaplan-Meier) + `/prognosis` | 6 (metadata) | ⬜ |
| 9 | Streamlit frontend pages wired to all endpoints | 1–8 | ⬜ |
| 10 | Docker, Compose, Nginx, deployment docs | all | ⬜ |
| 11 | Full test suite + CI | all | ⬜ |

## Why this order

- **Classification before segmentation**: gives the API and frontend a
  working end-to-end slice fastest.
- **Explainability right after classification**: Grad-CAM needs a trained
  classifier to hook into.
- **Auth is independent** and can be built any time after the scaffold —
  placed early so every later router can require it from day one instead of
  retrofitting auth later.
- **RAG before prognosis**: the prognosis module reuses the same
  document-ingestion utilities for guideline-based risk explanations.
- **Frontend last**: wiring real pages to real endpoints avoids rebuilding
  UI against changing API contracts.

## How we'll proceed

Each module will be delivered as: training script(s) → saved model
artifact/inference wrapper → Pydantic schemas → router → service → tests →
short written explanation of the architecture/math involved. Say "build
module 1" (or any module) when ready to continue.
