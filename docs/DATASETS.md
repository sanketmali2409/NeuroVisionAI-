# Dataset Layout

Place raw data under `datasets/raw/` exactly as below. Nothing under
`datasets/` is committed to git (see `.gitignore`) — only the structure is
tracked via `.gitkeep`.

## Brain Tumor MRI Dataset (classification)

```
datasets/raw/brain_tumor_mri/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

Each subfolder contains `.jpg`/`.png` MRI slices for that class. Source:
Kaggle "Brain Tumor MRI Dataset" (Masoud Nickparvar) or equivalent.

## LGG Segmentation Dataset (segmentation)

```
datasets/raw/lgg_segmentation/
└── <patient_id>/
    ├── <patient_id>_<slice>.tif        # MRI slice
    └── <patient_id>_<slice>_mask.tif   # binary tumor mask
```

Source: Kaggle "LGG MRI Segmentation" (Mateusz Buda et al., TCGA-LGG).

## TCGA clinical metadata (multimodal + prognosis)

```
datasets/raw/tcga_clinical/
└── clinical_data.csv
```

Expected columns (rename to match if your export differs — the loader in
`training/multimodal/dataset.py` will document the exact schema when that
module is built): `patient_id, age, sex, tumor_size_mm, symptoms,
survival_months, event_observed, ...`.

## RAG corpus (WHO / NCCN guidelines, papers)

```
rag/documents/
├── who_cns_tumor_classification.pdf
├── nccn_cns_cancers_guideline.pdf
└── papers/*.pdf
```

Any PDF dropped here is picked up by the ingestion pipeline once Module 6
(RAG) is built.

## After placing data

Run the split script (added with Module 1) to generate
`datasets/splits/{train,val,test}.csv` — training code never reads directly
from `raw/`, only from the generated split manifests, so re-splitting is
reproducible and versioned.
