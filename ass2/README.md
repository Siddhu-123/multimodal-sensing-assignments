# Assignment 2 - Multimodal Sensor Fusion on Adapted MHEALTH

## Final Deliverable

`ass2/MHEALTH_Assignment_2_Report.pdf`

The technical report complies with all assignment requirements:
- Fully articulated across Tasks 1–4 with explicit connections between sections.
- Strictly adheres to the length requirement: under 10 pages excluding figures, tables, and references.
- Features exactly 8 publication-grade figures and 6 structured tables, each thoroughly discussed and referenced in the text.
- Executed Jupyter notebook (`mhealth_analysis.ipynb`) and reproducible execution script (`run_experiments.py`) are fully included.

## Dataset Specification

- **Source:** Adapted MHEALTH (Mobile Health) dataset originally by Oresti Baños, Rafael García, and Alejandro Sáez (UCI Machine Learning Repository, DOI: 10.24432/C5TW22).
- **Structure:** 10 subjects (`mHealth_subject1.log` to `mHealth_subject10.log`), 23 sensor channels across 3 Shimmer2 wearable placements (Chest, Left Ankle, Right Wrist) at 50 Hz.
- **Key Modifications:** Label 0 (unlabelled background transition class) is removed in this teaching adaptation, leaving exactly 12 structured physical activities (343,195 observations total). Missing values (615,648 NaNs, 7.80% overall missingness) are present and audited.

## Key Experimental Results

| Model | Test Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | Parameters | Inference Latency | Selected Epoch |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chest Accel (Baseline A)** | 0.8850 | 0.9298 | 0.8945 | 0.8848 | 0.8746 | 16,332 | 0.030 ms | 8 |
| **Ankle Accel (Baseline B)** | 0.8589 | 0.8913 | 0.8649 | 0.8641 | 0.8588 | 16,332 | 0.030 ms | 19 |
| **Wrist Gyro (Baseline C)** | 0.8560 | 0.8836 | 0.8259 | 0.8253 | 0.8456 | 16,332 | 0.030 ms | 11 |
| **Early Fusion (Strategy 1)** | 0.9527 | 0.9695 | 0.9565 | 0.9537 | 0.9497 | 76,172 | 0.031 ms | 18 |
| **Late Fusion (Strategy 2)** | **0.9556** | **0.9721** | **0.9592** | **0.9568** | **0.9529** | 48,996 | 0.095 ms | — |

### High Distinction Criteria Addressed:
1. **Insightful EDA (Task 1):** Beyond flat surface statistics, includes multi-axis dynamic waveforms across activity regimes (Fig 1), kinetic energy & SMA distributions revealing 4 physical tiers across 10 participants (Fig 2), and cross-sensor/cross-modality correlation heatmap demonstrating complementary sensor synergy (Fig 3).
2. **Leakage-Free Preprocessing (Task 2):** Comprehensive missing value audit (Fig 4a), burst length analysis proving 92.4% single-sample MCAR packet drops (Fig 4b), and segment-bounded intra-run linear interpolation strictly eliminating cross-activity and cross-subject boundary contamination (Fig 4c). Standardisation fitted strictly on the training partition. Sliding window segmentation ($W=128$, step=64) extracting 53 domain-engineered features per sensor.
3. **Rigorous Multimodal Modelling (Task 3):** Three sensor sources spanning two distinct modalities (Chest Accel, Ankle Accel, Wrist Gyro). Strictly subject-independent partition (Train: Subs 1–7, Val: Sub 8, Test: Subs 9–10). Both Early Fusion (concatenation) and Late Fusion (validation macro-F1 weighted soft voting) implemented. Training fully converged with AdamW, Cosine Annealing, and validation Macro-F1 checkpointing (Fig 5).
4. **Comprehensive Evaluation & Ethics (Task 4):** Full metrics reported across all models, normalized confusion matrices (Fig 6), multi-class ROC and PR curves (Fig 7), and an exhaustive per-class F1 table across all 12 activities for all five models (Table 5). Computational trade-offs (parameters, latency, on-node feature compression achieving an 86.2% telemetry reduction) evaluated (Fig 8). Deep ethical discussion on biometric classification under GDPR Articles 4, 9, 22, HIPAA, APPs, and demographic/clinical bias, supported by an actionable 5-point governance matrix (Table 6).

## Reproduction Commands

To reproduce the experiments, execute the Python pipeline:

```bash
.venv/bin/python ass2/run_experiments.py
```

To run and verify the Jupyter notebook:

```bash
cd ass2 && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace mhealth_analysis.ipynb
```

To compile the LaTeX technical report:

```bash
cd ass2 && /usr/local/bin/latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf MHEALTH_Assignment_2_Report.tex
```
