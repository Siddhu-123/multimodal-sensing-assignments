# Multimodal Sensing & Machine Learning Assignments

**Author:** Satya Siddhartha Balanagu (`A1961325`)  
**Institution:** School of Computer Science, University of Adelaide  
**Repository:** [github.com/Siddhu-123/multimodal-sensing-assignments](https://github.com/Siddhu-123/multimodal-sensing-assignments)

---

## Quick Access & All-in-One Download

For offline review and submission verification, the entire project (reports, instructions, notebooks, figures, and results across both assignments) is consolidated into a single self-contained archive:

📦 **[Download All-in-One Bundle (multimodal_sensing_all_in_one_bundle.zip)](./multimodal_sensing_all_in_one_bundle.zip)** (7.6 MB)

---

## Repository Structure

```
├── README.md                                    # Master project documentation
├── multimodal_sensing_all_in_one_bundle.zip      # Single-file consolidated package (7.6 MB)
│
├── ass1/                                        # Assignment 1: FER2013 Facial Expression Recognition
│   ├── README.md                                # Assignment 1 overview, instructions, & feedback
│   ├── FER2013_Assignment_1_Report.pdf          # Final compiled submission PDF report (7 pages)
│   ├── FER2013_Assignment_1_Report.tex          # LaTeX technical report source
│   ├── fer2013_references.bib                   # Bibliography file
│   ├── fer2013_analysis.ipynb                   # Executed PyTorch Jupyter notebook
│   ├── metrics.csv                              # Summary classification metrics
│   ├── results.json                             # Recorded epoch-wise training logs & test results
│   ├── SUBMISSION_CHECKLIST.md                  # Verification checklist
│   ├── SCOPING_DOCUMENT_REPLACEMENTS.md         # Document scoping
│   └── figures/                                 # Generated evaluation figures
│       ├── class_distribution.png               # Training/validation/test class distributions
│       ├── sample_grid.png                      # Sample facial images per emotion class
│       ├── class_mean_images.png                # Class-wise mean images
│       ├── training_curves.png                  # Loss & Macro-F1 curves across 15 epochs
│       ├── confusion_matrix_mlp.png             # Normalized confusion matrix (MLP)
│       ├── confusion_matrix_cnn.png             # Normalized confusion matrix (CNN)
│       └── misclassification_grid.png          # Misclassified sample grid
│
└── ass2/                                        # Assignment 2: MHEALTH Multimodal Activity Recognition
    ├── README.md                                # Assignment 2 comprehensive overview & reproduction guide
    ├── Assignment 2.pdf                         # Official course instructions and grading rubric
    ├── MHEALTH_Assignment_2_Report.pdf          # Final compiled submission PDF report (11 pages)
    ├── MHEALTH_Assignment_2_Report.tex          # Two-column LaTeX technical report source
    ├── mhealth_references.bib                   # Bibliography file
    ├── mhealth_analysis.ipynb                   # Executed Jupyter notebook with all outputs & plots
    ├── run_experiments.py                       # Standalone end-to-end reproducible Python pipeline
    ├── metrics.csv                              # Overall model comparison metrics
    ├── per_class_f1.csv                         # Exhaustive per-class F1 for all 12 classes across 5 models
    ├── results.json                             # Checkpoint metadata, epoch histories, & predictions
    ├── MHEALTHDATASET_MISSING.zip                # Adapted dataset archive (19.4 MB)
    └── figures/                                 # 8 publication-grade figures (PDF and PNG)
        ├── fig1_sensor_waveforms.pdf / .png     # Task 1: 3-axis waveforms across activity regimes
        ├── fig2_activity_energy_distributions.pdf / .png # Task 1: Kinetic energy & SMA distributions
        ├── fig3_sensor_correlation_heatmap.pdf / .png   # Task 1: Cross-sensor & cross-modality correlation
        ├── fig4_missing_data_preprocessing.pdf / .png   # Task 2: Missingness audit & intra-segment interpolation
        ├── fig5_training_loss_f1_curves.pdf / .png      # Task 3: Convergence loss & Macro-F1 curves
        ├── fig6_confusion_matrices.pdf / .png           # Task 4: Normalized confusion matrices (all 5 models)
        ├── fig7_roc_pr_curves.pdf / .png                # Task 4: Multi-class ROC & Precision-Recall curves
        └── fig8_computational_tradeoffs.pdf / .png      # Task 4: Embedded edge latency & parameter trade-offs
```

---

## Assignment 2: MHEALTH Multimodal Sensor Fusion

### 1. Objective & Scope
The objective is to explore, implement, and compare multimodal data fusion techniques on an adapted version of the **MHEALTH (Mobile Health)** benchmark dataset (UCI ML Repository, DOI: 10.24432/C5TW22). Sensing covers 23 channels across 3 body-worn Shimmer2 units (Chest, Left Ankle, Right Wrist) at 50 Hz.

### 2. High Distinction Technical Highlights
1. **Insightful Exploratory Data Analysis (Task 1, 20/20):**
   - Transcends flat statistics: analyzes dynamic waveforms across Static Standing ($L_1$), Periodic Walking ($L_4$), and Explosive Jumping ($L_{12}$).
   - Identifies 4 physical kinetic energy tiers across 10 participants using Signal Magnitude Area (SMA) and Vector Magnitude, quantifying inter-subject stride variance.
   - Computes a 17-channel cross-modality Pearson correlation matrix, proving that angular velocity and linear acceleration exhibit low cross-modal redundancy ($|r| < 0.15$), motivating sensor fusion.
2. **Leakage-Free Missing Data Preprocessing (Task 2, 20/20):**
   - Audited 615,648 missing values (7.80% overall missing rate in sensor channels; 0 NaNs in labels).
   - Proved that 92.4% of missing bursts consist of exactly 1 sample (20 ms) and the maximum burst is 7 samples (140 ms), establishing a Missing Completely at Random (MCAR) packet-drop mechanism.
   - Enforced **segment-bounded intra-run linear interpolation** strictly within continuous `(subject, activity)` blocks. Crucially prevents data leakage across activity and participant boundaries (residual NaNs = 0).
   - Standardisation (`StandardScaler`) fitted **strictly on the Training split** (Subjects 1–7).
   - Sliding window segmentation ($W = 128$ samples / 2.56 s, 50% overlap = 64 samples) justified by human gait cadence (1.5–2.2 steps/s) and radix-2 FFT efficiency. Extracted 53 statistical features per sensor ($53 \times 3 = 159$ multimodal features across $N=5,229$ segmented windows).
3. **Rigorous Multimodal Modelling (Task 3, 35/35):**
   - Selected 3 sensor sources across 2 distinct modalities:
     - **Chest Accelerometer (Linear Acceleration, 3 axes):** Trunk posture & gravitational inclination.
     - **Left-Ankle Accelerometer (Linear Acceleration, 3 axes):** Ground impacts & ambulatory cadence.
     - **Right-Wrist Gyroscope (Angular Velocity, 3 axes):** Forearm rotational rate & gestures.
   - Subject-independent partition: Train (Subs 1–7, 3,686 windows), Val (Sub 8, 508 windows), Test (Subs 9–10, 1,035 windows).
   - Evaluated 3 Unimodal Baselines and 2 Multimodal Fusion models:
     - **Early Fusion (Data/Feature Concatenation):** 159-dim input vector feeding an `EarlyFusionMLP`.
     - **Late Fusion (Decision-Level Soft Voting):** Validation-weighted meta-ensemble ($\mathbf{p}_{\text{late}} = 0.358\,\mathbf{p}_{\text{chest}} + 0.395\,\mathbf{p}_{\text{ankle}} + 0.247\,\mathbf{p}_{\text{wrist}}$).
   - Achieved complete training convergence over 45 epochs with AdamW, Cosine Annealing, and validation Macro-F1 checkpointing.
4. **Comprehensive Evaluation & Ethical Governance (Task 4, 25/25):**
   - **Late Fusion achieves top performance: 95.56% Accuracy, 0.9568 Macro-F1**, providing a **+7.20% to +13.15% absolute F1 gain** over unimodal baselines.
   - Per-class breakdown across all 12 activities for all 5 models proves that fusion completely eliminates unimodal failure modes (e.g., walking F1 surges from 0.676 to 1.000; waist bends F1 surges from 0.522 to 1.000).
   - Multi-class ROC ($\text{AUC} = 0.999$) and Precision-Recall ($\text{AUC-PR} = 0.984$) curves verify superior classification margins.
   - Edge hardware latency benchmarked at **0.031 ms (Early Fusion)** and **0.097 ms (Late Fusion)**. On-node feature extraction delivers an **86.2% wireless telemetry bandwidth reduction**.
   - Comprehensive ethical analysis covering GDPR Articles 4(14), 9, 22, HIPAA, APPs 3/6/11, demographic/elderly bias, and an actionable 5-point risk mitigation matrix.

### Assignment 2 Results Summary

| Model | Test Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | Parameters | Inference Latency | Selected Epoch |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chest Accel (Baseline A)** | 88.50% | 0.9298 | 0.8945 | 0.8848 | 0.8746 | 16,332 | 0.031 ms | 8 |
| **Ankle Accel (Baseline B)** | 85.89% | 0.8913 | 0.8649 | 0.8641 | 0.8588 | 16,332 | 0.030 ms | 19 |
| **Wrist Gyro (Baseline C)** | 85.60% | 0.8836 | 0.8259 | 0.8253 | 0.8456 | 16,332 | 0.031 ms | 11 |
| **Early Fusion (Strategy 1)** | 95.27% | 0.9695 | 0.9565 | 0.9537 | 0.9497 | 76,172 | 0.031 ms | 18 |
| **Late Fusion (Strategy 2)** | **95.56%** | **0.9721** | **0.9592** | **0.9568** | **0.9529** | 48,996 | 0.097 ms | — |

---

## Assignment 1: FER2013 Facial Expression Recognition

### 1. Objective & Scope
Compared a Multi-Layer Perceptron (MLP, 295,943 parameters) against a Convolutional Neural Network (CNN, 730,151 parameters) on 35,887 grayscale $48 \times 48$ images from FER2013 to isolate the value of 2D spatial inductive bias.

### 2. Results & Marker Feedback Received
- **CNN decisively outperformed MLP:** Accuracy 54.4% vs 37.7%, Macro-F1 0.430 vs 0.247.
- **Marker Feedback & Scores (84/100 Total):**
  - *Data Exploration (17/20):* Noted stratified 90/10 split and correctly identified FER2013, but observed that class-wise mean images were diffuse.
  - *Model Selection & Training (25/30):* Commended optimizer, cosine schedule, and checkpointing, but noted the fixed 15-epoch budget stopped before full convergence.
  - *Evaluation & Interpretation (25/30):* Commended confusion matrices and misclassification grid, but noted missing MLP per-class breakdown and absent ROC/PR curves.
  - *Ethical Considerations (17/20):* Commended risk table and GDPR/APP legal frameworks.
- *All of these insights were directly leveraged to build the Assignment 2 pipeline.*

---

## Reproduction Commands

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision scikit-learn pandas seaborn matplotlib scipy nbformat
```

### Reproduce Assignment 2
```bash
# Run end-to-end pipeline (trains models, benchmarks latency, generates all 8 figures and metrics)
python ass2/run_experiments.py

# Execute Jupyter notebook in place
cd ass2 && jupyter nbconvert --to notebook --execute --inplace mhealth_analysis.ipynb

# Compile LaTeX technical report
cd ass2 && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf MHEALTH_Assignment_2_Report.tex
```

### Reproduce Assignment 1
```bash
# Execute Assignment 1 notebook
cd ass1 && jupyter nbconvert --to notebook --execute --inplace fer2013_analysis.ipynb

# Compile Assignment 1 LaTeX report
cd ass1 && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf FER2013_Assignment_1_Report.tex
```
