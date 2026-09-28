"""
End-to-End Activity Recognition & Multimodal Fusion Pipeline on MHEALTH.
Author: Satya Siddhartha Balanagu (A1961325)
Course: Machine Learning / Multimodal Sensing - Assignment 2
"""

import os
import sys
import glob
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import rfft, rfftfreq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Set seeds for complete reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = "ass2/MHEALTHDATASET_MISSING"
FIG_DIR = "ass2/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Sensor Column Mapping based on UCI MHEALTH Specification:
# Col 0-2: Chest Accelerometer (X, Y, Z) [m/s^2]
# Col 3-4: Chest ECG (Lead 1, Lead 2) [mV]
# Col 5-7: Left-Ankle Accelerometer (X, Y, Z) [m/s^2]
# Col 8-10: Left-Ankle Gyroscope (X, Y, Z) [deg/s]
# Col 11-13: Left-Ankle Magnetometer (X, Y, Z) [local]
# Col 14-16: Right-Wrist Accelerometer (X, Y, Z) [m/s^2]
# Col 17-19: Right-Wrist Gyroscope (X, Y, Z) [deg/s]
# Col 20-22: Right-Wrist Magnetometer (X, Y, Z) [local]
# Col 23: Activity Label (1 to 12)

COLUMN_NAMES = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_ecg_1", "chest_ecg_2",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyr_x", "ankle_gyr_y", "ankle_gyr_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
    "wrist_gyr_x", "wrist_gyr_y", "wrist_gyr_z",
    "wrist_mag_x", "wrist_mag_y", "wrist_mag_z",
    "activity"
]

ACTIVITY_NAMES = {
    1: "Standing still",
    2: "Sitting & relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending (crouching)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back"
}

ACTIVITY_SHORT = {
    1: "Standing",
    2: "Sitting",
    3: "Lying",
    4: "Walking",
    5: "Stairs",
    6: "Waist bends",
    7: "Arm elevation",
    8: "Knee bends",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jumping"
}

# Selected 3 Sensor Sources across 2 Modalities:
# Source 1: Chest Accelerometer (cols 0, 1, 2) -> Modality: Acceleration
# Source 2: Left-Ankle Accelerometer (cols 5, 6, 7) -> Modality: Acceleration
# Source 3: Right-Wrist Gyroscope (cols 17, 18, 19) -> Modality: Angular Velocity (Gyroscope)
SELECTED_SENSORS = {
    "chest_acc": [0, 1, 2],
    "ankle_acc": [5, 6, 7],
    "wrist_gyr": [17, 18, 19]
}


# ==============================================================================
# 1. TASK 1: DATA LOADING & EXPLORATION
# ==============================================================================

def load_raw_dataset():
    """Loads raw data, preserves participant identifiers, and tracks runs."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "mHealth_subject*.log")))
    raw_dfs = {}
    for f in files:
        sub_id = int(os.path.basename(f).replace("mHealth_subject", "").replace(".log", ""))
        df = pd.read_csv(f, sep=r"\s+", header=None)
        df.columns = COLUMN_NAMES
        df["subject"] = sub_id
        raw_dfs[sub_id] = df
    return raw_dfs


def plot_task1_figures(raw_dfs):
    """Generates Figure 1, Figure 2, and Figure 3 for Task 1."""
    plt.rcParams.update({'font.sans-serif': 'Helvetica', 'axes.edgecolor': '#333333', 'axes.linewidth': 0.8})
    
    # -------------------------------------------------------------
    # Figure 1: Sensor Waveforms across Contrasting Activities
    # Standing Still (L1) vs Walking (L4) vs Jump Front & Back (L12)
    # -------------------------------------------------------------
    sub1 = raw_dfs[1]
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=False)
    
    activities_to_plot = [
        (1, "Standing still (Static)", 1000, 1250), # 5 seconds = 250 samples
        (4, "Walking (Periodic Locomotion)", 1000, 1250),
        (12, "Jump front & back (Dynamic Burst)", 400, 650)
    ]
    
    sensor_groups = [
        ("Chest Accel ($m/s^2$)", ["chest_acc_x", "chest_acc_y", "chest_acc_z"]),
        ("Ankle Accel ($m/s^2$)", ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"]),
        ("Wrist Gyro (sensor units)", ["wrist_gyr_x", "wrist_gyr_y", "wrist_gyr_z"])
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes_labels = ['X', 'Y', 'Z']
    
    for col_idx, (act_id, act_title, start_idx, end_idx) in enumerate(activities_to_plot):
        act_data = sub1[sub1["activity"] == act_id].reset_index(drop=True)
        chunk = act_data.iloc[start_idx:end_idx]
        t = np.arange(len(chunk)) / 50.0  # seconds
        
        for row_idx, (group_name, cols) in enumerate(sensor_groups):
            ax = axes[row_idx, col_idx]
            for c_i, col in enumerate(cols):
                # Clean interpolation just for clean visualization if NaNs exist
                sig = chunk[col].interpolate().bfill().ffill().values
                ax.plot(t, sig, label=axes_labels[c_i], color=colors[c_i], lw=1.2, alpha=0.85)
            
            if row_idx == 0:
                ax.set_title(act_title, fontsize=11, fontweight='bold', pad=10)
            if col_idx == 0:
                ax.set_ylabel(group_name, fontsize=10, fontweight='bold')
            if row_idx == 2:
                ax.set_xlabel("Time (s)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', frameon=True, ncol=3, fontsize=8)
                
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_sensor_waveforms.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig1_sensor_waveforms.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig1_sensor_waveforms")

    # -------------------------------------------------------------
    # Figure 2: Signal Magnitude Area (SMA) & Energy across Activities
    # -------------------------------------------------------------
    all_sub = pd.concat(list(raw_dfs.values()), ignore_index=True)
    # Compute instantaneous acceleration magnitude for chest and ankle
    all_sub["chest_vm"] = np.sqrt(all_sub["chest_acc_x"]**2 + all_sub["chest_acc_y"]**2 + all_sub["chest_acc_z"]**2)
    all_sub["ankle_vm"] = np.sqrt(all_sub["ankle_acc_x"]**2 + all_sub["ankle_acc_y"]**2 + all_sub["ankle_acc_z"]**2)
    all_sub["wrist_gyro_vm"] = np.sqrt(all_sub["wrist_gyr_x"]**2 + all_sub["wrist_gyr_y"]**2 + all_sub["wrist_gyr_z"]**2)
    
    # Aggregate per subject and activity to examine inter-subject variance
    summary = all_sub.groupby(["activity", "subject"])[["chest_vm", "ankle_vm", "wrist_gyro_vm"]].mean().reset_index()
    summary["activity_name"] = summary["activity"].map(ACTIVITY_SHORT)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 4.4), sharex=True)
    
    order = [ACTIVITY_SHORT[i] for i in range(1, 13)]
    
    # Ankle Acceleration Vector Magnitude (m/s^2)
    sns.boxplot(x="activity_name", y="ankle_vm", data=summary, order=order, ax=ax1, palette="Blues_d", width=0.6, hue="activity_name", legend=False)
    sns.stripplot(x="activity_name", y="ankle_vm", data=summary, order=order, ax=ax1, color="crimson", size=4, jitter=0.2)
    ax1.set_title("(a) Ankle Acceleration Magnitude ($m/s^2$)", fontsize=11, fontweight='bold', pad=3)
    ax1.set_ylabel("Mean Accel ($m/s^2$)", fontsize=9.5, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=8.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Wrist Gyroscope Angular Rate (sensor units)
    sns.boxplot(x="activity_name", y="wrist_gyro_vm", data=summary, order=order, ax=ax2, palette="Oranges_d", width=0.6, hue="activity_name", legend=False)
    sns.stripplot(x="activity_name", y="wrist_gyro_vm", data=summary, order=order, ax=ax2, color="darkblue", size=4, jitter=0.2)
    ax2.set_title("(b) Wrist Angular Velocity (sensor units)", fontsize=11, fontweight='bold', pad=3)
    ax2.set_ylabel("Mean Rate (sensor units)", fontsize=9.5, fontweight='bold')
    ax2.set_xlabel("Activity", fontsize=9.5, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=8.5)
    ax2.tick_params(axis='y', labelsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_activity_energy_distributions.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig2_activity_energy_distributions.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig2_activity_energy_distributions")

    # -------------------------------------------------------------
    # Figure 3: Cross-Sensor & Cross-Modality Correlation Heatmap
    # -------------------------------------------------------------
    corr_cols = [
        "chest_acc_x", "chest_acc_y", "chest_acc_z",
        "chest_ecg_1", "chest_ecg_2",
        "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
        "ankle_gyr_x", "ankle_gyr_y", "ankle_gyr_z",
        "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
        "wrist_gyr_x", "wrist_gyr_y", "wrist_gyr_z"
    ]
    corr_labels = [
        "Ch-AccX", "Ch-AccY", "Ch-AccZ", "ECG-1", "ECG-2",
        "Ank-AccX", "Ank-AccY", "Ank-AccZ", "Ank-GyrX", "Ank-GyrY", "Ank-GyrZ",
        "Wri-AccX", "Wri-AccY", "Wri-AccZ", "Wri-GyrX", "Wri-GyrY", "Wri-GyrZ"
    ]
    # Sample 50k points to calculate correlation robustly
    sample_df = all_sub[corr_cols].dropna().sample(n=50000, random_state=SEED)
    corr_mat = sample_df.corr()
    
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(
        corr_mat, mask=mask, cmap="vlag", vmin=-1.0, vmax=1.0, center=0,
        xticklabels=corr_labels, yticklabels=corr_labels,
        annot=True, fmt=".2f", annot_kws={"size": 7.5, "weight": "bold"},
        cbar_kws={"shrink": 0.82},
        ax=ax
    )
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10.5)
    cbar.set_label("Pearson Correlation Coefficient", fontsize=11, fontweight='bold')
    ax.set_title("Cross-Sensor & Cross-Modality Correlation Matrix (MHEALTH)", fontsize=12.5, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_sensor_correlation_heatmap.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig3_sensor_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig3_sensor_correlation_heatmap")


# ==============================================================================
# 2. TASK 2: MISSING DATA AUDIT, INTERPOLATION & SEGMENTATION
# ==============================================================================

def analyze_and_clean_data(raw_dfs):
    """
    Quantifies missing data patterns and applies segment-bounded linear interpolation.
    Strictly avoids mixing across participant or activity boundaries.
    """
    total_samples = 0
    total_nans = 0
    col_nans = np.zeros(23)
    burst_lengths_per_col = {c: [] for c in range(23)}
    
    cleaned_dfs = {}
    
    for sub_id, df in raw_dfs.items():
        total_samples += len(df)
        total_nans += df.iloc[:, :23].isna().sum().sum()
        col_nans += df.iloc[:, :23].isna().sum().values
        
        # Analyze burst lengths
        for c in range(23):
            is_na = df.iloc[:, c].isna().values
            diffs = np.diff(np.concatenate(([0], is_na.view(np.int8), [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            lengths = ends - starts
            burst_lengths_per_col[c].extend(lengths)
            
        # Segment-bounded intra-run interpolation
        changes = (df["activity"] != df["activity"].shift()).cumsum()
        cleaned_runs = []
        for run_id, grp in df.groupby(changes):
            # Interpolate strictly within this continuous activity segment
            interp_signals = grp.iloc[:, :23].interpolate(method="linear", limit_direction="both")
            grp_clean = pd.concat([interp_signals, grp[["activity", "subject"]]], axis=1)
            cleaned_runs.append(grp_clean)
            
        cleaned_df = pd.concat(cleaned_runs, ignore_index=True)
        assert cleaned_df.iloc[:, :23].isna().sum().sum() == 0, f"NaNs remain in subject {sub_id}!"
        cleaned_dfs[sub_id] = cleaned_df
        
    print(f"Missing Data Audit: {total_nans} NaNs across {total_samples} samples ({total_nans / (total_samples * 23) * 100:.2f}%).")
    
    return cleaned_dfs, col_nans, total_samples, burst_lengths_per_col


def plot_task2_figures(col_nans, total_samples, burst_lengths_per_col, raw_dfs, cleaned_dfs):
    """Generates Figure 4: Missing data analysis & interpolation demonstration."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    # 4a: Missingness Rate per Sensor Column
    ax1 = fig.add_subplot(gs[0, 0])
    cols_pct = (col_nans / total_samples) * 100
    bars = ax1.bar(range(23), cols_pct, color='#3470a3', edgecolor='black', lw=0.6, width=0.65)
    ax1.set_title("(a) Missing Value Rate by Sensor Channel (%)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Sensor Channel Index (0 to 22)", fontsize=10)
    ax1.set_ylabel("Missingness (%)", fontsize=10)
    ax1.set_ylim(0, 18)
    ax1.set_xticks(range(23))
    ax1.grid(True, linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        if h > 8:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.1f}%", ha='center', va='bottom', fontsize=6.5)
            
    # 4b: Burst Length Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    all_bursts = []
    for lengths in burst_lengths_per_col.values():
        all_bursts.extend(lengths)
    burst_series = pd.Series(all_bursts).value_counts().sort_index()
    burst_pct = (burst_series / burst_series.sum()) * 100
    ax2.bar(burst_series.index, burst_pct.values, color='#e26d5c', edgecolor='black', lw=0.6, width=0.55)
    ax2.set_title("(b) Missing Data Burst Length Distribution", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Consecutive Missing Samples (at 50 Hz)", fontsize=10)
    ax2.set_ylabel("Percentage of Bursts (%)", fontsize=10)
    ax2.set_xticks(range(1, 8))
    ax2.set_xticklabels([f"{i} ({i*20}ms)" for i in range(1, 8)])
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle='--', alpha=0.5)
    for i, (idx, val) in enumerate(burst_pct.items()):
        ax2.text(idx, val + 1.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    # 4c & 4d: Visual demonstration of Intra-Segment Linear Interpolation
    # Show raw with missing values vs interpolated signal
    ax3 = fig.add_subplot(gs[1, :])
    raw_sub1 = raw_dfs[1]
    clean_sub1 = cleaned_dfs[1]
    
    # Pick a 200-sample window with missing values in Chest Accel Y during Activity 4 (Walking)
    walk_mask = (raw_sub1["activity"] == 4)
    walk_raw = raw_sub1[walk_mask].iloc[100:250]["chest_acc_y"].values
    walk_clean = clean_sub1[walk_mask].iloc[100:250]["chest_acc_y"].values
    t = np.arange(len(walk_raw)) / 50.0
    
    ax3.plot(t, walk_clean, color='#2ca02c', lw=1.8, label="Preprocessed (Intra-Segment Linear Interpolation)", zorder=2)
    # Plot raw valid points
    ax3.scatter(t[~np.isnan(walk_raw)], walk_raw[~np.isnan(walk_raw)], color='#1f77b4', s=20, label="Observed Sensor Packets", zorder=3)
    # Highlight imputed missing points
    missing_idx = np.where(np.isnan(walk_raw))[0]
    ax3.scatter(t[missing_idx], walk_clean[missing_idx], color='red', marker='x', s=60, lw=2.0, label="Imputed Missing Samples (No Boundary Leakage)", zorder=4)
    
    ax3.set_title("(c) Segment-Bounded Time-Series Interpolation on Chest Accel-Y (Activity 4: Walking)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylabel("Acceleration ($m/s^2$)", fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='lower right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_missing_data_preprocessing.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig4_missing_data_preprocessing.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig4_missing_data_preprocessing")


# ==============================================================================
# 3. FEATURE EXTRACTION & SEGMENTATION
# ==============================================================================

def extract_sensor_features(w, fs=50):
    """
    Extracts 53 comprehensive domain features from a 3-axis window (W, 3).
    - Time-domain per axis: mean, std, var, min, max, ptp, median, iqr, skew, kurt, rms (11 x 3 = 33)
    - Vector magnitude SMV: mean, std, min, max (4)
    - Signal Magnitude Area (SMA): (1)
    - Pairwise correlations: rho(xy), rho(xz), rho(yz) (3)
    - Frequency-domain FFT per axis: spectral energy, entropy, dominant freq, dom magnitude (4 x 3 = 12)
    Total = 53 features.
    """
    feats = []
    
    # 1. Per-axis time-domain features
    for ax in range(3):
        sig = w[:, ax]
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        var_val = np.var(sig)
        min_val = np.min(sig)
        max_val = np.max(sig)
        ptp_val = max_val - min_val
        median_val = np.median(sig)
        iqr_val = stats.iqr(sig)
        skew_val = stats.skew(sig)
        kurt_val = stats.kurtosis(sig)
        rms_val = np.sqrt(np.mean(sig**2))
        feats.extend([mean_val, std_val, var_val, min_val, max_val, ptp_val, median_val, iqr_val, skew_val, kurt_val, rms_val])
        
    # 2. Vector magnitude & cross-axis features
    smv = np.sqrt(np.sum(w**2, axis=1))
    feats.extend([np.mean(smv), np.std(smv), np.min(smv), np.max(smv)])
    
    sma = np.mean(np.sum(np.abs(w), axis=1))
    feats.append(sma)
    
    # Pairwise correlations
    corr_xy = np.corrcoef(w[:, 0], w[:, 1])[0, 1] if np.std(w[:, 0]) > 1e-6 and np.std(w[:, 1]) > 1e-6 else 0.0
    corr_xz = np.corrcoef(w[:, 0], w[:, 2])[0, 1] if np.std(w[:, 0]) > 1e-6 and np.std(w[:, 2]) > 1e-6 else 0.0
    corr_yz = np.corrcoef(w[:, 1], w[:, 2])[0, 1] if np.std(w[:, 1]) > 1e-6 and np.std(w[:, 2]) > 1e-6 else 0.0
    feats.extend([
        0.0 if np.isnan(corr_xy) else corr_xy,
        0.0 if np.isnan(corr_xz) else corr_xz,
        0.0 if np.isnan(corr_yz) else corr_yz
    ])
    
    # 3. Frequency domain features via FFT
    freqs = rfftfreq(len(w), 1.0 / fs)
    for ax in range(3):
        sig = w[:, ax]
        fft_vals = np.abs(rfft(sig))
        psd = fft_vals**2
        psd_norm = psd / (np.sum(psd) + 1e-9)
        spectral_energy = np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-9))
        dom_idx = np.argmax(fft_vals[1:]) + 1 if len(fft_vals) > 1 else 0
        dom_freq = freqs[dom_idx]
        dom_mag = fft_vals[dom_idx]
        feats.extend([spectral_energy, spectral_entropy, dom_freq, dom_mag])
        
    return np.array(feats, dtype=np.float32)


def build_windowed_dataset(cleaned_dfs, window_size=128, step_size=64):
    """
    Segments sensor data into sliding windows strictly within each (subject, activity) run.
    Extracts features for:
    - Chest Accel (53 feats)
    - Ankle Accel (53 feats)
    - Wrist Gyro (53 feats)
    - Early Fusion (159 feats)
    """
    records = []
    
    print("Segmenting time windows and extracting features...")
    start_t = time.time()
    
    for sub_id, df in cleaned_dfs.items():
        changes = (df["activity"] != df["activity"].shift()).cumsum()
        for run_id, grp in df.groupby(changes):
            act = int(grp.iloc[0]["activity"])
            n_samples = len(grp)
            n_wins = (n_samples - window_size) // step_size + 1
            if n_wins <= 0:
                continue
                
            chest_data = grp.iloc[:, SELECTED_SENSORS["chest_acc"]].values
            ankle_data = grp.iloc[:, SELECTED_SENSORS["ankle_acc"]].values
            wrist_data = grp.iloc[:, SELECTED_SENSORS["wrist_gyr"]].values
            
            for w_i in range(n_wins):
                s_idx = w_i * step_size
                e_idx = s_idx + window_size
                
                f_chest = extract_sensor_features(chest_data[s_idx:e_idx])
                f_ankle = extract_sensor_features(ankle_data[s_idx:e_idx])
                f_wrist = extract_sensor_features(wrist_data[s_idx:e_idx])
                
                f_early = np.concatenate([f_chest, f_ankle, f_wrist])
                
                records.append({
                    "subject": sub_id,
                    "activity": act,
                    "f_chest": f_chest,
                    "f_ankle": f_ankle,
                    "f_wrist": f_wrist,
                    "f_early": f_early
                })
                
    elapsed = time.time() - start_t
    print(f"Extracted {len(records)} windows in {elapsed:.2f} seconds.")
    return records


# ==============================================================================
# 4. TASK 3: MULTIMODAL MODELLING & TRAINING
# ==============================================================================

# Subject-independent Partition:
# Train: Subjects 1 to 7 (~70%)
# Validation: Subject 8 (~10%)
# Test: Subjects 9 and 10 (~20%)
TRAIN_SUBS = [1, 2, 3, 4, 5, 6, 7]
VAL_SUBS = [8]
TEST_SUBS = [9, 10]


class UnimodalMLP(nn.Module):
    """Deep Multilayer Perceptron for unimodal sensor feature stream."""
    def __init__(self, in_features=53, num_classes=12, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


class EarlyFusionMLP(nn.Module):
    """Deep Multilayer Perceptron for concatenated multimodal feature vector."""
    def __init__(self, in_features=159, num_classes=12, hidden_dim1=256, hidden_dim2=128, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, epochs=45, lr=1e-3, weight_decay=1e-4):
    """
    Trains PyTorch model with AdamW, Cosine Annealing, and validation Macro-F1 checkpointing.
    Tracks all training and validation metrics per epoch for convergence diagnostics.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "train_acc": [], "val_acc": []
    }
    
    best_val_f1 = -1.0
    best_weights = None
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * len(y_b)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(y_b.cpu().numpy())
            
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                out = model(X_b)
                loss = criterion(out, y_b)
                val_loss += loss.item() * len(y_b)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_b.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_macro_f1"].append(train_f1)
        history["val_macro_f1"].append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best checkpoint
    model.load_state_dict(best_weights)
    print(f"Loaded best checkpoint from Epoch {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")
    return model, history, best_epoch


# ==============================================================================
# 5. EXPERIMENT EXECUTION & BENCHMARKING
# ==============================================================================

def run_all_experiments(records):
    """Prepares datasets, trains 5 models, and records detailed metrics."""
    # Partition records
    train_records = [r for r in records if r["subject"] in TRAIN_SUBS]
    val_records = [r for r in records if r["subject"] in VAL_SUBS]
    test_records = [r for r in records if r["subject"] in TEST_SUBS]
    
    print(f"Dataset Partitions (Subject-Independent):")
    print(f"  Train Set (Subs {TRAIN_SUBS}): {len(train_records)} windows")
    print(f"  Val Set   (Subs {VAL_SUBS}):   {len(val_records)} windows")
    print(f"  Test Set  (Subs {TEST_SUBS}):  {len(test_records)} windows")
    
    # Target vectors (0-indexed: 0 to 11)
    y_train = np.array([r["activity"] - 1 for r in train_records], dtype=np.int64)
    y_val = np.array([r["activity"] - 1 for r in val_records], dtype=np.int64)
    y_test = np.array([r["activity"] - 1 for r in test_records], dtype=np.int64)
    
    # Feature matrices
    def get_features(split_records, feat_key):
        return np.array([r[feat_key] for r in split_records], dtype=np.float32)
        
    X_chest_tr = get_features(train_records, "f_chest")
    X_chest_va = get_features(val_records, "f_chest")
    X_chest_te = get_features(test_records, "f_chest")
    
    X_ankle_tr = get_features(train_records, "f_ankle")
    X_ankle_va = get_features(val_records, "f_ankle")
    X_ankle_te = get_features(test_records, "f_ankle")
    
    X_wrist_tr = get_features(train_records, "f_wrist")
    X_wrist_va = get_features(val_records, "f_wrist")
    X_wrist_te = get_features(test_records, "f_wrist")
    
    X_early_tr = get_features(train_records, "f_early")
    X_early_va = get_features(val_records, "f_early")
    X_early_te = get_features(test_records, "f_early")
    
    # Fit StandardScaler STRICTLY on Train split to prevent any data leakage
    sc_chest = StandardScaler().fit(X_chest_tr)
    X_chest_tr = sc_chest.transform(X_chest_tr)
    X_chest_va = sc_chest.transform(X_chest_va)
    X_chest_te = sc_chest.transform(X_chest_te)
    
    sc_ankle = StandardScaler().fit(X_ankle_tr)
    X_ankle_tr = sc_ankle.transform(X_ankle_tr)
    X_ankle_va = sc_ankle.transform(X_ankle_va)
    X_ankle_te = sc_ankle.transform(X_ankle_te)
    
    sc_wrist = StandardScaler().fit(X_wrist_tr)
    X_wrist_tr = sc_wrist.transform(X_wrist_tr)
    X_wrist_va = sc_wrist.transform(X_wrist_va)
    X_wrist_te = sc_wrist.transform(X_wrist_te)
    
    sc_early = StandardScaler().fit(X_early_tr)
    X_early_tr = sc_early.transform(X_early_tr)
    X_early_va = sc_early.transform(X_early_va)
    X_early_te = sc_early.transform(X_early_te)
    
    # Create DataLoaders
    batch_size = 64
    def make_loader(X, y, shuffle=True):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        
    loader_chest_tr = make_loader(X_chest_tr, y_train, shuffle=True)
    loader_chest_va = make_loader(X_chest_va, y_val, shuffle=False)
    loader_chest_te = make_loader(X_chest_te, y_test, shuffle=False)
    
    loader_ankle_tr = make_loader(X_ankle_tr, y_train, shuffle=True)
    loader_ankle_va = make_loader(X_ankle_va, y_val, shuffle=False)
    loader_ankle_te = make_loader(X_ankle_te, y_test, shuffle=False)
    
    loader_wrist_tr = make_loader(X_wrist_tr, y_train, shuffle=True)
    loader_wrist_va = make_loader(X_wrist_va, y_val, shuffle=False)
    loader_wrist_te = make_loader(X_wrist_te, y_test, shuffle=False)
    
    loader_early_tr = make_loader(X_early_tr, y_train, shuffle=True)
    loader_early_va = make_loader(X_early_va, y_val, shuffle=False)
    loader_early_te = make_loader(X_early_te, y_test, shuffle=False)
    
    # -------------------------------------------------------------
    # Train Models
    # -------------------------------------------------------------
    epochs = 45
    histories = {}
    best_epochs = {}
    
    print("\n--- Training Model 1: Unimodal Chest Accelerometer ---")
    model_chest = UnimodalMLP(in_features=53).to(DEVICE)
    model_chest, histories["Chest Accel"], best_epochs["Chest Accel"] = train_model(
        model_chest, loader_chest_tr, loader_chest_va, epochs=epochs, lr=1e-3
    )
    
    print("\n--- Training Model 2: Unimodal Ankle Accelerometer ---")
    model_ankle = UnimodalMLP(in_features=53).to(DEVICE)
    model_ankle, histories["Ankle Accel"], best_epochs["Ankle Accel"] = train_model(
        model_ankle, loader_ankle_tr, loader_ankle_va, epochs=epochs, lr=1e-3
    )
    
    print("\n--- Training Model 3: Unimodal Wrist Gyroscope ---")
    model_wrist = UnimodalMLP(in_features=53).to(DEVICE)
    model_wrist, histories["Wrist Gyro"], best_epochs["Wrist Gyro"] = train_model(
        model_wrist, loader_wrist_tr, loader_wrist_va, epochs=epochs, lr=1e-3
    )
    
    print("\n--- Training Model 4: Multimodal Early Fusion (Concatenation) ---")
    model_early = EarlyFusionMLP(in_features=159).to(DEVICE)
    model_early, histories["Early Fusion"], best_epochs["Early Fusion"] = train_model(
        model_early, loader_early_tr, loader_early_va, epochs=epochs, lr=1e-3
    )
    
    # -------------------------------------------------------------
    # Inference on Test Set
    # -------------------------------------------------------------
    def get_probabilities(model, loader):
        model.eval()
        probs_list = []
        with torch.no_grad():
            for X_b, _ in loader:
                X_b = X_b.to(DEVICE)
                logits = model(X_b)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        return np.vstack(probs_list)
        
    probs_chest = get_probabilities(model_chest, loader_chest_te)
    probs_ankle = get_probabilities(model_ankle, loader_ankle_te)
    probs_wrist = get_probabilities(model_wrist, loader_wrist_te)
    probs_early = get_probabilities(model_early, loader_early_te)
    
    # Model 5: Multimodal Late Fusion (Decision-level Soft Voting)
    # Weights determined by validation performance
    w_chest = histories["Chest Accel"]["val_macro_f1"][best_epochs["Chest Accel"] - 1]
    w_ankle = histories["Ankle Accel"]["val_macro_f1"][best_epochs["Ankle Accel"] - 1]
    w_wrist = histories["Wrist Gyro"]["val_macro_f1"][best_epochs["Wrist Gyro"] - 1]
    w_sum = w_chest + w_ankle + w_wrist
    w_chest /= w_sum; w_ankle /= w_sum; w_wrist /= w_sum
    print(f"\nLate Fusion Weights: Chest={w_chest:.3f}, Ankle={w_ankle:.3f}, Wrist={w_wrist:.3f}")
    
    probs_late = (w_chest * probs_chest) + (w_ankle * probs_ankle) + (w_wrist * probs_wrist)
    
    # Collect predictions for all 5 models
    all_models = {
        "Chest Accel": (model_chest, probs_chest, X_chest_te),
        "Ankle Accel": (model_ankle, probs_ankle, X_ankle_te),
        "Wrist Gyro": (model_wrist, probs_wrist, X_wrist_te),
        "Early Fusion": (model_early, probs_early, X_early_te),
        "Late Fusion": (None, probs_late, None)
    }
    
    # Latency benchmarking & parameter counts
    metrics_summary = []
    per_class_summary = {}
    
    for name, (mod, probs, sample_X) in all_models.items():
        preds = np.argmax(probs, axis=1)
        
        acc = accuracy_score(y_test, preds)
        macro_p = precision_score(y_test, preds, average="macro", zero_division=0)
        macro_r = recall_score(y_test, preds, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        
        # Per class F1
        c_report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        per_class_summary[name] = {
            f"Class_{c+1}": c_report[str(c)]["f1-score"] for c in range(12)
        }
        
        # Parameter count
        if mod is not None:
            n_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        else:
            n_params = sum(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in [model_chest, model_ankle, model_wrist])
            
        # Benchmark Latency (1000 single-window inferences on CPU)
        if mod is not None:
            mod_cpu = mod.to("cpu")
            dummy = torch.randn(1, mod_cpu.net[0].in_features, dtype=torch.float32)
            # warmup
            for _ in range(50): _ = mod_cpu(dummy)
            t0 = time.perf_counter()
            for _ in range(1000):
                _ = mod_cpu(dummy)
            latency_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0
            mod.to(DEVICE)
        else:
            # Late fusion latency = sum of 3 unimodal + weighted sum
            latency_ms = metrics_summary[0]["latency_ms"] + metrics_summary[1]["latency_ms"] + metrics_summary[2]["latency_ms"] + 0.005
            
        metrics_summary.append({
            "model": name,
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "parameters": n_params,
            "latency_ms": round(latency_ms, 3),
            "selected_epoch": best_epochs.get(name, "-")
        })
        
    return (
        all_models, histories, best_epochs, metrics_summary,
        per_class_summary, y_test, probs_chest, probs_ankle, probs_wrist, probs_early, probs_late
    )


# ==============================================================================
# 6. EVALUATION VISUALISATIONS (FIG 5, 6, 7, 8)
# ==============================================================================

def plot_evaluation_figures(histories, all_models, y_test, metrics_summary):
    """Plots Figures 5, 6, 7, 8."""
    
    # -------------------------------------------------------------
    # Figure 5: Training Diagnostics & Loss / Macro-F1 Curves
    # -------------------------------------------------------------
    # Figure 5: Training Diagnostics (Loss and F1 Evolution)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 5.6))
    palette = {
        "Chest Accel": "#1f77b4",
        "Ankle Accel": "#ff7f0e",
        "Wrist Gyro": "#2ca02c",
        "Early Fusion": "#d62728"
    }
    
    epochs_range = range(1, len(histories["Chest Accel"]["train_loss"]) + 1)
    
    for name, hist in histories.items():
        color = palette[name]
        # Loss curves: dashed for train, solid for val
        ax1.plot(epochs_range, hist["val_loss"], color=color, lw=2.2, label=f"{name} (Val)")
        ax1.plot(epochs_range, hist["train_loss"], color=color, lw=1.3, linestyle="--", alpha=0.55)
        
        # F1 curves
        ax2.plot(epochs_range, hist["val_macro_f1"], color=color, lw=2.2, label=f"{name} (Val)")
        ax2.plot(epochs_range, hist["train_macro_f1"], color=color, lw=1.3, linestyle="--", alpha=0.55)
        
    ax1.set_title("(a) Training (dashed) & Validation (solid) Loss", fontsize=11.5, fontweight='bold', pad=6)
    ax1.set_xlabel("Epoch", fontsize=10.5, fontweight='bold')
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10.5, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.9, ncol=2)
    
    ax2.set_title("(b) Training (dashed) & Validation (solid) Macro-F1", fontsize=11.5, fontweight='bold', pad=6)
    ax2.set_xlabel("Epoch", fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("Macro-F1 Score", fontsize=10.5, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_training_loss_f1_curves.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig5_training_loss_f1_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig5_training_loss_f1_curves")

    # -------------------------------------------------------------
    # Figure 6: Normalized Confusion Matrices (Baselines vs Multimodal)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    class_short_labels = [ACTIVITY_SHORT[i] for i in range(1, 13)]
    
    model_order = ["Chest Accel", "Ankle Accel", "Wrist Gyro", "Early Fusion", "Late Fusion"]
    
    for idx, name in enumerate(model_order):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        probs = all_models[name][1]
        preds = np.argmax(probs, axis=1)
        cm = confusion_matrix(y_test, preds, normalize="true")
        
        annot_matrix = np.empty_like(cm, dtype=object)
        for r_i in range(12):
            for c_i in range(12):
                v = cm[r_i, c_i]
                pct = int(round(v * 100))
                annot_matrix[r_i, c_i] = f"{pct}" if pct >= 1 else ""

        sns.heatmap(
            cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=False,
            xticklabels=class_short_labels, yticklabels=class_short_labels,
            annot_kws={"size": 7.5, "weight": "bold"}, ax=ax, square=True
        )
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        ax.set_title(f"{name} (Acc: {acc*100:.1f}%, F1: {f1:.3f})", fontsize=11.5, fontweight='bold', pad=5)
        ax.set_ylabel("True Activity", fontsize=10, fontweight='bold')
        ax.set_xlabel("Predicted Activity", fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=8.5)
        ax.tick_params(axis='y', rotation=0, labelsize=8.5)
        
    # Hide the 6th subplot
    axes[1, 2].axis("off")
    # Add a summary legend / annotation in the empty space
    summary_text = (
        "Evaluation Insights:\n"
        "• Values are row-normalised percentages (%)\n"
        "• Unimodal Chest confuses Sitting vs Standing\n"
        "  and Walking vs Climbing Stairs.\n"
        "• Unimodal Ankle confuses static trunk postures\n"
        "  (Standing vs Sitting) and upper arm gestures.\n"
        "• Unimodal Wrist fails on leg-dominant tasks\n"
        "  (Walking vs Cycling vs Jumping).\n"
        "• Multimodal Early & Late Fusion resolve\n"
        "  modality blind spots, achieving near-perfect\n"
        "  discrimination across all 12 activities."
    )
    axes[1, 2].text(0.05, 0.45, summary_text, fontsize=10.5, va='center', ha='left',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4f8", edgecolor="#3470a3", lw=1.2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_confusion_matrices.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig6_confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig6_confusion_matrices")

    # -------------------------------------------------------------
    # Figure 7: Multi-Class ROC & Precision-Recall Curves
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    
    # Binarize targets for multi-class ROC and PR
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_test, classes=list(range(12)))
    
    colors_dict = {
        "Chest Accel": "#1f77b4",
        "Ankle Accel": "#ff7f0e",
        "Wrist Gyro": "#2ca02c",
        "Early Fusion": "#d62728",
        "Late Fusion": "#9467bd"
    }
    
    for name in model_order:
        probs = all_models[name][1]
        
        # Macro-average ROC
        fpr_dict, tpr_dict = {}, {}
        for c in range(12):
            fpr_dict[c], tpr_dict[c], _ = roc_curve(y_bin[:, c], probs[:, c])
        # Aggregate all FPR
        all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in range(12)]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(12):
            mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
        mean_tpr /= 12.0
        roc_auc = auc(all_fpr, mean_tpr)
        
        ax1.plot(all_fpr, mean_tpr, color=colors_dict[name], lw=2.2,
                 label=f"{name} (AUC = {roc_auc:.3f})")
                 
        # Macro-average Precision-Recall
        recall_grid = np.linspace(0, 1, 100)
        mean_prec = np.zeros_like(recall_grid)
        for c in range(12):
            p, r, _ = precision_recall_curve(y_bin[:, c], probs[:, c])
            mean_prec += np.interp(recall_grid, r[::-1], p[::-1])
        mean_prec /= 12.0
        pr_auc = auc(recall_grid, mean_prec)
        
        ax2.plot(recall_grid, mean_prec, color=colors_dict[name], lw=2.2,
                 label=f"{name} (AUC-PR = {pr_auc:.3f})")
                 
    ax1.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax1.set_title("(a) Macro-Average ROC Curves", fontsize=13.5, fontweight='bold', pad=8)
    ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
    
    ax2.set_title("(b) Macro-Average Precision-Recall Curves", fontsize=13.5, fontweight='bold', pad=8)
    ax2.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_roc_pr_curves.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig7_roc_pr_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig7_roc_pr_curves")

    # -------------------------------------------------------------
    # Figure 8: Computational Trade-off Analysis (Side-by-Side 1x2)
    # -------------------------------------------------------------
    df_metrics = pd.DataFrame(metrics_summary)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))

    short_names = {
        "Chest Accel": "Chest",
        "Ankle Accel": "Ankle",
        "Wrist Gyro": "Wrist",
        "Early Fusion": "Early Fus.",
        "Late Fusion": "Late Fus."
    }
    scatter_colors = [colors_dict[m] for m in df_metrics["model"]]

    # (a) Parameters vs F1
    ax1.scatter(df_metrics["parameters"], df_metrics["macro_f1"], s=90, c=scatter_colors, edgecolors="black", linewidths=0.9, zorder=3)
    for _, row in df_metrics.iterrows():
        m = row["model"]
        s = short_names[m]
        p = int(row["parameters"])
        f = float(row["macro_f1"])
        if m == "Late Fusion":
            ax1.annotate(f"{s} ({p//1000}k)", (p, f), xytext=(-8, 7), textcoords="offset points", ha="right", fontsize=8.0, fontweight="bold")
        elif m == "Early Fusion":
            ax1.annotate(f"{s} ({p//1000}k)", (p, f), xytext=(-8, -13), textcoords="offset points", ha="right", fontsize=8.0, fontweight="bold")
        elif m == "Chest Accel":
            ax1.annotate(f"{s} ({p//1000}k)", (p, f), xytext=(8, 2), textcoords="offset points", ha="left", fontsize=8.0, fontweight="bold")
        elif m == "Ankle Accel":
            ax1.annotate(f"{s} ({p//1000}k)", (p, f), xytext=(8, -10), textcoords="offset points", ha="left", fontsize=8.0, fontweight="bold")
        else:
            ax1.annotate(f"{s} ({p//1000}k)", (p, f), xytext=(8, -3), textcoords="offset points", ha="left", fontsize=8.0, fontweight="bold")

    ax1.set_title("(a) Macro-F1 vs Parameters", fontsize=9.5, fontweight="bold", pad=4)
    ax1.set_xlabel("Trainable Parameters", fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Test Macro-F1", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0.81, 0.985)
    ax1.set_xlim(8000, 88000)
    ax1.tick_params(axis="both", labelsize=8.0)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # (b) Latency vs F1
    ax2.scatter(df_metrics["latency_ms"], df_metrics["macro_f1"], s=90, c=scatter_colors, edgecolors="black", linewidths=0.9, zorder=3)
    for _, row in df_metrics.iterrows():
        m = row["model"]
        s = short_names[m]
        f = float(row["macro_f1"])
        lat = float(row["latency_ms"])
        if m == "Late Fusion":
            ax2.annotate(f"{s} ({lat:.3f}ms)", (lat, f), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.0, fontweight="bold")
        elif m == "Early Fusion":
            ax2.annotate(f"{s} ({lat:.3f}ms)", (lat, f), xytext=(8, 5), textcoords="offset points", ha="left", fontsize=8.0, fontweight="bold")
        elif m == "Chest Accel":
            ax2.annotate(f"{s} ({lat:.3f}ms)", (lat, f), xytext=(8, 2), textcoords="offset points", ha="left", fontsize=8.0, fontweight="bold")
        elif m == "Ankle Accel":
            ax2.annotate(f"{s} ({lat:.3f}ms)", (lat, f), xytext=(8, -10), textcoords="offset points", ha="left", fontsize=8.0, fontweight="bold")
        else:
            ax2.annotate(f"{s} ({lat:.3f}ms)", (lat, f), xytext=(8, -3), textcoords="offset points", ha="left", fontsize=8.0, fontweight="bold")

    ax2.set_title("(b) Macro-F1 vs Latency", fontsize=9.5, fontweight="bold", pad=4)
    ax2.set_xlabel("Inference Latency per Window (ms)", fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("Test Macro-F1", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0.81, 0.99)
    ax2.set_xlim(0.024, 0.106)
    ax2.tick_params(axis="both", labelsize=8.0)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_computational_tradeoffs.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig8_computational_tradeoffs.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig8_computational_tradeoffs")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    print("=" * 70)
    print("MHEALTH Multimodal Fusion Pipeline - Starting Full Run")
    print("=" * 70)
    
    raw_dfs = load_raw_dataset()
    print("Plotting Task 1 Exploratory Data Analysis figures...")
    plot_task1_figures(raw_dfs)
    
    print("\nAuditing missing data and applying intra-segment interpolation...")
    cleaned_dfs, col_nans, total_samples, burst_lengths_per_col = analyze_and_clean_data(raw_dfs)
    plot_task2_figures(col_nans, total_samples, burst_lengths_per_col, raw_dfs, cleaned_dfs)
    
    print("\nExtracting sliding window features...")
    records = build_windowed_dataset(cleaned_dfs, window_size=128, step_size=64)
    
    print("\nTraining models and evaluating fusion strategies...")
    (
        all_models, histories, best_epochs, metrics_summary,
        per_class_summary, y_test, probs_chest, probs_ankle, probs_wrist, probs_early, probs_late
    ) = run_all_experiments(records)
    
    print("\nGenerating evaluation and trade-off figures...")
    plot_evaluation_figures(histories, all_models, y_test, metrics_summary)
    
    # Save results to disk
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv("ass2/metrics.csv", index=False)
    print("\nSaved ass2/metrics.csv:")
    print(df_metrics.to_string(index=False))
    
    # Per-class metrics table
    df_per_class = pd.DataFrame(per_class_summary)
    df_per_class.index = [ACTIVITY_NAMES[i] for i in range(1, 13)]
    df_per_class.to_csv("ass2/per_class_f1.csv")
    print("\nSaved ass2/per_class_f1.csv:")
    print(df_per_class.to_string())
    
    # JSON results
    with open("ass2/results.json", "w") as fp:
        json.dump({
            "metrics": metrics_summary,
            "per_class_f1": per_class_summary,
            "best_epochs": best_epochs
        }, fp, indent=2)
    print("\nSaved ass2/results.json.")
    print("=" * 70)
    print("Pipeline Complete! All figures, metrics, and models generated.")
    print("=" * 70)

if __name__ == "__main__":
    main()
