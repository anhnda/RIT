"""
TabPFN BASELINE EXT: Enriched Temporal Stats + Static Context → TabPFN

Extends the basic TabPFN baseline by replacing raw last-values with six
per-feature temporal statistics extracted from the time-series window:
  - Last value, Mean, Std, Min, Max, Slope (last − first)

These are concatenated with static context features and fed directly into
TabPFN — no RNN, no reinforcement learning.

Mutual-information feature selection (top-80) is applied before TabPFN to
stay within its recommended feature-count limit.
"""

import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import torch

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc,
)

os.environ["SEGMENT_WRITE_KEY"]        = ""
os.environ["ANALYTICS_WRITE_KEY"]      = ""
os.environ["TABPFN_DISABLE_ANALYTICS"] = "1"

try:
    import analytics
    analytics.write_key = None
    analytics.disable()
except Exception:
    pass

PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from utils.rl_common import FIXED_FEATURES, seed_everything, SimpleStaticEncoder, select_features_mi
from TimeEmbeddingVal import (
    get_all_temporal_features, extract_temporal_data, load_and_prepare_patients,
)
from utils.prepare_data import trainTestPatients

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: tabpfn not installed. Run: pip install tabpfn")

xseed = 42
seed_everything(xseed)


# ==============================================================================
# Enriched Temporal Feature Extraction
# ==============================================================================

def extract_enriched_temporal_stats(patients, feature_names, static_encoder,
                                     normalization_stats=None):
    """Extract 6 temporal statistics per feature plus static features.

    Parameters
    ----------
    patients           : Patients object or list of Patient
    feature_names      : list of temporal feature names
    static_encoder     : fitted SimpleStaticEncoder
    normalization_stats: dict {'mean', 'std'} from training split, or None

    Returns
    -------
    X          : np.ndarray (n_patients, n_static + 6 * n_temporal)
    y          : np.ndarray (n_patients,)
    norm_stats : dict {'mean', 'std'}
    """
    patient_list = patients.patientList if hasattr(patients, "patientList") else patients

    if normalization_stats is None:
        all_values = []
        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue
            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0:
                        all_values.append(v)
        all_values = np.array(all_values) if all_values else np.array([0.0])
        norm_mean  = float(np.mean(all_values))
        norm_std   = float(np.std(all_values)) if np.std(all_values) > 0 else 1.0
    else:
        norm_mean = normalization_stats["mean"]
        norm_std  = normalization_stats["std"]

    norm_stats = {"mean": norm_mean, "std": norm_std}

    X_rows = []
    y_list = []

    for patient in patient_list:
        times, values, masks = extract_temporal_data(patient, feature_names)
        if times is None:
            continue

        norm_values = [
            [(v - norm_mean) / norm_std if m > 0 else 0.0
             for v, m in zip(v_vec, m_vec)]
            for v_vec, m_vec in zip(values, masks)
        ]
        values_arr = np.array(norm_values)   # (T, F)
        masks_arr  = np.array(masks)         # (T, F)

        n_feats    = len(feature_names)
        last_vals  = np.zeros(n_feats)
        mean_vals  = np.zeros(n_feats)
        std_vals   = np.zeros(n_feats)
        min_vals   = np.zeros(n_feats)
        max_vals   = np.zeros(n_feats)
        slope_vals = np.zeros(n_feats)

        for f_idx in range(n_feats):
            valid_idx = np.where(masks_arr[:, f_idx] > 0)[0]
            if len(valid_idx) > 0:
                vv = values_arr[valid_idx, f_idx]
                last_vals[f_idx]  = vv[-1]
                mean_vals[f_idx]  = np.mean(vv)
                std_vals[f_idx]   = np.std(vv) if len(vv) > 1 else 0.0
                min_vals[f_idx]   = np.min(vv)
                max_vals[f_idx]   = np.max(vv)
                slope_vals[f_idx] = vv[-1] - vv[0] if len(vv) > 1 else 0.0

        static_vec = np.array(static_encoder.transform(patient), dtype=np.float32)
        row = np.concatenate([
            static_vec,
            last_vals, mean_vals, std_vals,
            min_vals,  max_vals,  slope_vals,
        ])
        X_rows.append(row)
        y_list.append(1 if patient.akdPositive else 0)

    X = np.vstack(X_rows).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, norm_stats


# ==============================================================================
# Main
# ==============================================================================

def main():
    if not TABPFN_AVAILABLE:
        print("Error: TabPFN is not available. Please install it first.")
        return

    print("=" * 80)
    print("TabPFN BASELINE EXT: Enriched Temporal Stats + Static Context")
    print("=" * 80)

    patients       = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print(f"Temporal features : {len(temporal_feats)}")
    print(f"Static features   : {len(FIXED_FEATURES)}")
    print(f"Total raw dims    : {len(temporal_feats) * 6 + len(FIXED_FEATURES)}")

    static_encoder = SimpleStaticEncoder(FIXED_FEATURES)
    static_encoder.fit(patients.patientList)

    metrics = {k: [] for k in ["auc", "acc", "spec", "prec", "rec", "auc_pr"]}
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n--- Fold {fold} ---")

        train_list = train_full.patientList if hasattr(train_full, "patientList") else train_full
        test_list  = test_p.patientList    if hasattr(test_p,    "patientList") else test_p

        print("  Extracting enriched temporal statistics...")
        X_train, y_train, norm_stats = extract_enriched_temporal_stats(
            train_list, temporal_feats, static_encoder
        )
        X_test, y_test, _ = extract_enriched_temporal_stats(
            test_list, temporal_feats, static_encoder, normalization_stats=norm_stats
        )

        print(f"    Train: {X_train.shape[0]} × {X_train.shape[1]}")
        print(f"    Test : {X_test.shape[0]} × {X_test.shape[1]}")

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

        top_k = min(80, X_train.shape[1])
        X_train_sel, X_test_sel, _ = select_features_mi(X_train, y_train, X_test, top_k=top_k)
        print(f"    After MI selection: {X_train_sel.shape[1]} features")

        # Subsample if needed for TabPFN's row limit
        max_train = 3000
        if len(X_train_sel) > max_train:
            print(f"    Subsampling to {max_train} rows (TabPFN limit)")
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            pos_ratio = len(pos_idx) / len(y_train)
            n_pos = int(max_train * pos_ratio)
            n_neg = max_train - n_pos
            s_pos = np.random.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
            s_neg = np.random.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
            sel   = np.random.permutation(np.concatenate([s_pos, s_neg]))
            X_train_sel = X_train_sel[sel]
            y_train     = y_train[sel]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Training TabPFN (device={device})...")
        tabpfn = TabPFNClassifier(device=device, n_estimators=16)
        tabpfn.fit(X_train_sel, y_train)

        y_prob = tabpfn.predict_proba(X_test_sel)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)

        fold_auc    = roc_auc_score(y_test, y_prob)
        fold_auc_pr = auc(rec_c, prec_c)

        metrics["auc"].append(fold_auc)
        metrics["auc_pr"].append(fold_auc_pr)
        metrics["acc"].append(accuracy_score(y_test, y_pred))
        metrics["spec"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        metrics["prec"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["rec"].append(recall_score(y_test, y_pred, zero_division=0))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC={fold_auc:.3f})")

        print(f"  Fold {fold} → AUC: {fold_auc:.4f} | AUC-PR: {fold_auc_pr:.4f}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("TabPFN Baseline Ext: Enriched Stats + Static Context",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("result", exist_ok=True)
    plt.savefig("result/tabpfn_baseline_ext.png", dpi=300)
    print("\nPlot saved to result/tabpfn_baseline_ext.png")

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    def print_stat(name, vals):
        print(f"{name:15s} | {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print_stat("AUC",         metrics["auc"])
    print_stat("AUC-PR",      metrics["auc_pr"])
    print_stat("Accuracy",    metrics["acc"])
    print_stat("Specificity", metrics["spec"])
    print_stat("Precision",   metrics["prec"])
    print_stat("Recall",      metrics["rec"])


if __name__ == "__main__":
    main()
