"""
FEATURE EXTRACTION FOR CatBoostRL USING SHAP

Two modes:
1. train_model : Train CatBoostRL on all folds, save the best fold's model and data
2. extract     : Use SHAP TreeExplainer on the saved model
3. interpret   : Trace latent factors back to temporal input features via correlation
                 and gradient-based attribution
"""

import sys
import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from catboost import CatBoostClassifier
import pickle
import argparse
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from scipy.stats import pearsonr
import shap

PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from utils.rl_common import (
    FIXED_FEATURES, seed_everything,
    SimpleStaticEncoder, HybridDataset, hybrid_collate_fn,
    RNNPolicyNetwork, extract_features_and_logprobs,
)
from TimeEmbedding import DEVICE
from TimeEmbeddingVal import (
    get_all_temporal_features, load_and_prepare_patients, split_patients_train_val,
)
from utils.prepare_data import trainTestPatients
from CatBoostRL import train_policy_with_catboost_reward

xseed = 42
seed_everything(xseed)


# ==============================================================================
# MODE 1: Train and save best fold
# ==============================================================================

def train_and_save_best_fold(output_dir="models/catboostrl"):
    """Train CatBoostRL on all folds, save the best fold's artifacts."""
    print("=" * 80)
    print("MODE 1: TRAINING AND SAVING BEST FOLD")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    patients       = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)
    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    best_fold_idx  = -1
    best_fold_aupr = 0.0
    best_fold_data = None

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}\nFold {fold}\n{'='*80}")

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42 + fold)
        train_p     = train_p_obj.patientList
        val_p       = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = HybridDataset(train_p,       temporal_feats, encoder)
        stats    = train_ds.get_normalization_stats()
        val_ds   = HybridDataset(val_p,          temporal_feats, encoder, stats)
        test_ds  = HybridDataset(test_p_list,    temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=hybrid_collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        latent_dim = 16
        policy_net = RNNPolicyNetwork(
            input_dim=len(temporal_feats), hidden_dim=12,
            latent_dim=latent_dim, time_dim=32,
        ).to(DEVICE)

        ratio = (
            sum(1 for _, l, _ in train_ds if l == 0)
            / max(1, sum(1 for _, l, _ in train_ds if l == 1))
        )
        catboost_params = {
            "iterations":       200,
            "depth":            4,
            "learning_rate":    0.05,
            "loss_function":    "Logloss",
            "eval_metric":      "AUC",
            "scale_pos_weight": ratio,
            "random_seed":      42,
            "verbose":          False,
            "allow_writing_files": False,
            "task_type":        "CPU",
        }

        policy_net = train_policy_with_catboost_reward(
            policy_net, train_loader, val_loader, catboost_params,
            epochs=100, update_catboost_every=5,
        )

        print("\n  [Final Test Evaluation]")
        policy_net.eval()
        with torch.no_grad():
            X_train_f, y_train_f, _ = extract_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_f, y_test_f, _ = extract_features_and_logprobs(
                policy_net, test_loader, deterministic=True
            )

        final_cb = CatBoostClassifier(**catboost_params)
        final_cb.fit(X_train_f, y_train_f, verbose=False)

        y_test_proba = final_cb.predict_proba(X_test_f)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test_f, y_test_proba)
        fold_auc  = roc_auc_score(y_test_f, y_test_proba)
        fold_aupr = auc(rec, prec)

        print(f"  Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

        if fold_aupr > best_fold_aupr:
            best_fold_aupr = fold_aupr
            best_fold_idx  = fold
            best_fold_data = {
                "policy_net_state": copy.deepcopy(policy_net.state_dict()),
                "catboost_model":   final_cb,
                "X_train":          X_train_f,
                "y_train":          y_train_f,
                "X_test":           X_test_f,
                "y_test":           y_test_f,
                "encoder":          encoder,
                "temporal_feats":   temporal_feats,
                "stats":            stats,
                "latent_dim":       latent_dim,
                "catboost_params":  catboost_params,
                "fold_idx":         fold,
                "fold_auc":         fold_auc,
                "fold_aupr":        fold_aupr,
            }

    if best_fold_data is None:
        print("No valid fold found!")
        return

    print(f"\n{'='*80}")
    print(f"BEST FOLD: {best_fold_idx} (AUPR: {best_fold_aupr:.4f})")
    print(f"Saving to {output_dir}/")
    print("=" * 80)

    torch.save(
        best_fold_data["policy_net_state"],
        os.path.join(output_dir, "policy_net.pth"),
    )
    best_fold_data["catboost_model"].save_model(
        os.path.join(output_dir, "catboost_model.cbm")
    )
    with open(os.path.join(output_dir, "fold_data.pkl"), "wb") as f:
        save_keys = [
            "X_train", "y_train", "X_test", "y_test",
            "encoder", "temporal_feats", "stats",
            "latent_dim", "catboost_params",
            "fold_idx", "fold_auc", "fold_aupr",
        ]
        pickle.dump({k: best_fold_data[k] for k in save_keys}, f)

    print("Saved: policy_net.pth, catboost_model.cbm, fold_data.pkl")


# ==============================================================================
# MODE 2: Extract SHAP feature contributions
# ==============================================================================

def extract_features_with_shap(model_dir="models/catboostrl", top_k=20):
    """Load saved model and use SHAP TreeExplainer for feature attribution."""
    print("=" * 80)
    print("MODE 2: EXTRACTING FEATURES WITH SHAP")
    print("=" * 80)

    with open(os.path.join(model_dir, "fold_data.pkl"), "rb") as f:
        data = pickle.load(f)

    catboost_model = CatBoostClassifier()
    catboost_model.load_model(os.path.join(model_dir, "catboost_model.cbm"))

    X_train        = data["X_train"]
    X_test         = data["X_test"]
    y_test         = data["y_test"]
    temporal_feats = data["temporal_feats"]
    latent_dim     = data["latent_dim"]

    print(f"Fold {data['fold_idx']} (AUC: {data['fold_auc']:.4f}, AUPR: {data['fold_aupr']:.4f})")

    n_static   = len(FIXED_FEATURES)
    n_temporal = len(temporal_feats)

    feature_names = (
        [f"static_{f}"  for f in FIXED_FEATURES]
        + [f"last_{f}"  for f in temporal_feats]
        + [f"latent_z{i}" for i in range(latent_dim)]
    )

    static_indices         = list(range(n_static))
    temporal_last_indices  = list(range(n_static, n_static + n_temporal))
    latent_indices         = list(range(n_static + n_temporal, n_static + n_temporal + latent_dim))
    canonical_indices      = static_indices + temporal_last_indices

    print("\nComputing SHAP values for test set...")
    explainer   = shap.TreeExplainer(catboost_model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    def _print_top(title, indices, names, top):
        vals = np.abs(shap_values[:, indices]).mean(axis=0)
        sorted_idx = np.argsort(vals)[::-1]
        print(f"\n{title}")
        print("-" * 70)
        print(f"{'Rank':<6} {'Feature':<40} {'Mean |SHAP|':<15}")
        print("-" * 70)
        for rank, i in enumerate(sorted_idx[:top]):
            print(f"{rank+1:<6} {names[i]:<40} {vals[i]:<15.6f}")

    _print_top("CANONICAL FEATURES (Static + Last Values)",
               canonical_indices,
               [feature_names[i] for i in canonical_indices], top_k)

    _print_top("LATENT FEATURES (Learned Z)",
               latent_indices,
               [feature_names[i] for i in latent_indices], top_k)

    mean_abs_all = np.abs(shap_values).mean(axis=0)
    sorted_all   = np.argsort(mean_abs_all)[::-1]
    print(f"\nOVERALL TOP {top_k} FEATURES")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<40} {'Mean |SHAP|':<15} {'Type':<8}")
    print("-" * 80)
    for rank, idx in enumerate(sorted_all[:top_k]):
        ftype = "Static" if idx in static_indices else ("Last" if idx in temporal_last_indices else "Latent")
        print(f"{rank+1:<6} {feature_names[idx]:<40} {mean_abs_all[idx]:<15.6f} {ftype:<8}")

    # Contribution by type
    sc = mean_abs_all[static_indices].sum()
    lc = mean_abs_all[temporal_last_indices].sum()
    zc = mean_abs_all[latent_indices].sum()
    tot = sc + lc + zc
    print(f"\nContribution by type:")
    print(f"  Static  : {sc:.4f} ({100*sc/tot:.1f}%)")
    print(f"  Last    : {lc:.4f} ({100*lc/tot:.1f}%)")
    print(f"  Latent Z: {zc:.4f} ({100*zc/tot:.1f}%)")

    output_file = os.path.join(model_dir, "shap_results.pkl")
    with open(output_file, "wb") as f:
        pickle.dump({
            "shap_values":            shap_values,
            "feature_names":          feature_names,
            "canonical_indices":      canonical_indices,
            "latent_indices":         latent_indices,
            "mean_abs_shap_all":      mean_abs_all,
            "mean_abs_shap_canonical": np.abs(shap_values[:, canonical_indices]).mean(axis=0),
            "mean_abs_shap_latent":   np.abs(shap_values[:, latent_indices]).mean(axis=0),
            "X_test":                 X_test,
            "y_test":                 y_test,
        }, f)
    print(f"\nSHAP results saved to {output_file}")
    return shap_values, feature_names


# ==============================================================================
# MODE 3: Interpret latent factors
# ==============================================================================

def interpret_latent_factors(model_dir="models/catboostrl", top_k=10):
    """Trace latent dimensions back to temporal input features via correlation
    and gradient-based attribution."""
    print("=" * 80)
    print("MODE 3: INTERPRETING LATENT FACTORS")
    print("=" * 80)

    with open(os.path.join(model_dir, "fold_data.pkl"), "rb") as f:
        data = pickle.load(f)

    temporal_feats = data["temporal_feats"]
    latent_dim     = data["latent_dim"]
    fold_idx       = data["fold_idx"]
    encoder        = data["encoder"]
    stats          = data["stats"]

    policy_net = RNNPolicyNetwork(
        input_dim=len(temporal_feats), hidden_dim=12,
        latent_dim=latent_dim, time_dim=32,
    ).to(DEVICE)
    policy_net.load_state_dict(
        torch.load(os.path.join(model_dir, "policy_net.pth"), map_location=DEVICE)
    )
    policy_net.eval()

    patients = load_and_prepare_patients()
    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        if fold == fold_idx:
            test_p_list = test_p.patientList
            break

    test_ds     = HybridDataset(test_p_list, temporal_feats, encoder, stats)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

    all_temporal_inputs = []
    all_temporal_masks  = []
    all_latent_outputs  = []
    all_labels          = []

    with torch.no_grad():
        for t_data, labels, _ in test_loader:
            _, _, mean = policy_net(t_data, deterministic=True)
            all_latent_outputs.append(mean.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_temporal_inputs.append(t_data["values"].cpu().numpy())
            all_temporal_masks.append(t_data["masks"].cpu().numpy())

    latent_outputs   = np.vstack(all_latent_outputs)
    n_samples        = len(all_labels)
    n_temporal_feats = len(temporal_feats)

    # --- Correlation analysis ---
    print("\nComputing temporal-to-latent correlations...")
    temporal_aggs = np.zeros((n_samples, n_temporal_feats * 4))  # mean, std, min, max
    sample_idx    = 0
    for batch_vals, batch_masks in zip(all_temporal_inputs, all_temporal_masks):
        for i in range(len(batch_vals)):
            for f_idx in range(n_temporal_feats):
                valid = batch_vals[i, :, f_idx][batch_masks[i, :, f_idx] > 0]
                if len(valid) > 0:
                    temporal_aggs[sample_idx, f_idx * 4 + 0] = np.mean(valid)
                    temporal_aggs[sample_idx, f_idx * 4 + 1] = np.std(valid) if len(valid) > 1 else 0.0
                    temporal_aggs[sample_idx, f_idx * 4 + 2] = np.min(valid)
                    temporal_aggs[sample_idx, f_idx * 4 + 3] = np.max(valid)
            sample_idx += 1

    latent_to_temporal = {}
    agg_names = ["mean", "std", "min", "max"]
    for z_idx in range(latent_dim):
        z_vals = latent_outputs[:, z_idx]
        corrs  = []
        for f_idx in range(n_temporal_feats):
            best_corr, best_agg = 0.0, "n/a"
            for a, name in enumerate(agg_names):
                col = temporal_aggs[:, f_idx * 4 + a]
                if np.std(col) > 1e-6:
                    c = abs(pearsonr(col, z_vals)[0])
                    if not np.isnan(c) and c > best_corr:
                        best_corr, best_agg = c, name
            corrs.append((temporal_feats[f_idx], best_corr, best_agg))
        corrs.sort(key=lambda x: x[1], reverse=True)
        latent_to_temporal[z_idx] = corrs

    shap_file = os.path.join(model_dir, "shap_results.pkl")
    if os.path.exists(shap_file):
        with open(shap_file, "rb") as f:
            shap_data = pickle.load(f)
        mean_abs_shap_latent = shap_data["mean_abs_shap_latent"]
        sorted_latent_idx    = np.argsort(mean_abs_shap_latent)[::-1]
    else:
        sorted_latent_idx = list(range(latent_dim))

    print("\nTop latent dims × temporal feature correlations:")
    for rank, z_idx in enumerate(sorted_latent_idx[:min(10, latent_dim)]):
        print(f"\nLatent Z{z_idx}")
        print(f"  {'Rank':<5} {'Feature':<30} {'Corr':<10} {'Agg'}")
        for i, (feat, corr, agg) in enumerate(latent_to_temporal[z_idx][:top_k]):
            print(f"  {i+1:<5} {feat:<30} {corr:<10.4f} {agg}")

    # --- Gradient attribution ---
    print("\nComputing gradient-based attribution...")
    policy_net.train()
    grad_attr = {z_idx: np.zeros(n_temporal_feats) for z_idx in range(latent_dim)}

    for t_data, _, _ in test_loader:
        t_vals = t_data["values"].to(DEVICE).requires_grad_(True)
        t_data_grad = {
            "times":   t_data["times"].to(DEVICE),
            "values":  t_vals,
            "masks":   t_data["masks"].to(DEVICE),
            "lengths": t_data["lengths"].to(DEVICE),
        }
        _, _, mean = policy_net(t_data_grad, deterministic=True)
        for z_idx in range(latent_dim):
            if z_idx < mean.shape[1]:
                mean[:, z_idx].sum().backward(retain_graph=True)
                grads = t_vals.grad
                masks = t_data_grad["masks"]
                for f_idx in range(n_temporal_feats):
                    valid_count = masks[:, :, f_idx].sum().item()
                    if valid_count > 0:
                        grad_attr[z_idx][f_idx] += (
                            torch.abs(grads[:, :, f_idx] * masks[:, :, f_idx]).sum().item()
                            / valid_count
                        )
                t_vals.grad.zero_()

    for z_idx in range(latent_dim):
        grad_attr[z_idx] /= len(test_loader)

    print("\nTop latent dims × gradient attribution:")
    for z_idx in sorted_latent_idx[:min(10, latent_dim)]:
        pairs = [(temporal_feats[i], grad_attr[z_idx][i]) for i in range(n_temporal_feats)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        print(f"\nLatent Z{z_idx}")
        print(f"  {'Rank':<5} {'Feature':<30} {'Grad Attr'}")
        for i, (feat, g) in enumerate(pairs[:top_k]):
            print(f"  {i+1:<5} {feat:<30} {g:.6f}")

    out_file = os.path.join(model_dir, "latent_interpretation.pkl")
    with open(out_file, "wb") as f:
        pickle.dump({
            "latent_to_temporal_corr": latent_to_temporal,
            "gradient_attributions":   grad_attr,
            "temporal_features":       temporal_feats,
            "latent_dim":              latent_dim,
        }, f)
    print(f"\nLatent interpretation saved to {out_file}")
    policy_net.eval()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction for CatBoostRL")
    parser.add_argument(
        "--mode", required=True,
        choices=["train_model", "extract", "interpret"],
        help="train_model | extract | interpret",
    )
    parser.add_argument("--output_dir", default="models/catboostrl")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "train_model":
        train_and_save_best_fold(output_dir=args.output_dir)
    elif args.mode == "extract":
        extract_features_with_shap(model_dir=args.output_dir, top_k=args.top_k)
    elif args.mode == "interpret":
        interpret_latent_factors(model_dir=args.output_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
