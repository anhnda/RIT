"""
REINFORCEMENT LEARNING: RNN Policy Network → XGBoost Judge

Architecture:
1. RNN processes temporal data → stochastic latent Z (policy, REINFORCE)
2. Concatenate [Static + Last_Values + Z] → XGBoost (reward judge)
3. XGBoost provides per-sample reward; policy is updated via policy gradient

Shared building blocks (SimpleStaticEncoder, HybridDataset, hybrid_collate_fn,
RNNPolicyNetwork, extract_features_and_logprobs) live in utils/rl_common.py.
"""

import sys
import os
import copy
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, auc,
)

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

xseed = 42
seed_everything(xseed)


# ==============================================================================
# RL Training Loop
# ==============================================================================

def train_policy_with_xgboost_reward(
    policy_net,
    train_loader,
    val_loader,
    xgb_params,
    epochs=100,
    update_xgb_every=5,
):
    """Train RNN policy using XGBoost as non-differentiable reward judge.

    Algorithm (REINFORCE):
    1. Sample Z from policy → features = [Static + Last + Z]
    2. Train/update XGBoost on those features
    3. Compute per-sample rewards from XGBoost predictions
    4. Update policy: loss = -E[log π(z|s) · R(z)]
    """
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0005)

    best_val_aupr  = 0.0
    best_state     = None
    patience       = 15
    patience_counter = 0

    print("  [RL Training] XGBoost reward judge...")

    xgb_model = None

    for epoch in range(epochs):
        policy_net.train()

        X_train, y_train, log_probs_train = extract_features_and_logprobs(
            policy_net, train_loader, deterministic=False
        )

        if epoch % update_xgb_every == 0 or xgb_model is None:
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(X_train, y_train, verbose=False)

        y_pred_proba = xgb_model.predict_proba(X_train)[:, 1]
        y_pred       = (y_pred_proba > 0.5).astype(int)

        # Combine binary correctness with smooth probability reward
        rewards_binary = (y_pred == y_train).astype(np.float32)
        rewards_smooth = np.where(y_train == 1, y_pred_proba, 1 - y_pred_proba)
        rewards_combined = 0.5 * rewards_binary + 0.5 * rewards_smooth

        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss     = -(log_probs_train * rewards_tensor).mean()
        entropy_bonus   = 0.01 * log_probs_train.mean()
        total_loss      = policy_loss - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val, y_val, _ = extract_features_and_logprobs(
                    policy_net, val_loader, deterministic=True
                )

            xgb_val = XGBClassifier(**xgb_params)
            xgb_val.fit(X_train, y_train, verbose=False)
            y_val_proba = xgb_val.predict_proba(X_val)[:, 1]

            val_auc  = roc_auc_score(y_val, y_val_proba)
            val_aupr = average_precision_score(y_val, y_val_proba)

            print(f"    Epoch {epoch+1:3d} | Reward: {rewards_combined.mean():.4f} | "
                  f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_state    = copy.deepcopy(policy_net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    return policy_net


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print("RL POLICY (RNN + XGBoost)")
    print("=" * 80)

    patients      = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)
    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    metrics_rl = {k: [] for k in ["auc", "auc_pr"]}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}\nFold {fold}\n{'='*80}")

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42 + fold)
        train_p     = train_p_obj.patientList
        val_p       = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats    = train_ds.get_normalization_stats()
        val_ds   = HybridDataset(val_p,       temporal_feats, encoder, stats)
        test_ds  = HybridDataset(test_p_list, temporal_feats, encoder, stats)

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
        xgb_params = {
            "n_estimators":    200,
            "max_depth":       4,
            "learning_rate":   0.05,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": ratio,
            "random_state":    42,
            "eval_metric":     "auc",
        }

        policy_net = train_policy_with_xgboost_reward(
            policy_net, train_loader, val_loader, xgb_params,
            epochs=100, update_xgb_every=5,
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

        final_xgb = XGBClassifier(**xgb_params)
        final_xgb.fit(X_train_f, y_train_f, verbose=False)

        y_test_proba = final_xgb.predict_proba(X_test_f)[:, 1]
        y_test_pred  = (y_test_proba > 0.5).astype(int)

        prec, rec, _ = precision_recall_curve(y_test_f, y_test_proba)
        fold_auc  = roc_auc_score(y_test_f, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_rl["auc"].append(fold_auc)
        metrics_rl["auc_pr"].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_f, y_test_proba)
        ax.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC={fold_auc:.3f})")

        print(f"  Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("RL Policy (RNN + XGBoost)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("result", exist_ok=True)
    plt.savefig("result/xg_rl.png", dpi=300)
    print("\nPlot saved to result/xg_rl.png")

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    def print_stat(name, vals):
        print(f"{name:15s} | {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print_stat("AUC",    metrics_rl["auc"])
    print_stat("AUC-PR", metrics_rl["auc_pr"])


if __name__ == "__main__":
    main()
