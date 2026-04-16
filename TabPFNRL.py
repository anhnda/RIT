"""
REINFORCEMENT LEARNING V5: Fixed Normal Variance for Z

Key change from V4: Policy network outputs ONLY the mean.
Variance is fixed at 1.0 (scaled by temperature for exploration).
This removes the noisy learned std that RL couldn't optimize.

Other improvements carried from V4:
- Adaptive RL guard (revert if degraded)
- Smaller latent_dim=12, larger hidden_dim=64
- Per-sample precision-weighted rewards
- Feature selection (MI top-K)
- Cosine LR schedule
- Pretrained/RL ensemble blend
- TabPFN n_estimators=16
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import random
import os
import math

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)
from sklearn.feature_selection import mutual_info_classif

os.environ["SEGMENT_WRITE_KEY"] = ""
os.environ["ANALYTICS_WRITE_KEY"] = ""
os.environ["TABPFN_DISABLE_ANALYTICS"] = "1"

try:
    import analytics
    analytics.write_key = None
    analytics.disable()
except Exception:
    pass

from tabpfn import TabPFNClassifier

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

xseed = 42
seed_everything(xseed)

PT = "/Users/anhnd/CodingSpace/Python/PREDKIT"
if sys.platform != "darwin":
    PT = "/home/anhnda/PREKIT"
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients, encodeCategoricalData
from TimeEmbeddingVal import (
    get_all_temporal_features,
    extract_temporal_data,
    load_and_prepare_patients,
    split_patients_train_val,
)
from TimeEmbedding import DEVICE, TimeEmbeddedRNNCell

FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable"
]

# ==============================================================================
# 1. Static Encoder
# ==============================================================================

class SimpleStaticEncoder:
    def __init__(self, features):
        self.features = features
        self.mappings = {f: {} for f in features}
        self.counts = {f: 0 for f in features}

    def fit(self, patients):
        for p in patients:
            for f in self.features:
                val = p.measures.get(f, 0.0)
                if hasattr(val, 'values') and len(val) > 0:
                    val = list(val.values())[0]
                elif hasattr(val, 'values'):
                    val = 0.0
                val_str = str(val)
                try:
                    float(val)
                except ValueError:
                    if val_str not in self.mappings[f]:
                        self.mappings[f][val_str] = float(self.counts[f])
                        self.counts[f] += 1

    def transform(self, patient):
        vec = []
        for f in self.features:
            val = patient.measures.get(f, 0.0)
            if hasattr(val, 'values') and len(val) > 0:
                val = list(val.values())[0]
            elif hasattr(val, 'values'):
                val = 0.0
            try:
                numeric_val = float(val)
            except ValueError:
                numeric_val = self.mappings[f].get(str(val), -1.0)
            vec.append(numeric_val)
        return vec

# ==============================================================================
# 2. Dataset
# ==============================================================================

class HybridDataset(Dataset):
    def __init__(self, patients, feature_names, static_encoder, normalization_stats=None):
        self.data = []
        self.labels = []
        self.static_data = []
        self.feature_names = feature_names

        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue
            s_vec = static_encoder.transform(patient)
            self.static_data.append(torch.tensor(s_vec, dtype=torch.float32))
            self.data.append({'times': times, 'values': values, 'masks': masks})
            self.labels.append(1 if patient.akdPositive else 0)
            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0:
                        all_values.append(v)

        if normalization_stats is None:
            all_values = np.array(all_values)
            self.mean = np.mean(all_values) if len(all_values) > 0 else 0.0
            self.std = np.std(all_values) if len(all_values) > 0 else 1.0
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']

        for i in range(len(self.data)):
            norm_values = []
            for v_vec, m_vec in zip(self.data[i]['values'], self.data[i]['masks']):
                norm = [(v - self.mean)/self.std if m > 0 else 0.0
                        for v, m in zip(v_vec, m_vec)]
                norm_values.append(norm)
            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(norm_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32)
            }

    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.static_data[idx]

def hybrid_collate_fn(batch):
    data_list, label_list, static_list = zip(*batch)
    lengths = [len(d['times']) for d in data_list]
    max_len = max(lengths)
    feat_dim = data_list[0]['values'].shape[-1]
    batch_size = len(data_list)

    padded_times = torch.zeros(batch_size, max_len)
    padded_values = torch.zeros(batch_size, max_len, feat_dim)
    padded_masks = torch.zeros(batch_size, max_len, feat_dim)

    for i, d in enumerate(data_list):
        l = lengths[i]
        padded_times[i, :l] = d['times']
        padded_values[i, :l] = d['values']
        padded_masks[i, :l] = d['masks']

    temporal_batch = {
        'times': padded_times,
        'values': padded_values,
        'masks': padded_masks,
        'lengths': torch.tensor(lengths)
    }
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)

# ==============================================================================
# 3. RNN Policy Network — FIXED UNIT VARIANCE
# ==============================================================================

class RNNPolicyNetwork(nn.Module):
    """
    RNN policy that outputs ONLY the mean.
    Variance is fixed at 1.0 (no learned log_std).
    During stochastic mode: z = mean + temperature * eps, eps ~ N(0, I)
    During deterministic mode: z = mean
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)

        # Only mean — no fc_logstd!
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, batch_data, deterministic=False, temperature=0.05):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        h = self.rnn_cell(times, values, masks, lengths)
        mean = self.fc_mean(h)

        if deterministic:
            z = mean
            log_prob = None
        else:
            eps = torch.randn_like(mean)
            z = mean + temperature * eps
            diff = z.detach() - mean  # numerically = temperature*eps - 0, grad = -1 w.r.t mean
            log_prob = -0.5 * (diff**2).sum(dim=-1) / (temperature**2)

        return z, log_prob, mean

class SupervisedHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

# ==============================================================================
# 4. Enhanced Pretraining
# ==============================================================================

def pretrain_rnn_enhanced(policy_net, train_loader, val_loader, epochs=50):
    print("  [Enhanced Pretraining] Longer supervised learning...")

    supervised_head = SupervisedHead(
        policy_net.latent_dim + len(FIXED_FEATURES)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(supervised_head.parameters()),
        lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()

    best_auc = 0
    best_state = None
    patience = 12
    counter = 0

    for epoch in range(epochs):
        policy_net.train()
        supervised_head.train()

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)
            z, _, _ = policy_net(t_data, deterministic=True)
            combined = torch.cat([z, s_data], dim=1)
            preds = supervised_head(combined).squeeze(-1)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(supervised_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

        if (epoch + 1) % 3 == 0:
            policy_net.eval()
            supervised_head.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    z, _, _ = policy_net(t_data, deterministic=True)
                    combined = torch.cat([z, s_data], dim=1)
                    preds = supervised_head(combined).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            val_aupr = average_precision_score(all_labels, all_preds)
            scheduler.step(val_aupr)
            print(f"    Pretrain Epoch {epoch+1} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_auc:
                best_auc = val_aupr
                best_state = copy.deepcopy(policy_net.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    if best_state is not None:
        policy_net.load_state_dict(best_state)
    print(f"  [Enhanced Pretraining] Completed. Best Val AUPR: {best_auc:.4f}")
    return policy_net

# ==============================================================================
# 5. Feature Extraction (enriched temporal stats)
# ==============================================================================

def extract_enriched_features_and_logprobs(policy_net, loader, deterministic=False, temperature=1.0):
    policy_net.eval() if deterministic else policy_net.train()

    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data in loader:
            z, log_prob, mean = policy_net(t_data, deterministic=deterministic, temperature=temperature)
            z_np = (mean if deterministic else z).detach().cpu().numpy()

            vals = t_data['values'].cpu().numpy()
            masks = t_data['masks'].cpu().numpy()

            batch_last_vals = []
            batch_mean_vals = []
            batch_std_vals = []
            batch_min_vals = []
            batch_max_vals = []
            batch_slope_vals = []

            for i in range(len(vals)):
                patient_last = []
                patient_mean = []
                patient_std = []
                patient_min = []
                patient_max = []
                patient_slope = []

                for f_idx in range(vals.shape[2]):
                    f_vals = vals[i, :, f_idx]
                    f_mask = masks[i, :, f_idx]
                    valid_idx = np.where(f_mask > 0)[0]

                    if len(valid_idx) > 0:
                        valid_vals = f_vals[valid_idx]
                        patient_last.append(valid_vals[-1])
                        patient_mean.append(np.mean(valid_vals))
                        patient_std.append(np.std(valid_vals) if len(valid_vals) > 1 else 0.0)
                        patient_min.append(np.min(valid_vals))
                        patient_max.append(np.max(valid_vals))
                        if len(valid_vals) > 1:
                            patient_slope.append(valid_vals[-1] - valid_vals[0])
                        else:
                            patient_slope.append(0.0)
                    else:
                        patient_last.append(0.0)
                        patient_mean.append(0.0)
                        patient_std.append(0.0)
                        patient_min.append(0.0)
                        patient_max.append(0.0)
                        patient_slope.append(0.0)

                batch_last_vals.append(patient_last)
                batch_mean_vals.append(patient_mean)
                batch_std_vals.append(patient_std)
                batch_min_vals.append(patient_min)
                batch_max_vals.append(patient_max)
                batch_slope_vals.append(patient_slope)

            last_vals_arr = np.array(batch_last_vals)
            mean_vals_arr = np.array(batch_mean_vals)
            std_vals_arr = np.array(batch_std_vals)
            min_vals_arr = np.array(batch_min_vals)
            max_vals_arr = np.array(batch_max_vals)
            slope_vals_arr = np.array(batch_slope_vals)
            s_np = s_data.numpy()

            combined = np.hstack([
                s_np, last_vals_arr, mean_vals_arr, std_vals_arr,
                min_vals_arr, max_vals_arr, slope_vals_arr, z_np
            ])

            all_features.append(combined)
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs

# ==============================================================================
# 6. Feature Selection
# ==============================================================================

def select_features_mi(X_train, y_train, X_val_or_test, top_k=80):
    X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    mi = mutual_info_classif(X_train_clean, y_train, random_state=42, n_neighbors=5)
    top_indices = np.argsort(mi)[-top_k:]
    top_indices = np.sort(top_indices)
    return X_train[:, top_indices], X_val_or_test[:, top_indices], top_indices

# ==============================================================================
# 7. Adaptive RL Training
# ==============================================================================

def evaluate_with_tabpfn(policy_net, train_loader, val_loader, tabpfn_params, feat_indices=None):
    policy_net.eval()
    with torch.no_grad():
        X_train, y_train, _ = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=True
        )
        X_val, y_val, _ = extract_enriched_features_and_logprobs(
            policy_net, val_loader, deterministic=True
        )

    if feat_indices is not None:
        X_train = X_train[:, feat_indices]
        X_val = X_val[:, feat_indices]

    model = TabPFNClassifier(**tabpfn_params)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, y_proba)
    val_aupr = average_precision_score(y_val, y_proba)
    return val_auc, val_aupr


def train_policy_adaptive_rl(
    policy_net,
    train_loader,
    val_loader,
    tabpfn_params,
    epochs=40,
    update_tabpfn_every=5,
    top_k_features=80,
):
    # Save pre-RL state and evaluate baseline
    pre_rl_state = copy.deepcopy(policy_net.state_dict())

    policy_net.eval()
    with torch.no_grad():
        X_train_pre, y_train_pre, _ = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=True
        )
        X_val_pre, y_val_pre, _ = extract_enriched_features_and_logprobs(
            policy_net, val_loader, deterministic=True
        )

    _, _, feat_indices = select_features_mi(X_train_pre, y_train_pre, X_val_pre, top_k=top_k_features)

    pre_rl_auc, pre_rl_aupr = evaluate_with_tabpfn(
        policy_net, train_loader, val_loader, tabpfn_params, feat_indices
    )
    print(f"  [Pre-RL Baseline] AUC: {pre_rl_auc:.4f} | AUPR: {pre_rl_aupr:.4f}")

    # Optimizer with cosine schedule
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_aupr = pre_rl_aupr
    best_state = copy.deepcopy(policy_net.state_dict())
    patience = 15
    patience_counter = 0
    degradation_counter = 0

    print(f"  [Adaptive RL] Fixed-variance policy, AUPR-focused...")

    tabpfn_model = None

    for epoch in range(epochs):
        # Temperature for exploration (now directly controls std since variance is fixed)
        temperature = max(0.3, 0.7 - epoch / (epochs * 0.8))

        policy_net.train()

        # Sample with exploration
        X_train, y_train, log_probs_train = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )

        X_train_sel = X_train[:, feat_indices]

        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train_sel, y_train)

        # Per-sample precision-weighted rewards
        y_train_proba = tabpfn_model.predict_proba(X_train_sel)[:, 1]

        rewards = np.where(
            (y_train == 1) & (y_train_proba > 0.5),
            y_train_proba * 2.0,
            np.where(
                (y_train == 1),
                y_train_proba * 0.5,
                np.where(
                    y_train_proba < 0.5,
                    (1 - y_train_proba),
                    (1 - y_train_proba) * 0.5
                )
            )
        )

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Policy gradient — simpler now with fixed variance
        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.3)
        optimizer.step()
        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            val_auc, val_aupr = evaluate_with_tabpfn(
                policy_net, train_loader, val_loader, tabpfn_params, feat_indices
            )

            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | LR: {current_lr:.2e} | "
                  f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_state = copy.deepcopy(policy_net.state_dict())
                patience_counter = 0
                degradation_counter = 0
            else:
                patience_counter += 1

                if val_aupr < pre_rl_aupr - 0.005:
                    degradation_counter += 1
                    if degradation_counter >= 3:
                        print(f"    RL is degrading below pretrained baseline. Reverting.")
                        policy_net.load_state_dict(pre_rl_state)
                        return policy_net, feat_indices, True
                else:
                    degradation_counter = 0

                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    if best_val_aupr > pre_rl_aupr:
        policy_net.load_state_dict(best_state)
        print(f"  [Adaptive RL] Best AUPR: {best_val_aupr:.4f} (improved over pre-RL: {pre_rl_aupr:.4f})")
        return policy_net, feat_indices, False
    else:
        policy_net.load_state_dict(pre_rl_state)
        print(f"  [Adaptive RL] No improvement. Reverted to pretrained. (Pre-RL: {pre_rl_aupr:.4f})")
        return policy_net, feat_indices, True

# ==============================================================================
# 8. Main
# ==============================================================================

def main():
    print("="*80)
    print("RL POLICY V5 (Fixed Variance + Adaptive) TabPFN")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print("Encoding static features...")
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    metrics_rl = {k: [] for k in ['auc', 'auc_pr']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    reverted_folds = []

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, k=10, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p = train_p_obj.patientList
        val_p = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        latent_dim = 12
        policy_net = RNNPolicyNetwork(
            input_dim=len(temporal_feats),
            hidden_dim=64,
            latent_dim=latent_dim,
            time_dim=32
        ).to(DEVICE)

        tabpfn_params = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_estimators': 16,
        }

        # STEP 1: Enhanced Pretraining
        policy_net = pretrain_rnn_enhanced(policy_net, train_loader, val_loader, epochs=50)

        # Save pretrained state
        pretrained_state = copy.deepcopy(policy_net.state_dict())

        # STEP 2: Adaptive RL
        policy_net, feat_indices, was_reverted = train_policy_adaptive_rl(
            policy_net,
            train_loader,
            val_loader,
            tabpfn_params,
            epochs=40,
            update_tabpfn_every=5,
            top_k_features=80,
        )

        if was_reverted:
            reverted_folds.append(fold)

        # STEP 3: Final Evaluation with ensemble blend
        print("\n  [Final Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_rl, y_train_final, _ = extract_enriched_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_rl, y_test_final, _ = extract_enriched_features_and_logprobs(
                policy_net, test_loader, deterministic=True
            )

        if not was_reverted:
            pretrained_net = RNNPolicyNetwork(
                input_dim=len(temporal_feats),
                hidden_dim=64,
                latent_dim=latent_dim,
                time_dim=32
            ).to(DEVICE)
            pretrained_net.load_state_dict(pretrained_state)
            pretrained_net.eval()

            with torch.no_grad():
                X_train_pre, _, _ = extract_enriched_features_and_logprobs(
                    pretrained_net, train_loader, deterministic=True
                )
                X_test_pre, _, _ = extract_enriched_features_and_logprobs(
                    pretrained_net, test_loader, deterministic=True
                )

            X_train_final = 0.5 * X_train_pre + 0.5 * X_train_rl
            X_test_final = 0.5 * X_test_pre + 0.5 * X_test_rl
        else:
            X_train_final = X_train_rl
            X_test_final = X_test_rl

        X_train_sel = X_train_final[:, feat_indices]
        X_test_sel = X_test_final[:, feat_indices]

        final_tabpfn = TabPFNClassifier(**tabpfn_params)
        final_tabpfn.fit(X_train_sel, y_train_final)

        y_test_proba = final_tabpfn.predict_proba(X_test_sel)[:, 1]
        y_test_pred = (y_test_proba > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)

        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_rl['auc'].append(fold_auc)
        metrics_rl['auc_pr'].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_final, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  RL Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.set_title('ROC Curves (RL V5 - Fixed Variance)')
    ax1.legend(fontsize=8)

    ax2.bar(range(10), metrics_rl['auc_pr'], alpha=0.7, label='AUPR')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('AUPR')
    ax2.set_title('AUPR per Fold')
    ax2.legend()

    plt.tight_layout()
    os.makedirs('result', exist_ok=True)
    plt.savefig('result/tabpfn_rl_v5.png', dpi=150)
    plt.close()

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, rl_metrics):
        rl_mean, rl_std = np.mean(rl_metrics), np.std(rl_metrics)
        print(f"{name:15s} | RL: {rl_mean:.4f} ± {rl_std:.4f}")

    print_stat("AUC", metrics_rl['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'])

    if reverted_folds:
        print(f"\nFolds reverted to pretrained (RL hurt): {reverted_folds}")
    else:
        print(f"\nAll folds improved with RL.")

if __name__ == "__main__":
    main()