"""
Shared RL components for AKI prediction.

Provides common building blocks used across XGRL, CatBoostRL, and TabPFNRL:

  - FIXED_FEATURES         : static/demographic feature list
  - seed_everything        : reproducible seeding
  - SimpleStaticEncoder    : numeric encoding for categorical static features
  - HybridDataset          : PyTorch Dataset combining temporal + static data
  - hybrid_collate_fn      : zero-padding collation for variable-length sequences
  - RNNPolicyNetwork       : standard stochastic RNN policy (Gaussian, learned std)
  - extract_features_and_logprobs : [Static + Last_Values + Z] extraction
  - select_features_mi     : mutual-information feature selection
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import Dataset
from sklearn.feature_selection import mutual_info_classif

from TimeEmbedding import DEVICE, TimeEmbeddedRNNCell, extract_temporal_data


# Static/demographic features used as fixed context in RL models.
# (Distinct from TimeEmbedding.FIXED_FEATURES which includes "egfr" for
# temporal-feature filtering; here we encode these as static model input.)
FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable",
]


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Static Encoder
# ==============================================================================

class SimpleStaticEncoder:
    """Encode categorical static patient features into a numeric vector.

    Categorical values are mapped to consecutive integers (fitted on the
    training split).  Numeric values are passed through unchanged.
    """

    def __init__(self, features):
        self.features = features
        self.mappings = {f: {} for f in features}
        self.counts   = {f: 0  for f in features}

    def fit(self, patients):
        for p in patients:
            for f in self.features:
                val = p.measures.get(f, 0.0)
                if hasattr(val, "values") and len(val) > 0:
                    val = list(val.values())[0]
                elif hasattr(val, "values"):
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
            if hasattr(val, "values") and len(val) > 0:
                val = list(val.values())[0]
            elif hasattr(val, "values"):
                val = 0.0
            try:
                numeric_val = float(val)
            except ValueError:
                numeric_val = self.mappings[f].get(str(val), -1.0)
            vec.append(numeric_val)
        return vec


# ==============================================================================
# Dataset and Collation
# ==============================================================================

class HybridDataset(Dataset):
    """PyTorch Dataset combining temporal irregular time series with static features.

    Temporal values are z-score normalised using statistics computed from the
    training split.  Pass ``normalization_stats`` (from a training
    ``HybridDataset``) when constructing validation / test datasets to avoid
    data leakage.
    """

    def __init__(self, patients, feature_names, static_encoder, normalization_stats=None):
        self.data        = []
        self.labels      = []
        self.static_data = []
        self.feature_names = feature_names

        all_values   = []
        patient_list = patients.patientList if hasattr(patients, "patientList") else patients

        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue
            s_vec = static_encoder.transform(patient)
            self.static_data.append(torch.tensor(s_vec, dtype=torch.float32))
            self.data.append({"times": times, "values": values, "masks": masks})
            self.labels.append(1 if patient.akdPositive else 0)
            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0:
                        all_values.append(v)

        if normalization_stats is None:
            all_values  = np.array(all_values)
            self.mean   = float(np.mean(all_values)) if len(all_values) > 0 else 0.0
            self.std    = float(np.std(all_values))  if len(all_values) > 0 else 1.0
        else:
            self.mean = normalization_stats["mean"]
            self.std  = normalization_stats["std"]

        for i in range(len(self.data)):
            norm_values = [
                [(v - self.mean) / self.std if m > 0 else 0.0
                 for v, m in zip(v_vec, m_vec)]
                for v_vec, m_vec in zip(self.data[i]["values"], self.data[i]["masks"])
            ]
            self.data[i] = {
                "times":  torch.tensor(self.data[i]["times"],  dtype=torch.float32),
                "values": torch.tensor(norm_values,            dtype=torch.float32),
                "masks":  torch.tensor(self.data[i]["masks"],  dtype=torch.float32),
            }

    def get_normalization_stats(self):
        return {"mean": self.mean, "std": self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.static_data[idx]


def hybrid_collate_fn(batch):
    """Collate variable-length temporal sequences with zero-padding."""
    data_list, label_list, static_list = zip(*batch)
    lengths   = [len(d["times"]) for d in data_list]
    max_len   = max(lengths)
    feat_dim  = data_list[0]["values"].shape[-1]
    batch_size = len(data_list)

    padded_times  = torch.zeros(batch_size, max_len)
    padded_values = torch.zeros(batch_size, max_len, feat_dim)
    padded_masks  = torch.zeros(batch_size, max_len, feat_dim)

    for i, d in enumerate(data_list):
        l = lengths[i]
        padded_times[i,  :l]    = d["times"]
        padded_values[i, :l]    = d["values"]
        padded_masks[i,  :l]    = d["masks"]

    temporal_batch = {
        "times":   padded_times,
        "values":  padded_values,
        "masks":   padded_masks,
        "lengths": torch.tensor(lengths),
    }
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)


# ==============================================================================
# RNN Policy Network (standard Gaussian policy with learned std)
# ==============================================================================

class RNNPolicyNetwork(nn.Module):
    """Stochastic RNN policy: outputs a Gaussian distribution over latent Z.

    Training (stochastic): z ~ N(mean, exp(log_std))
    Evaluation (deterministic): z = mean

    The log-probability is evaluated at ``z.detach()`` so the REINFORCE
    gradient flows only through distribution parameters, not through z itself.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.rnn_cell   = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)
        self.fc_mean    = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd  = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, batch_data, deterministic=False):
        """
        Returns
        -------
        z        : sampled (or mean) latent vector  [batch, latent_dim]
        log_prob : per-sample log-probability       [batch]  (None if deterministic)
        mean     : distribution mean                [batch, latent_dim]
        """
        times   = batch_data["times"].to(DEVICE)
        values  = batch_data["values"].to(DEVICE)
        masks   = batch_data["masks"].to(DEVICE)
        lengths = batch_data["lengths"].to(DEVICE)

        h       = self.rnn_cell(times, values, masks, lengths)
        mean    = self.fc_mean(h)
        log_std = torch.clamp(self.fc_logstd(h), min=-20, max=2)
        std     = torch.exp(log_std)

        policy_dist = dist.Normal(mean, std)

        if deterministic:
            return mean, None, mean

        z        = policy_dist.rsample()
        log_prob = policy_dist.log_prob(z.detach()).sum(dim=-1)
        return z, log_prob, mean


# ==============================================================================
# Feature Extraction
# ==============================================================================

def extract_features_and_logprobs(policy_net, loader, deterministic=False):
    """Extract [Static + Last_Values + Z] feature vectors and log-probabilities.

    Parameters
    ----------
    policy_net    : RNNPolicyNetwork
    loader        : DataLoader (HybridDataset / hybrid_collate_fn)
    deterministic : bool — use distribution mean instead of sampling

    Returns
    -------
    features  : np.ndarray (n_samples, n_features)
    labels    : np.ndarray (n_samples,)
    log_probs : torch.Tensor or None
    """
    if deterministic:
        policy_net.eval()
    else:
        policy_net.train()

    all_features  = []
    all_labels    = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data in loader:
            z, log_prob, mean = policy_net(t_data, deterministic=deterministic)
            z_np = (mean if deterministic else z).detach().cpu().numpy()

            vals  = t_data["values"].cpu().numpy()
            masks = t_data["masks"].cpu().numpy()

            batch_last_vals = []
            for i in range(len(vals)):
                patient_last = []
                for f_idx in range(vals.shape[2]):
                    valid_idx = np.where(masks[i, :, f_idx] > 0)[0]
                    patient_last.append(
                        vals[i, valid_idx[-1], f_idx] if len(valid_idx) > 0 else 0.0
                    )
                batch_last_vals.append(patient_last)

            last_vals_arr = np.array(batch_last_vals)
            s_np          = s_data.numpy()

            all_features.append(np.hstack([s_np, last_vals_arr, z_np]))
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features  = np.vstack(all_features)
    labels    = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs


# ==============================================================================
# Feature Selection
# ==============================================================================

def select_features_mi(X_train, y_train, X_other, top_k=80):
    """Select top-k features by mutual information (computed on training data).

    Parameters
    ----------
    X_train : training feature matrix
    y_train : training labels
    X_other : validation or test matrix to apply the same selection
    top_k   : number of features to retain

    Returns
    -------
    X_train_sel, X_other_sel, top_indices
    """
    X_clean     = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    mi          = mutual_info_classif(X_clean, y_train, random_state=42, n_neighbors=5)
    top_indices = np.sort(np.argsort(mi)[-top_k:])
    return X_train[:, top_indices], X_other[:, top_indices], top_indices
