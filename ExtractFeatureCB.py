"""
FEATURE EXTRACTION FOR CATBOOSTRL USING SHAP

Two modes:
1. train_model: Train CatBoostRL, find best fold, save model and data
2. extract: Use SHAP to extract feature contributions for:
   - Canonical features (Static + Last values)
   - Temporal features (learned Z representation)
"""

import numpy as np
import sys
import copy
import torch
from torch.utils.data import DataLoader
from catboost import CatBoostClassifier
import pickle
import os
import argparse
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from scipy.stats import pearsonr

# SHAP library
import shap

PT = os.path.dirname(os.path.abspath(__file__))
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

# Import from CatBoostRL
from CatBoostRL import (
    SimpleStaticEncoder,
    HybridDataset,
    hybrid_collate_fn,
    RNNPolicyNetwork,
    extract_features_and_logprobs,
    train_policy_with_catboost_reward,
    FIXED_FEATURES,
    seed_everything,
)

xseed = 42
seed_everything(xseed)

# ==============================================================================
# MODE 1: TRAIN MODEL AND SAVE BEST FOLD
# ==============================================================================

def train_and_save_best_fold(output_dir="models/catboostrl"):
    """
    Train CatBoostRL on all folds, find the best fold, and save:
    - Policy network state
    - CatBoost model
    - Training and test data
    - Feature names
    """
    print("="*80)
    print("MODE 1: TRAINING AND SAVING BEST FOLD")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print("Encoding static features...")
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    best_fold_idx = -1
    best_fold_aupr = 0
    best_fold_data = None

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p = train_p_obj.patientList
        val_p = val_p_obj.patientList
        test_p_list = test_p.patientList

        # Create Datasets
        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # Initialize Policy Network
        latent_dim = 16
        policy_net = RNNPolicyNetwork(
            input_dim=len(temporal_feats),
            hidden_dim=12,
            latent_dim=latent_dim,
            time_dim=32
        ).to(DEVICE)

        # CatBoost Parameters
        ratio = np.sum([1 for _, l, _ in train_ds if l == 0]) / max(1, np.sum([1 for _, l, _ in train_ds if l == 1]))

        catboost_params = {
            'iterations': 200,
            'depth': 4,
            'learning_rate': 0.05,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'scale_pos_weight': ratio,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            'task_type': 'CPU'
        }

        # Train Policy with RL
        policy_net = train_policy_with_catboost_reward(
            policy_net,
            train_loader,
            val_loader,
            catboost_params,
            epochs=100,
            update_catboost_every=5
        )

        # Final Evaluation on Test Set
        print("\n  [Final Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_features_and_logprobs(
                policy_net, test_loader, deterministic=True
            )

        # Train final CatBoost
        final_catboost = CatBoostClassifier(**catboost_params)
        final_catboost.fit(X_train_final, y_train_final, verbose=False)

        y_test_proba = final_catboost.predict_proba(X_test_final)[:, 1]

        # Compute Metrics
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)
        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        print(f"  Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

        # Check if this is the best fold
        if fold_aupr > best_fold_aupr:
            best_fold_aupr = fold_aupr
            best_fold_idx = fold

            # Store data for best fold
            best_fold_data = {
                'policy_net_state': copy.deepcopy(policy_net.state_dict()),
                'catboost_model': final_catboost,
                'X_train': X_train_final,
                'y_train': y_train_final,
                'X_test': X_test_final,
                'y_test': y_test_final,
                'encoder': encoder,
                'temporal_feats': temporal_feats,
                'stats': stats,
                'latent_dim': latent_dim,
                'catboost_params': catboost_params,
                'fold_idx': fold,
                'fold_auc': fold_auc,
                'fold_aupr': fold_aupr
            }

    # Save best fold
    if best_fold_data is not None:
        print(f"\n{'='*80}")
        print(f"BEST FOLD: {best_fold_idx} (AUPR: {best_fold_aupr:.4f})")
        print(f"Saving to {output_dir}/")
        print('='*80)

        # Save policy network
        torch.save(best_fold_data['policy_net_state'],
                   os.path.join(output_dir, 'policy_net.pth'))

        # Save CatBoost model
        best_fold_data['catboost_model'].save_model(
            os.path.join(output_dir, 'catboost_model.cbm')
        )

        # Save data and metadata
        with open(os.path.join(output_dir, 'fold_data.pkl'), 'wb') as f:
            pickle.dump({
                'X_train': best_fold_data['X_train'],
                'y_train': best_fold_data['y_train'],
                'X_test': best_fold_data['X_test'],
                'y_test': best_fold_data['y_test'],
                'encoder': best_fold_data['encoder'],
                'temporal_feats': best_fold_data['temporal_feats'],
                'stats': best_fold_data['stats'],
                'latent_dim': best_fold_data['latent_dim'],
                'catboost_params': best_fold_data['catboost_params'],
                'fold_idx': best_fold_data['fold_idx'],
                'fold_auc': best_fold_data['fold_auc'],
                'fold_aupr': best_fold_data['fold_aupr']
            }, f)

        print(f"Saved successfully!")
        print(f"  - Policy network: policy_net.pth")
        print(f"  - CatBoost model: catboost_model.cbm")
        print(f"  - Data & metadata: fold_data.pkl")
    else:
        print("No valid fold found!")

# ==============================================================================
# MODE 2: EXTRACT FEATURES USING SHAP
# ==============================================================================

def extract_features_with_shap(model_dir="models/catboostrl", top_k=20):
    """
    Load saved model and use SHAP to extract feature contributions.

    Two types of analysis:
    1. Canonical features: Static + Last values
    2. Temporal features: Learned Z representation
    """
    print("="*80)
    print("MODE 2: EXTRACTING FEATURES WITH SHAP")
    print("="*80)

    # Load saved data
    print(f"\nLoading model from {model_dir}/")

    with open(os.path.join(model_dir, 'fold_data.pkl'), 'rb') as f:
        data = pickle.load(f)

    catboost_model = CatBoostClassifier()
    catboost_model.load_model(os.path.join(model_dir, 'catboost_model.cbm'))

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    temporal_feats = data['temporal_feats']

    print(f"Loaded fold {data['fold_idx']} (AUC: {data['fold_auc']:.4f}, AUPR: {data['fold_aupr']:.4f})")
    print(f"Feature dimensions:")
    print(f"  - Static features: {len(FIXED_FEATURES)}")
    print(f"  - Last values (temporal): {len(temporal_feats)}")
    print(f"  - Latent Z dimension: {data['latent_dim']}")
    print(f"  - Total features: {X_train.shape[1]}")

    # Create feature names
    n_static = len(FIXED_FEATURES)
    n_temporal = len(temporal_feats)
    n_latent = data['latent_dim']

    feature_names = (
        [f"static_{f}" for f in FIXED_FEATURES] +
        [f"last_{f}" for f in temporal_feats] +
        [f"latent_z{i}" for i in range(n_latent)]
    )

    # Indices for different feature types
    static_indices = list(range(n_static))
    temporal_last_indices = list(range(n_static, n_static + n_temporal))
    latent_indices = list(range(n_static + n_temporal, n_static + n_temporal + n_latent))

    canonical_indices = static_indices + temporal_last_indices  # Static + Last values

    print("\n" + "="*80)
    print("COMPUTING SHAP VALUES")
    print("="*80)

    # Use TreeExplainer for CatBoost (fast and exact)
    explainer = shap.TreeExplainer(catboost_model)

    # Compute SHAP values for test set
    print("\nComputing SHAP values for test set...")
    shap_values = explainer.shap_values(X_test)

    # Handle both binary and multi-class cases
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class

    print(f"SHAP values shape: {shap_values.shape}")

    # ==============================================================================
    # ANALYSIS 1: CANONICAL FEATURES (Static + Last Values)
    # ==============================================================================
    print("\n" + "="*80)
    print("CANONICAL FEATURES (Static + Last Values)")
    print("="*80)

    canonical_shap = shap_values[:, canonical_indices]
    canonical_names = [feature_names[i] for i in canonical_indices]

    # Compute mean absolute SHAP values
    mean_abs_shap_canonical = np.abs(canonical_shap).mean(axis=0)

    # Sort by importance
    sorted_idx_canonical = np.argsort(mean_abs_shap_canonical)[::-1]

    print(f"\nTop {top_k} Canonical Features by SHAP Importance:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature Name':<40} {'Mean |SHAP|':<15} {'Type':<10}")
    print("-" * 80)

    for rank, idx in enumerate(sorted_idx_canonical[:top_k]):
        feat_name = canonical_names[idx]
        shap_val = mean_abs_shap_canonical[idx]

        # Determine type
        if feat_name.startswith('static_'):
            feat_type = 'Static'
        else:
            feat_type = 'Last'

        print(f"{rank+1:<6} {feat_name:<40} {shap_val:<15.6f} {feat_type:<10}")

    # ==============================================================================
    # ANALYSIS 2: TEMPORAL FEATURES (Latent Z)
    # ==============================================================================
    print("\n" + "="*80)
    print("TEMPORAL FEATURES (Learned Latent Z)")
    print("="*80)

    latent_shap = shap_values[:, latent_indices]
    latent_names = [feature_names[i] for i in latent_indices]

    # Compute mean absolute SHAP values
    mean_abs_shap_latent = np.abs(latent_shap).mean(axis=0)

    # Sort by importance
    sorted_idx_latent = np.argsort(mean_abs_shap_latent)[::-1]

    print(f"\nTop {min(top_k, len(latent_names))} Latent Features by SHAP Importance:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature Name':<40} {'Mean |SHAP|':<15}")
    print("-" * 80)

    for rank, idx in enumerate(sorted_idx_latent[:min(top_k, len(latent_names))]):
        feat_name = latent_names[idx]
        shap_val = mean_abs_shap_latent[idx]
        print(f"{rank+1:<6} {feat_name:<40} {shap_val:<15.6f}")

    # ==============================================================================
    # COMBINED ANALYSIS: Overall Top Features
    # ==============================================================================
    print("\n" + "="*80)
    print("OVERALL TOP FEATURES (All Types)")
    print("="*80)

    mean_abs_shap_all = np.abs(shap_values).mean(axis=0)
    sorted_idx_all = np.argsort(mean_abs_shap_all)[::-1]

    print(f"\nTop {top_k} Features Overall by SHAP Importance:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature Name':<40} {'Mean |SHAP|':<15} {'Type':<10}")
    print("-" * 80)

    for rank, idx in enumerate(sorted_idx_all[:top_k]):
        feat_name = feature_names[idx]
        shap_val = mean_abs_shap_all[idx]

        # Determine type
        if idx in static_indices:
            feat_type = 'Static'
        elif idx in temporal_last_indices:
            feat_type = 'Last'
        else:
            feat_type = 'Latent'

        print(f"{rank+1:<6} {feat_name:<40} {shap_val:<15.6f} {feat_type:<10}")

    # ==============================================================================
    # SAVE SHAP RESULTS
    # ==============================================================================
    output_file = os.path.join(model_dir, 'shap_results.pkl')
    print(f"\n{'='*80}")
    print(f"Saving SHAP results to {output_file}")
    print('='*80)

    with open(output_file, 'wb') as f:
        pickle.dump({
            'shap_values': shap_values,
            'feature_names': feature_names,
            'canonical_indices': canonical_indices,
            'latent_indices': latent_indices,
            'mean_abs_shap_all': mean_abs_shap_all,
            'mean_abs_shap_canonical': mean_abs_shap_canonical,
            'mean_abs_shap_latent': mean_abs_shap_latent,
            'X_test': X_test,
            'y_test': y_test
        }, f)

    print("SHAP results saved successfully!")

    # ==============================================================================
    # ADDITIONAL ANALYSIS: Contribution by Type
    # ==============================================================================
    print("\n" + "="*80)
    print("FEATURE TYPE CONTRIBUTION SUMMARY")
    print("="*80)

    static_contrib = mean_abs_shap_all[static_indices].sum()
    last_contrib = mean_abs_shap_all[temporal_last_indices].sum()
    latent_contrib = mean_abs_shap_all[latent_indices].sum()
    total_contrib = static_contrib + last_contrib + latent_contrib

    print(f"\nTotal contribution by feature type:")
    print(f"  Static features:        {static_contrib:.4f} ({100*static_contrib/total_contrib:.2f}%)")
    print(f"  Last values (temporal): {last_contrib:.4f} ({100*last_contrib/total_contrib:.2f}%)")
    print(f"  Latent Z (learned):     {latent_contrib:.4f} ({100*latent_contrib/total_contrib:.2f}%)")
    print(f"  Total:                  {total_contrib:.4f}")

    return shap_values, feature_names

# ==============================================================================
# MODE 3: INTERPRET LATENT FACTORS (Trace back to RNN inputs)
# ==============================================================================

def interpret_latent_factors(model_dir="models/catboostrl", top_k=10):
    """
    Trace latent factors back to RNN input level.

    For each latent dimension, identify which temporal input features
    contribute most to its activation.

    Methods:
    1. Correlation analysis: Correlate temporal inputs with latent outputs
    2. Gradient-based attribution: Use gradients to see feature importance
    3. Temporal pattern analysis: Analyze which time points matter most
    """
    print("="*80)
    print("MODE 3: INTERPRETING LATENT FACTORS")
    print("="*80)

    # Load saved data
    print(f"\nLoading model and data from {model_dir}/")

    with open(os.path.join(model_dir, 'fold_data.pkl'), 'rb') as f:
        data = pickle.load(f)

    temporal_feats = data['temporal_feats']
    latent_dim = data['latent_dim']

    # Load policy network
    policy_net = RNNPolicyNetwork(
        input_dim=len(temporal_feats),
        hidden_dim=12,
        latent_dim=latent_dim,
        time_dim=32
    ).to(DEVICE)

    policy_net.load_state_dict(
        torch.load(os.path.join(model_dir, 'policy_net.pth'),
                   map_location=DEVICE)
    )
    policy_net.eval()

    print(f"Loaded policy network with {latent_dim} latent dimensions")
    print(f"Temporal features: {len(temporal_feats)}")

    # Load test data to reconstruct temporal inputs
    patients = load_and_prepare_patients()

    # Need to reconstruct the test set from the same fold
    fold_idx = data['fold_idx']
    encoder = data['encoder']
    stats = data['stats']

    # Iterate to the correct fold
    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        if fold == fold_idx:
            test_p_list = test_p.patientList
            break

    # Create test dataset
    test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

    print(f"\nReconstructed test set with {len(test_ds)} samples")

    # ==============================================================================
    # Extract temporal inputs and latent outputs
    # ==============================================================================
    print("\n" + "="*80)
    print("EXTRACTING TEMPORAL INPUTS AND LATENT OUTPUTS")
    print("="*80)

    all_temporal_inputs = []  # Raw temporal sequences (keep as list due to variable lengths)
    all_temporal_masks = []   # Masks for valid values
    all_latent_outputs = []   # Latent Z vectors
    all_labels = []

    with torch.no_grad():
        for t_data, labels, s_data in test_loader:
            # Get latent outputs (deterministic)
            z, _, mean = policy_net(t_data, deterministic=True)

            # Store latent outputs
            all_latent_outputs.append(mean.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Store temporal inputs (keep in 3D format)
            vals = t_data['values'].cpu().numpy()
            masks = t_data['masks'].cpu().numpy()

            all_temporal_inputs.append(vals)
            all_temporal_masks.append(masks)

    # Concatenate latent outputs
    latent_outputs = np.vstack(all_latent_outputs)
    labels = np.array(all_labels)

    print(f"Number of samples: {len(labels)}")
    print(f"Latent outputs shape: {latent_outputs.shape}")
    print(f"Number of temporal batches: {len(all_temporal_inputs)}")

    # ==============================================================================
    # ANALYSIS 1: Correlation between temporal features and latent dimensions
    # ==============================================================================
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: Temporal Features → Latent Dimensions")
    print("="*80)

    # For each temporal feature, compute statistics across time
    # (mean, std, min, max, last, count of valid measurements)

    n_samples = len(labels)
    n_temporal_feats = len(temporal_feats)

    # Aggregate temporal features per sample
    print(f"\nComputing temporal aggregates for {n_samples} samples...")
    temporal_aggregates = np.zeros((n_samples, n_temporal_feats * 4))  # mean, std, min, max

    sample_idx = 0
    for batch_vals, batch_masks in zip(all_temporal_inputs, all_temporal_masks):
        for i in range(len(batch_vals)):
            vals = batch_vals[i]  # shape: (seq_len, n_feats)
            masks = batch_masks[i]

            for f_idx in range(n_temporal_feats):
                f_vals = vals[:, f_idx]
                f_mask = masks[:, f_idx]
                valid_vals = f_vals[f_mask > 0]

                if len(valid_vals) > 0:
                    temporal_aggregates[sample_idx, f_idx * 4 + 0] = np.mean(valid_vals)
                    temporal_aggregates[sample_idx, f_idx * 4 + 1] = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
                    temporal_aggregates[sample_idx, f_idx * 4 + 2] = np.min(valid_vals)
                    temporal_aggregates[sample_idx, f_idx * 4 + 3] = np.max(valid_vals)

            sample_idx += 1

    print(f"Temporal aggregates shape: {temporal_aggregates.shape}")

    # Compute correlation between temporal aggregates and latent dimensions
    # For each latent dimension, find most correlated temporal features
    latent_to_temporal = {}

    print("\nComputing correlations between temporal aggregates and latent dimensions...")
    for z_idx in range(latent_dim):
        z_values = latent_outputs[:, z_idx]
        correlations = []

        for f_idx in range(n_temporal_feats):
            # Check correlation with different aggregates
            try:
                # Only compute correlation if there's variation in both variables
                agg_mean = temporal_aggregates[:, f_idx * 4 + 0]
                agg_std = temporal_aggregates[:, f_idx * 4 + 1]
                agg_min = temporal_aggregates[:, f_idx * 4 + 2]
                agg_max = temporal_aggregates[:, f_idx * 4 + 3]

                corr_mean = abs(pearsonr(agg_mean, z_values)[0]) if np.std(agg_mean) > 1e-6 else 0.0
                corr_std = abs(pearsonr(agg_std, z_values)[0]) if np.std(agg_std) > 1e-6 else 0.0
                corr_min = abs(pearsonr(agg_min, z_values)[0]) if np.std(agg_min) > 1e-6 else 0.0
                corr_max = abs(pearsonr(agg_max, z_values)[0]) if np.std(agg_max) > 1e-6 else 0.0

                # Handle NaN values
                corr_mean = 0.0 if np.isnan(corr_mean) else corr_mean
                corr_std = 0.0 if np.isnan(corr_std) else corr_std
                corr_min = 0.0 if np.isnan(corr_min) else corr_min
                corr_max = 0.0 if np.isnan(corr_max) else corr_max

                max_corr = max(corr_mean, corr_std, corr_min, corr_max)
                agg_type = ['mean', 'std', 'min', 'max'][np.argmax([corr_mean, corr_std, corr_min, corr_max])]

                correlations.append((temporal_feats[f_idx], max_corr, agg_type))
            except Exception as e:
                # Skip features that cause errors
                correlations.append((temporal_feats[f_idx], 0.0, 'n/a'))

        # Sort by correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        latent_to_temporal[z_idx] = correlations

    # Display results for important latent dimensions
    # Load SHAP results to identify important latents
    shap_file = os.path.join(model_dir, 'shap_results.pkl')
    if os.path.exists(shap_file):
        with open(shap_file, 'rb') as f:
            shap_data = pickle.load(f)

        mean_abs_shap_latent = shap_data['mean_abs_shap_latent']
        sorted_latent_idx = np.argsort(mean_abs_shap_latent)[::-1]

        print("\nTop latent dimensions by SHAP importance and their temporal feature correlations:")
        print("="*80)

        for rank, z_idx in enumerate(sorted_latent_idx[:min(10, latent_dim)]):
            shap_importance = mean_abs_shap_latent[z_idx]
            print(f"\nLatent Z{z_idx} (SHAP importance: {shap_importance:.6f})")
            print("-" * 80)
            print(f"{'Rank':<6} {'Temporal Feature':<30} {'Correlation':<15} {'Aggregate':<10}")
            print("-" * 80)

            for i, (feat_name, corr, agg_type) in enumerate(latent_to_temporal[z_idx][:top_k]):
                print(f"{i+1:<6} {feat_name:<30} {corr:<15.6f} {agg_type:<10}")
    else:
        print("\nNo SHAP results found. Showing all latent dimensions:")
        for z_idx in range(min(5, latent_dim)):
            print(f"\nLatent Z{z_idx}")
            print("-" * 80)
            print(f"{'Rank':<6} {'Temporal Feature':<30} {'Correlation':<15} {'Aggregate':<10}")
            print("-" * 80)

            for i, (feat_name, corr, agg_type) in enumerate(latent_to_temporal[z_idx][:top_k]):
                print(f"{i+1:<6} {feat_name:<30} {corr:<15.6f} {agg_type:<10}")

    # ==============================================================================
    # ANALYSIS 2: Gradient-based attribution
    # ==============================================================================
    print("\n" + "="*80)
    print("GRADIENT-BASED ATTRIBUTION: Which temporal features influence each latent?")
    print("="*80)

    # Enable gradients
    policy_net.train()

    gradient_attributions = {z_idx: np.zeros(n_temporal_feats) for z_idx in range(latent_dim)}

    print("\nComputing gradients...")
    for t_data, labels, s_data in test_loader:
        t_data_grad = {
            'times': t_data['times'].to(DEVICE),
            'values': t_data['values'].to(DEVICE).requires_grad_(True),
            'masks': t_data['masks'].to(DEVICE),
            'lengths': t_data['lengths'].to(DEVICE)
        }

        # Forward pass
        z, _, mean = policy_net(t_data_grad, deterministic=True)

        # For each latent dimension, compute gradient
        for z_idx in range(latent_dim):
            if z_idx < mean.shape[1]:
                # Compute gradient of z_idx w.r.t. input values
                mean[:, z_idx].sum().backward(retain_graph=True)

                # Get gradients
                grads = t_data_grad['values'].grad
                masks = t_data_grad['masks']

                # Compute attribution per feature (averaged over time and samples)
                for f_idx in range(n_temporal_feats):
                    feat_grads = grads[:, :, f_idx]
                    feat_masks = masks[:, :, f_idx]

                    # Average absolute gradient for valid positions
                    valid_grads = torch.abs(feat_grads * feat_masks).sum().item()
                    valid_count = feat_masks.sum().item()

                    if valid_count > 0:
                        gradient_attributions[z_idx][f_idx] += valid_grads / valid_count

                # Zero gradients for next iteration
                t_data_grad['values'].grad.zero_()

    # Normalize attributions
    for z_idx in range(latent_dim):
        gradient_attributions[z_idx] /= len(test_loader)

    # Display gradient attributions for important latents
    if os.path.exists(shap_file):
        print("\nTop latent dimensions and their gradient-based temporal feature attributions:")
        print("="*80)

        for rank, z_idx in enumerate(sorted_latent_idx[:min(10, latent_dim)]):
            shap_importance = mean_abs_shap_latent[z_idx]
            print(f"\nLatent Z{z_idx} (SHAP importance: {shap_importance:.6f})")
            print("-" * 80)
            print(f"{'Rank':<6} {'Temporal Feature':<30} {'Gradient Attribution':<20}")
            print("-" * 80)

            # Sort features by gradient attribution
            feat_grad_pairs = [(temporal_feats[i], gradient_attributions[z_idx][i])
                               for i in range(n_temporal_feats)]
            feat_grad_pairs.sort(key=lambda x: x[1], reverse=True)

            for i, (feat_name, grad_attr) in enumerate(feat_grad_pairs[:top_k]):
                print(f"{i+1:<6} {feat_name:<30} {grad_attr:<20.6f}")

    # ==============================================================================
    # Save interpretation results
    # ==============================================================================
    output_file = os.path.join(model_dir, 'latent_interpretation.pkl')
    print(f"\n{'='*80}")
    print(f"Saving latent interpretation to {output_file}")
    print('='*80)

    with open(output_file, 'wb') as f:
        pickle.dump({
            'latent_to_temporal_corr': latent_to_temporal,
            'gradient_attributions': gradient_attributions,
            'temporal_features': temporal_feats,
            'latent_dim': latent_dim
        }, f)

    print("Latent interpretation saved successfully!")

    policy_net.eval()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction for CatBoostRL')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train_model', 'extract', 'interpret'],
                        help='Mode: train_model, extract, or interpret')
    parser.add_argument('--output_dir', type=str, default='models/catboostrl',
                        help='Directory to save/load models')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top features to display')

    args = parser.parse_args()

    if args.mode == 'train_model':
        train_and_save_best_fold(output_dir=args.output_dir)
    elif args.mode == 'extract':
        extract_features_with_shap(model_dir=args.output_dir, top_k=args.top_k)
    elif args.mode == 'interpret':
        interpret_latent_factors(model_dir=args.output_dir, top_k=args.top_k)

if __name__ == "__main__":
    main()
