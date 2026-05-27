"""
================================================================================
REINFORCEMENT LEARNING FOR TEMPORAL MEDICAL DATA ENHANCEMENT
================================================================================

This document summarizes the RL-based approach used in:
- XGRL.py: RNN Policy Network → XGBoost Judge
- CatBoostRL.py: RNN Policy Network → CatBoost Judge
- TabPFNRL.py: RNN Policy Network → TabPFN Judge

================================================================================
CORE IDEA
================================================================================

Problem:
--------
Traditional methods use simple aggregations (last value, mean, etc.) from
temporal data. These fixed aggregations may not capture the most useful
patterns for prediction.

Solution:
---------
Use Reinforcement Learning to LEARN what temporal patterns are useful by
training a policy network that:
1. Processes temporal sequences with RNN
2. Generates learned representations (Z) that enhance traditional features
3. Gets feedback from a predictor (XGBoost/CatBoost/TabPFN) about usefulness
4. Updates to generate better representations

The predictor is non-differentiable (tree-based or pre-trained), so we use
policy gradient (REINFORCE) to learn.

================================================================================
OVERALL PIPELINE
================================================================================

┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Temporal Data                                                           │
│  [T×F matrix]                                                           │
│       ↓                                                                  │
│  ┌────────────────────────┐                                            │
│  │   RNN Policy Network   │  ← Learnable                               │
│  │  (Stochastic Policy)   │                                            │
│  └────────────────────────┘                                            │
│       ↓                                                                  │
│  Latent Z ~ π(z|temporal)  ← Sampled from Gaussian                     │
│  [B×D vector]                                                           │
│       ↓                                                                  │
│  ┌──────────────────────────────────────────────┐                      │
│  │  Feature Concatenation                        │                      │
│  │  [Static + Temporal Stats + Z]            │                      │
│  │                       │
│  └──────────────────────────────────────────────┘                      │
│       ↓                                                                  │
│  ┌────────────────────────┐                                            │
│  │  XGBoost/CatBoost/     │  ← Non-differentiable                     │
│  │  TabPFN Predictor      │                                            │
│  └────────────────────────┘                                            │
│       ↓                                                                  │
│  Predictions ŷ                                                          │
│       ↓                                                                  │
│  ┌────────────────────────┐                                            │
│  │  Compute Rewards       │                                            │
│  │  R(z) = f(ŷ, y_true)  │                                            │
│  └────────────────────────┘                                            │
│       ↓                                                                  │
│  ┌────────────────────────┐                                            │
│  │  Policy Gradient       │                                            │
│  │  Update Policy Network │                                            │
│  │  ∇θ J = E[∇θ log π(z|s) R(z)]                                     │
│  └────────────────────────┘                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PHASE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Temporal Data → RNN Policy (deterministic) → Z = μ(temporal)          │
│                                                                          │
│  [Static + Temporal Stats + Z] → Trained Predictor → ŷ              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

================================================================================
DETAILED FORMULATION
================================================================================

1. INPUT DATA
-------------

For each patient i:

Temporal Data:
  T_i = {(t₁, v₁, m₁), (t₂, v₂, m₂), ..., (tₙ, vₙ, mₙ)}

  where:
    t_k ∈ ℝ₊         : timestamp
    v_k ∈ ℝ^F        : feature values (F = 25 temporal features)
    m_k ∈ {0,1}^F    : mask (1 = observed, 0 = missing)

Static Features:
  S_i ∈ ℝ^23       : age, gender, race, comorbidities, etc.

Label:
  y_i ∈ {0, 1}     : 1 = AKD positive, 0 = negative


2. TEMPORAL STATISTICS EXTRACTION
----------------------------------

For each temporal feature f ∈ {1, ..., F}:

Last Value:
  last_f = v_k[f]  where k = argmax{j : m_j[f] = 1}

Mean Value:
  mean_f = (1/|K|) Σ_{k∈K} v_k[f]  where K = {k : m_k[f] = 1}

Variance:
  var_f = (1/|K|) Σ_{k∈K} (v_k[f] - mean_f)²

This gives us:
  Last ∈ ℝ^25    (one per feature)
  Mean ∈ ℝ^25
  Var  ∈ ℝ^25


3. RNN POLICY NETWORK
---------------------

Architecture:
  π_θ(z | T) = N(μ_θ(T), σ_θ(T))  ← Gaussian policy

Components:

a) Time-Embedded RNN Cell:

   h_t = RNN(h_{t-1}, v_t, Δt)

   where:
     h_t ∈ ℝ^128           : hidden state at time t
     Δt = t - t_{t-1}      : time gap

   Time embedding:
     τ_t = TimeEmbed(Δt)   : embeds time gap into vector

   Update rule:
     h_t = tanh(W_h h_{t-1} + W_v v_t + W_τ τ_t + b)

b) Policy Head:

   Final hidden state h_final ∈ ℝ^128 (after processing full sequence)

   μ_θ(T) = W_μ h_final + b_μ     ∈ ℝ^D    (mean)
   log σ_θ(T) = W_σ h_final + b_σ  ∈ ℝ^D    (log std)

   where D = latent_dim (typically 16)

c) Sampling (Training):

   z ~ N(μ_θ(T), σ_θ(T))    ← Reparameterization trick
   z = μ_θ(T) + σ_θ(T) ⊙ ε   where ε ~ N(0, I)

d) Deterministic (Inference):

   z = μ_θ(T)               ← Use mean directly


4. FEATURE CONCATENATION
-------------------------

Combined Features:
  X_i = [S_i || Last_i || Mean_i || Var_i || z_i]

  where || denotes concatenation

  Dimensions:
    S_i     : 23 (static features)
    Last_i  : 25 (last values)
    Mean_i  : 25 (means)
    Var_i   : 25 (variances)
    z_i     : 16 (learned representation)
    ────────────────────────────────────
    Total   : 114 dimensions


5. PREDICTOR (NON-DIFFERENTIABLE)
----------------------------------

XGBoost / CatBoost / TabPFN:

  ŷ_i = Predictor(X_i)     ∈ [0, 1]

Properties:
  - Tree-based (XGBoost, CatBoost) or pre-trained (TabPFN)
  - Cannot backpropagate through predictor
  - Trained on current features X with labels y


6. REWARD FUNCTION
------------------

Per-sample reward based on prediction quality:

Binary Reward:
  R_binary(z_i) = 𝟙{ŷ_i = y_i}   ∈ {0, 1}

  where 𝟙 is indicator function

Smooth Reward (probability-based):
  R_smooth(z_i) = {  ŷ_i       if y_i = 1
                  {  1 - ŷ_i   if y_i = 0

Combined Reward:
  R(z_i) = α R_binary(z_i) + (1-α) R_smooth(z_i)

  Typically α = 0.5

Normalized Reward (for stable training):
  R̂(z_i) = (R(z_i) - mean(R)) / std(R)


7. POLICY GRADIENT (REINFORCE)
-------------------------------

Objective:
  Maximize expected reward over policy

  J(θ) = 𝔼_{z~π_θ}[R(z)]
       = 𝔼_{T,z}[R(z)]

Gradient:
  ∇_θ J(θ) = 𝔼_{z~π_θ}[∇_θ log π_θ(z|T) R(z)]

Policy Gradient Theorem:
  The gradient can be estimated by sampling:

  ∇_θ J(θ) ≈ (1/B) Σ_{i=1}^B ∇_θ log π_θ(z_i|T_i) R̂(z_i)

Log Probability of Gaussian:
  log π_θ(z|T) = log N(z; μ_θ, σ_θ)
               = -1/2 Σ_d [(z_d - μ_d)/σ_d]² - log σ_d - log(2π)/2

Gradient w.r.t. θ:
  ∇_θ log π_θ(z|T) = ∇_θ μ_θ · (z - μ_θ)/σ_θ² +
                      ∇_θ log σ_θ · [(z - μ_θ)²/σ_θ² - 1]

Loss Function:
  L(θ) = -(1/B) Σ_{i=1}^B log π_θ(z_i|T_i) R̂(z_i)

Entropy Regularization (encourages exploration):
  L_total(θ) = L(θ) - β H(π_θ)

  where H(π_θ) = 𝔼[log π_θ(z)] is entropy

  For Gaussian:
    H = 1/2 log(2πe σ²)


8. TRAINING ALGORITHM
---------------------

Algorithm: REINFORCE with Baseline

Input: Dataset D = {(T_i, S_i, y_i)}_{i=1}^N

1. Initialize:
   - Policy network π_θ with random weights θ
   - Optimizer (Adam with lr = 0.0005)
   - Predictor (XGBoost/CatBoost/TabPFN)

2. For epoch = 1 to max_epochs:

   a) Sample Z from Policy (Stochastic):
      For each (T_i, S_i, y_i) in minibatch:
        - Process T_i through RNN → h_i
        - Compute μ_θ(T_i), σ_θ(T_i)
        - Sample z_i ~ N(μ_θ(T_i), σ_θ(T_i))
        - Compute log π_θ(z_i|T_i)
        - Extract stats: Last_i, Mean_i, Var_i
        - Concatenate: X_i = [S_i || Last_i || Mean_i || Var_i || z_i]

   b) Train/Update Predictor (every k epochs):
      - Fit Predictor on {(X_i, y_i)}

   c) Compute Rewards:
      - Get predictions: ŷ_i = Predictor(X_i)
      - Compute rewards: R(z_i) from ŷ_i and y_i
      - Normalize: R̂(z_i) = (R(z_i) - mean(R)) / std(R)

   d) Policy Gradient Update:
      - Compute loss:
        L = -(1/B) Σ_i log π_θ(z_i|T_i) R̂(z_i) - β H(π_θ)

      - Backpropagate through policy network:
        θ ← θ - α ∇_θ L

      - Clip gradients for stability

   e) Validation (Deterministic):
      - Use z = μ_θ(T) (no sampling)
      - Evaluate AUC, AUPR on validation set
      - Early stopping if no improvement


9. INFERENCE PROCEDURE
----------------------

Input: New patient with (T_new, S_new)

Step 1: Extract Temporal Statistics
  Last_new, Mean_new, Var_new = compute_stats(T_new)

Step 2: Generate Learned Representation (Deterministic)
  h_final = RNN(T_new)
  z_new = μ_θ(T_new)        ← Use mean, no sampling!

Step 3: Concatenate Features
  X_new = [S_new || Last_new || Mean_new || Var_new || z_new]

Step 4: Predict with Trained Predictor
  ŷ_new = Predictor(X_new)

Output: Prediction ŷ_new ∈ [0, 1]


================================================================================
WHY DOES THIS WORK?
================================================================================

1. NON-DIFFERENTIABLE PREDICTOR PROBLEM
   -------------------------------------

   Traditional Approach:
     If predictor was differentiable (e.g., neural network), we could:

     Loss = BCE(Predictor([Static + Last + Mean + Var + z]), y)
     ∇_θ Loss → backprop through predictor → update RNN

   But tree-based models (XGBoost, CatBoost) are non-differentiable!

   RL Solution:
     Treat predictor as environment
     - RNN policy produces "actions" (z)
     - Predictor gives "rewards" (prediction quality)
     - REINFORCE learns without backprop through predictor

     Key Insight:
       ∇_θ E[R(z)] = E[∇_θ log π(z) R(z)]

       We can compute ∇_θ log π(z) without knowing ∇_z R(z)!


2. WHAT DOES THE POLICY LEARN?
   ----------------------------

   The policy network learns to generate z that:

   a) Captures Temporal Patterns:
      - Trends (increasing/decreasing)
      - Volatility (stable vs fluctuating)
      - Critical events (sudden changes)
      - Time-dependent relationships

   b) Complements Handcrafted Features:
      - Last value: point estimate
      - Mean/Var: aggregate statistics
      - z: learned patterns that simple stats miss

   c) Optimizes Predictor Performance:
      - z is shaped to make predictor's job easier
      - Encodes information predictor finds useful
      - Different from supervised pre-training!


3. EXPLORATION VS EXPLOITATION
   ----------------------------

   During Training:
     - Sample z ~ N(μ, σ) → explores different representations
     - High σ → more exploration
     - Low σ → exploitation of known good representations
     - Temperature annealing: σ decreases over time

   During Inference:
     - Use z = μ (deterministic) → best representation found


4. COMPARISON TO SUPERVISED PRE-TRAINING
   --------------------------------------

   Supervised Pre-training:
     - Train RNN → neural classifier
     - Learn z good for neural networks
     - May not be good for tree-based models

   RL Training:
     - Train RNN with feedback from actual predictor (XGBoost/etc)
     - Learn z that specifically helps this predictor
     - Direct optimization of end task


================================================================================
KEY HYPERPARAMETERS
================================================================================

Policy Network:
  - hidden_dim: 12-20 (RNN hidden state)
  - latent_dim: 16-28 (dimension of z)
  - time_dim: 32 (time embedding dimension)

Training:
  - learning_rate: 0.0005
  - batch_size: 32
  - epochs: 100
  - update_predictor_every: 5 epochs
  - entropy_bonus: 0.01
  - gradient_clip: 1.0

Predictor (XGBoost):
  - n_estimators: 200
  - max_depth: 4
  - learning_rate: 0.05
  - scale_pos_weight: ratio (for imbalance)

Predictor (CatBoost):
  - iterations: 200
  - depth: 4
  - learning_rate: 0.05

Predictor (TabPFN):
  - device: 'cuda' or 'cpu'
  - N_ensemble_configurations: 32


================================================================================
VARIANTS
================================================================================

1. XGRL (XGRL.py)
   - Predictor: XGBoost
   - Features: [Static + Last + Z]  (64 dims)
   - No enriched stats (simpler)
   - Good baseline performance

2. CatBoostRL (CatBoostRL.py)
   - Predictor: CatBoost
   - Features: [Static + Last + Z]  (64 dims)
   - Better handling of categorical features
   - Slightly different hyperparameters

3. TabPFNRL (TabPFNRL.py)
   - Predictor: TabPFN (pre-trained transformer)
   - Features: [Static + Last + Mean + Std + Z]  (126 dims)
   - Enhanced features work well with TabPFN
   - Dataset size limit: 1024 samples
   - Requires supervised pretraining for initialization


================================================================================
ADVANTAGES
================================================================================

1. Learns task-specific representations
2. No gradient needed through predictor
3. Can use any non-differentiable predictor
4. Exploration helps find better features
5. End-to-end optimization of final task


================================================================================
CHALLENGES
================================================================================

1. Sample efficiency: RL needs many samples
2. Reward design: must balance binary/smooth rewards
3. Stability: policy gradient can be noisy
4. Cold start: random policy generates noise initially
5. Hyperparameter sensitivity: learning rate, entropy, etc.


================================================================================
FUTURE DIRECTIONS
================================================================================

1. Multi-objective rewards (AUC + calibration + fairness)
2. Hierarchical policies (different z for different time scales)
3. Meta-learning across patient populations
4. Interpretable z (what patterns did it learn?)
5. Online learning (update policy with new patients)


================================================================================
REFERENCES
================================================================================

[1] Williams, R. J. (1992). Simple statistical gradient-following
    algorithms for connectionist reinforcement learning.

[2] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.

[3] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning:
    An Introduction.

================================================================================
"""

# Example usage demonstration
if __name__ == "__main__":
    print(__doc__)

    print("\n" + "="*80)
    print("EXAMPLE PIPELINE")
    print("="*80)

    print("""
    # Pseudocode for complete pipeline:

    # 1. Load and prepare data
    patients = load_patients()
    train_patients, test_patients = split(patients)

    # 2. Initialize policy network
    policy_net = RNNPolicyNetwork(
        input_dim=25,      # temporal features
        hidden_dim=12,     # RNN size
        latent_dim=16,     # learned representation size
        time_dim=32        # time embedding size
    )

    # 3. RL training loop
    for epoch in range(100):
        # Sample from policy
        for batch in train_loader:
            temporal, labels, static = batch

            # Generate stochastic z
            z, log_prob = policy_net(temporal, deterministic=False)

            # Extract stats
            last = extract_last_values(temporal)
            mean = extract_mean_values(temporal)
            var = extract_variance(temporal)

            # Concatenate features
            X = concat([static, last, mean, var, z])

            # Train predictor
            if epoch % 5 == 0:
                predictor.fit(X, labels)

            # Get rewards
            y_pred = predictor.predict_proba(X)
            rewards = compute_rewards(y_pred, labels)

            # Policy gradient update
            loss = -(log_prob * rewards).mean()
            loss.backward()
            optimizer.step()

    # 4. Inference on test set
    policy_net.eval()
    for batch in test_loader:
        temporal, labels, static = batch

        # Deterministic z
        z = policy_net(temporal, deterministic=True)

        # Extract stats and concatenate
        X = concat([static, last, mean, var, z])

        # Predict
        y_pred = predictor.predict_proba(X)
    """)

    print("\n" + "="*80)
    print("See XGRL.py, CatBoostRL.py, or TabPFNRL.py for full implementations")
    print("="*80)
