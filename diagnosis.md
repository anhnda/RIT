# Diagnosis: Why XGRLv2_fixed is Stuck

## Symptoms
```
Epoch 5-25: Reward=0.6150, AUC=0.5000, AUPR=0.4688 (NO CHANGE!)
```

## Root Causes

### 1. **Random Z is Useless Noise**
- Policy starts random → Z is pure noise (28 dimensions!)
- Features: [Static(23) + Last(25) + Mean(25) + Std(25) + **Z(28 random)**] = 126 dims
- XGBoost sees 28 noisy features out of 126 total
- **Result**: XGBoost ignores Z, just uses Static+Last+Mean+Std
- But if those alone worked, we wouldn't need RL!
- XGBoost probably just predicts majority class → AUC = 0.5

### 2. **Cold Start Problem**
- XGRL.py: 64 dims total, Z is 16 dims (25% noise)
- XGRLv2_fixed: 126 dims total, Z is 28 dims (22% noise)
- More features + more noise = harder for XGBoost to bootstrap

### 3. **Policy Gets No Learning Signal**
- If XGBoost predicts all same class → all samples get same reward
- No variance in rewards → policy gradient is zero!
- Policy doesn't learn → stays random → cycle continues

### 4. **Entropy Bonus Bug**
```python
entropy_bonus = 0.01 * log_probs_train.mean()  # log_probs are NEGATIVE!
total_loss = policy_loss - entropy_bonus       # WRONG SIGN
```
- log_prob of Gaussian is negative
- Subtracting negative = adding positive
- This REDUCES entropy instead of increasing it!

## Why XGRL.py Works

1. **Smaller dimensionality**: 64 dims vs 126 dims
2. **Less noise ratio**: 16/64 = 25% vs 28/126 = 22% (similar but simpler overall)
3. **Simpler features**: No Mean/Std, just Last
4. **Better chance**: XGBoost can learn from 64 dims even with some noise

## Solutions

### Option 1: Reduce Dimensionality (Quickest)
```python
# Use smaller latent_dim
latent_dim = 12  # Instead of 28
hidden_dim = 16  # Instead of 20

# Total: 23 + 25 + 25 + 25 + 12 = 110 dims (better than 126)
```

### Option 2: Initialize Policy Better
```python
# Add small supervised warm-up (5-10 epochs, not 50!)
def quick_warmup(policy_net, train_loader, epochs=5):
    head = nn.Linear(policy_net.latent_dim + len(FIXED_FEATURES), 1).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(head.parameters()), lr=0.01
    )

    for epoch in range(epochs):
        for t_data, labels, s_data in train_loader:
            z, _, _ = policy_net(t_data, deterministic=True)
            pred = torch.sigmoid(head(torch.cat([z, s_data.to(DEVICE)], dim=1)))
            loss = F.binary_cross_entropy(pred.squeeze(), labels.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Discard head, keep policy weights
    return policy_net
```

### Option 3: Fix Entropy Bonus (CRITICAL!)
```python
# WRONG:
entropy_bonus = 0.01 * log_probs_train.mean()
total_loss = policy_loss - entropy_bonus  # Reduces entropy!

# CORRECT:
# Method A: Use actual entropy
std = torch.exp(log_std)
entropy = 0.5 * torch.log(2 * np.pi * np.e * std**2).sum(dim=-1).mean()
total_loss = policy_loss - 0.01 * entropy  # Maximize entropy

# Method B: Just remove the minus sign
entropy_penalty = 0.01 * log_probs_train.mean()
total_loss = policy_loss + entropy_penalty  # Already negative, so adding encourages exploration
```

### Option 4: Bootstrap XGBoost First
```python
# Let XGBoost learn from deterministic features first
for epoch in range(10):  # Warm-up XGBoost
    X_train, y_train, _ = extract_enriched_features_and_logprobs(
        policy_net, train_loader, deterministic=True
    )
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)

# Now start RL with a working XGBoost
```

### Option 5: Curriculum Learning
```python
# Start with simple features, gradually add complexity
def get_features_curriculum(epoch, max_epochs):
    if epoch < max_epochs * 0.3:
        # Phase 1: Just [Static + Last + Z]
        return ['static', 'last', 'z']
    elif epoch < max_epochs * 0.6:
        # Phase 2: Add mean
        return ['static', 'last', 'mean', 'z']
    else:
        # Phase 3: Full features
        return ['static', 'last', 'mean', 'std', 'z']
```

## Recommended Fix (Combination)

1. **Fix entropy bonus** (MUST DO)
2. **Reduce latent_dim to 16** (quick win)
3. **Add 5-epoch warm-up** (helps bootstrap)
4. **Increase initial XGBoost training** (n_estimators=500 for first few epochs)

This should get the training moving!
