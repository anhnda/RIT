# Temporal Data Encoding for Irregular Medical Time Series

## Overview

Medical time series data presents unique challenges compared to regular time series (e.g., stock prices, weather data):
- **Irregular sampling**: Measurements occur at non-uniform time intervals
- **Sparse observations**: Each timestamp may only contain a subset of all possible measurements
- **Missing data**: Many features are not measured at every time point
- **Variable sequence length**: Different patients have different numbers of observations

This document explains how we encode and process such irregular, sparse temporal data using a Time-Embedded RNN.

---

## 1. The Nature of Medical Time Series Data

### 1.1 Irregular Time Sampling

Unlike regular time series with fixed intervals (e.g., hourly readings), medical measurements occur at **arbitrary time points**:

```
Regular Time Series (e.g., hourly temperature):
    t: 0h    1h    2h    3h    4h    5h    6h
    v: 20.1  20.3  20.5  20.7  20.9  21.1  21.3
    ↓    ↓    ↓    ↓    ↓    ↓    ↓
    Fixed Δt = 1 hour

Medical Time Series (e.g., blood pressure):
    t: 0.5h  1.2h  4.7h  7.1h  9.8h  15.3h
    v: 120   118   115   125   122   119
    ↓    ↓    ↓    ↓    ↓    ↓
    Variable Δt: 0.7h, 3.5h, 2.4h, 2.7h, 5.5h, ...
```

**Why this happens:**
- Measurements taken when clinically needed (not on a schedule)
- Different urgency levels lead to different sampling rates
- Critical patients: more frequent measurements
- Stable patients: less frequent measurements

### 1.2 Sparse Observations per Timestamp

At each timestamp, only a **subset of features** is measured:

```
Example: Patient ICU stay with 25 potential measurements

Time t=1.5h:
  ✓ Heart Rate: 85
  ✓ Blood Pressure: 120/80
  ✗ Creatinine: not measured
  ✗ Glucose: not measured
  ✓ Temperature: 37.2°C
  ✗ Potassium: not measured
  ... (only 8 out of 25 features measured)

Time t=4.2h:
  ✓ Heart Rate: 88
  ✗ Blood Pressure: not measured
  ✓ Creatinine: 1.2 mg/dL
  ✓ Glucose: 110 mg/dL
  ✗ Temperature: not measured
  ✓ Potassium: 4.1 mEq/L
  ... (only 12 out of 25 features measured)
```

**Key Issue:**
Different features are observed at different times, creating a highly irregular multivariate time series.

### 1.3 Real Data Example

```python
Patient ICU Timeline (6 hours):
─────────────────────────────────────────────────────────────────
Time     | 0.5h  | 1.2h  | 4.7h  | 5.8h  |
─────────────────────────────────────────────────────────────────
HR       | 85    | 88    | -     | 92    |  (heart rate)
SBP      | 120   | -     | 115   | -     |  (systolic BP)
DBP      | 80    | -     | 75    | -     |  (diastolic BP)
Temp     | 37.2  | -     | -     | 37.5  |  (temperature)
SCr      | -     | 1.2   | -     | 1.4   |  (creatinine)
Glucose  | -     | 110   | 105   | -     |  (glucose)
K+       | -     | 4.1   | -     | 4.3   |  (potassium)
Na+      | 138   | -     | -     | 140   |  (sodium)
...      | ...   | ...   | ...   | ...   |
─────────────────────────────────────────────────────────────────

Legend: - = not measured
```

---

## 2. Data Representation

### 2.1 Raw Data Structure

For each patient `i`, temporal data is represented as a sequence of tuples:

```
T_i = [(t₁, v₁, m₁), (t₂, v₂, m₂), ..., (tₙ, vₙ, mₙ)]
```

Where:
- `t_k` ∈ ℝ₊: **Timestamp** (e.g., hours since ICU admission)
- `v_k` ∈ ℝ^F: **Feature values** (F = 25 features)
- `m_k` ∈ {0, 1}^F: **Mask** indicating which features are observed

**Example for one timestamp:**

```python
t_k = 4.7  # 4.7 hours after admission

v_k = [
    0.0,    # Feature 0: HR (not measured → filled with 0)
    115.0,  # Feature 1: SBP (measured)
    75.0,   # Feature 2: DBP (measured)
    0.0,    # Feature 3: Temp (not measured → filled with 0)
    0.0,    # Feature 4: SCr (not measured → filled with 0)
    105.0,  # Feature 5: Glucose (measured)
    0.0,    # Feature 6: K+ (not measured → filled with 0)
    ...
]

m_k = [
    0,  # Feature 0: not observed
    1,  # Feature 1: observed
    1,  # Feature 2: observed
    0,  # Feature 3: not observed
    0,  # Feature 4: not observed
    1,  # Feature 5: observed
    0,  # Feature 6: not observed
    ...
]
```

**Key Points:**
1. Missing features are filled with 0 in `v_k`
2. Mask `m_k` tells us which values are real (1) vs missing (0)
3. This allows fixed-size vectors while preserving missingness information

### 2.2 Sequence Representation

A complete patient sequence:

```python
# Patient with 4 observations
times = [0.5, 1.2, 4.7, 5.8]  # Irregular timestamps

values = [
    [85.0, 120.0, 80.0, 37.2, 0.0, 0.0, ...],     # t=0.5h
    [88.0, 0.0, 0.0, 0.0, 1.2, 110.0, ...],       # t=1.2h
    [0.0, 115.0, 75.0, 0.0, 0.0, 105.0, ...],     # t=4.7h
    [92.0, 0.0, 0.0, 37.5, 1.4, 0.0, ...]         # t=5.8h
]  # Shape: (T=4, F=25)

masks = [
    [1, 1, 1, 1, 0, 0, ...],  # t=0.5h: 4 features observed
    [1, 0, 0, 0, 1, 1, ...],  # t=1.2h: 3 features observed
    [0, 1, 1, 0, 0, 1, ...],  # t=4.7h: 3 features observed
    [1, 0, 0, 1, 1, 0, ...]   # t=5.8h: 3 features observed
]  # Shape: (T=4, F=25)
```

---

## 3. Challenges for Standard RNN

### 3.1 Problem 1: Irregular Time Intervals

Standard RNN assumes **uniform time steps**:

```python
# Standard RNN update
h_t = tanh(W_h h_{t-1} + W_x x_t + b)
```

**Problem:**
- This treats all time gaps equally
- Gap of 0.5 hours vs 5.0 hours are processed the same way
- Loses critical temporal information!

**Example:**
```
Case A: Rapid deterioration
    t=0h: SCr=1.0  →  t=1h: SCr=1.5  (Δt = 1h, ΔSCr = +0.5)

Case B: Slow progression
    t=0h: SCr=1.0  →  t=10h: SCr=1.5  (Δt = 10h, ΔSCr = +0.5)

Standard RNN: Treats both the same (same ΔSCr)
Reality: Case A is much more critical!
```

### 3.2 Problem 2: Missing Data

Standard RNN with 0-filling:

```python
# Missing values are set to 0
x_t = [85.0, 0.0, 0.0, 37.2, 0.0, ...]
```

**Problems:**
- 0 might be a valid measurement (e.g., no pain score = 0)
- Can't distinguish "not measured" from "measured as 0"
- Model learns wrong patterns from false 0s

### 3.3 Problem 3: Variable Sequence Length

Different patients have different numbers of observations:

```
Patient A: 15 observations  (15 timestamps)
Patient B: 43 observations  (43 timestamps)
Patient C: 8 observations   (8 timestamps)
```

**Batching challenge:**
- Need fixed-length tensors for batching
- Requires padding
- Must track actual sequence lengths

---

## 4. Time-Embedded RNN Solution

### 4.1 Architecture Overview

Our **Time-Embedded RNN Cell** addresses all three challenges:

```python
class TimeEmbeddedRNNCell:
    def __init__(self, input_dim, hidden_dim, time_dim):
        # Standard RNN parameters
        self.W_h = Linear(hidden_dim, hidden_dim)
        self.W_x = Linear(input_dim, hidden_dim)

        # Time embedding (NEW!)
        self.time_embed = TimeEmbedding(time_dim)
        self.W_time = Linear(time_dim, hidden_dim)

        # Mask handling (NEW!)
        self.W_mask = Linear(input_dim, hidden_dim)
```

### 4.2 Time Embedding

**Idea:** Encode time gaps as learnable representations

```python
def time_embedding(Δt, time_dim=32):
    """
    Embed time gap into vector using sinusoidal basis
    (similar to positional encoding in Transformers)
    """
    # Create frequency components
    freqs = 2π * (1 / 10000)^(2i/d) for i in range(time_dim/2)

    # Compute sinusoidal embedding
    τ = [sin(Δt × freq_0), cos(Δt × freq_0),
         sin(Δt × freq_1), cos(Δt × freq_1),
         ...]

    return τ ∈ ℝ^time_dim
```

**Properties:**
- Different frequencies capture short/long-term patterns
- Continuous representation of time
- Learnable via neural network layers

**Example:**
```python
Δt = 0.5h  → τ = [0.99, 0.14, 0.87, 0.49, ...]  (32-dim)
Δt = 5.0h  → τ = [0.28, -0.96, -0.54, 0.84, ...] (32-dim)

# Different time gaps have different embeddings
# Network learns to use these embeddings appropriately
```

### 4.3 Modified RNN Update Rule

**Complete update equation:**

```python
h_t = tanh(W_h h_{t-1} + W_x (x_t ⊙ m_t) + W_time τ(Δt) + b)
      ︸︷︷︸        ︸︷︷︸           ︸︷︷︸            ︸︷︷︸︸︷︷︸
      Previous    Observed       Time            Bias
      state       values         embedding
```

**Components:**

1. **h_{t-1}**: Previous hidden state (standard)

2. **x_t ⊙ m_t**: Element-wise multiplication with mask
   - Only uses observed values
   - Zeros out missing values explicitly
   ```python
   x_t = [85.0, 0.0, 0.0, 37.2, 0.0, ...]
   m_t = [1,    0,   0,   1,    0,   ...]
   x_t ⊙ m_t = [85.0, 0.0, 0.0, 37.2, 0.0, ...]  # Same, but now we know!
   ```

3. **τ(Δt)**: Time embedding
   - Encodes Δt = t - t_{t-1}
   - Tells RNN how much time passed
   ```python
   If t=4.7h and t_{t-1}=1.2h:
       Δt = 4.7 - 1.2 = 3.5 hours
       τ = time_embed(3.5) = [0.42, -0.91, ...]
   ```

### 4.4 Processing a Sequence

**Step-by-step example:**

```python
# Initialize
h_0 = zeros(hidden_dim)

# Process each timestamp
for t in [0.5, 1.2, 4.7, 5.8]:
    # Compute time gap
    Δt = t - t_prev

    # Get time embedding
    τ_t = time_embed(Δt)

    # Get values and mask
    x_t = values[t]  # [85.0, 120.0, 80.0, 37.2, 0.0, ...]
    m_t = masks[t]   # [1,    1,     1,    1,    0,   ...]

    # Masked input
    x_masked = x_t * m_t

    # RNN update
    h_t = tanh(
        W_h @ h_{t-1} +      # Previous state
        W_x @ x_masked +      # Masked observations
        W_time @ τ_t +        # Time encoding
        b
    )

    # Update for next step
    h_{t-1} = h_t
    t_prev = t

# Final hidden state h_final is used for prediction
```

**Visualization:**

```
Step 1: t=0.5h, Δt=0.5h
  [85, 120, 80, 37.2, 0, 0, ...] ⊙ [1, 1, 1, 1, 0, 0, ...]
  + time_embed(0.5)
  → h_1

Step 2: t=1.2h, Δt=0.7h
  [88, 0, 0, 0, 1.2, 110, ...] ⊙ [1, 0, 0, 0, 1, 1, ...]
  + time_embed(0.7)
  + h_1
  → h_2

Step 3: t=4.7h, Δt=3.5h  (LARGE GAP!)
  [0, 115, 75, 0, 0, 105, ...] ⊙ [0, 1, 1, 0, 0, 1, ...]
  + time_embed(3.5)  ← Network learns 3.5h is significant!
  + h_2
  → h_3

Step 4: t=5.8h, Δt=1.1h
  [92, 0, 0, 37.5, 1.4, 0, ...] ⊙ [1, 0, 0, 1, 1, 0, ...]
  + time_embed(1.1)
  + h_3
  → h_4 (final)
```

---

## 5. Handling Variable Sequence Lengths (Batching)

### 5.1 The Problem

```python
Batch of 3 patients:
  Patient 1: 15 observations  (T₁ = 15)
  Patient 2: 43 observations  (T₂ = 43)
  Patient 3: 8 observations   (T₃ = 8)

Need: Fixed-size tensor [Batch, MaxSeqLen, Features]
```

### 5.2 Padding Solution

**Pad shorter sequences to match longest:**

```python
max_len = max(T₁, T₂, T₃) = 43

# Padded tensors
times_padded = [
    [t₁, t₂, ..., t₁₅, 0, 0, 0, ..., 0],      # Patient 1: pad 28 zeros
    [t₁, t₂, ..., t₄₃],                        # Patient 2: no padding
    [t₁, t₂, ..., t₈, 0, 0, ..., 0]           # Patient 3: pad 35 zeros
]  # Shape: [3, 43]

values_padded = [
    [v₁, v₂, ..., v₁₅, [0...], [0...], ..., [0...]],  # Patient 1
    [v₁, v₂, ..., v₄₃],                                 # Patient 2
    [v₁, v₂, ..., v₈, [0...], [0...], ..., [0...]]    # Patient 3
]  # Shape: [3, 43, 25]

# Track actual lengths
lengths = [15, 43, 8]
```

### 5.3 Processing with Length Awareness

```python
def forward(batch_times, batch_values, batch_masks, lengths):
    batch_size = len(lengths)
    max_len = max(lengths)

    # Initialize hidden states
    h = zeros([batch_size, hidden_dim])

    for t in range(max_len):
        # Get current step data
        x_t = batch_values[:, t, :]      # [batch_size, features]
        m_t = batch_masks[:, t, :]       # [batch_size, features]
        time_t = batch_times[:, t]       # [batch_size]

        # Compute time gaps
        if t > 0:
            Δt = time_t - batch_times[:, t-1]
        else:
            Δt = time_t

        # Time embedding
        τ = time_embed(Δt)  # [batch_size, time_dim]

        # RNN update
        h_new = tanh(
            W_h @ h +
            W_x @ (x_t * m_t) +
            W_time @ τ +
            b
        )

        # Only update for patients with valid observations at this step
        # Create mask: which patients have t < their actual length?
        valid_mask = (t < lengths).unsqueeze(1)  # [batch_size, 1]

        # Update only valid positions
        h = valid_mask * h_new + (1 - valid_mask) * h

    return h  # Final hidden states [batch_size, hidden_dim]
```

**Key:** We only process steps up to each patient's actual length, ignoring padded values.

---

## 6. Normalization

### 6.1 Why Normalize?

Medical measurements have vastly different scales:

```
Heart Rate:       40-200 bpm
Blood Pressure:   60-200 mmHg
Creatinine:       0.5-10 mg/dL
Temperature:      35-42 °C
```

Without normalization, large values dominate learning.

### 6.2 Z-Score Normalization

```python
# Compute statistics from training data
all_observed_values = []
for patient in training_data:
    for t, v, m in patient.sequence:
        for i, (val, mask) in enumerate(zip(v, m)):
            if mask == 1:  # Only observed values
                all_observed_values.append(val)

mean = np.mean(all_observed_values)
std = np.std(all_observed_values)

# Normalize
v_normalized = (v - mean) / std  # For observed values only
```

**Applied only to observed values:**

```python
for each (v, m) in sequence:
    v_norm = []
    for val, mask in zip(v, m):
        if mask == 1:
            v_norm.append((val - mean) / std)  # Normalize
        else:
            v_norm.append(0.0)  # Keep as 0 (will be masked anyway)
```

---

## 7. Complete Processing Pipeline

### 7.1 Data Flow

```
Raw Medical Records
         ↓
[1] Extract temporal sequences
    - Identify timestamps
    - Extract feature values
    - Create masks
         ↓
[2] Normalize values
    - Compute global mean/std from training set
    - Apply to train/val/test
         ↓
[3] Create batches
    - Group patients
    - Pad to max length
    - Track actual lengths
         ↓
[4] Feed to RNN
    - Process each timestamp
    - Use time embeddings for Δt
    - Apply masks to handle missing data
    - Stop at actual length
         ↓
[5] Extract final hidden state
    - h_final = learned representation
    - Encodes temporal patterns
         ↓
[6] Use for prediction
    - Concatenate with other features
    - Feed to XGBoost/CatBoost/TabPFN
```

### 7.2 Code Example

```python
# Step 1: Extract temporal data
times, values, masks = extract_temporal_data(patient, feature_names)
# times: [0.5, 1.2, 4.7, 5.8]
# values: [[85, 120, ...], [88, 0, ...], ...]
# masks: [[1, 1, ...], [1, 0, ...], ...]

# Step 2: Normalize
values_norm = normalize(values, mean, std, masks)

# Step 3: Convert to tensors
times_tensor = torch.tensor(times)
values_tensor = torch.tensor(values_norm)
masks_tensor = torch.tensor(masks)

# Step 4: Process with RNN
h_final = rnn_cell(times_tensor, values_tensor, masks_tensor, length=4)
# h_final: [128] - learned temporal representation

# Step 5: Use in downstream task
z = policy_network(h_final)  # Further processing
features = [static, last_value, mean_value, var_value, z]
prediction = xgboost(features)
```

---

## 8. Advantages of This Approach

### 8.1 Handles Irregularity

✅ **Time-aware**: Δt explicitly modeled via time embeddings
✅ **No information loss**: Different sampling rates captured correctly

### 8.2 Handles Sparsity

✅ **Explicit masking**: Distinguishes missing from zero
✅ **Only uses observed data**: x_t ⊙ m_t ensures no false information

### 8.3 Variable Lengths

✅ **Flexible**: Processes sequences of any length
✅ **Efficient batching**: Padding with length tracking

### 8.4 Learns Patterns

✅ **Temporal dynamics**: RNN captures sequential dependencies
✅ **Time-sensitive**: Knows 1-hour vs 10-hour gaps are different
✅ **Feature interactions**: Learns how measurements relate over time

---

## 9. Comparison to Alternatives

### 9.1 Simple Aggregation (Baseline)

```python
# Just take last observed value per feature
features = [last_HR, last_BP, last_Temp, ...]
```

**Limitations:**
- ❌ Ignores temporal trends
- ❌ Ignores time of measurement
- ❌ No sequential patterns

### 9.2 Fixed-Interval Interpolation

```python
# Interpolate to hourly grid
times_grid = [0, 1, 2, 3, 4, 5, 6]
values_interpolated = interpolate(times, values, times_grid)
```

**Limitations:**
- ❌ Creates artificial data points
- ❌ Assumes smooth transitions (may not be true)
- ❌ Loses information about actual sampling

### 9.3 Standard RNN with 0-filling

```python
# Fill missing with 0, ignore time gaps
h_t = rnn(h_{t-1}, values_with_zeros)
```

**Limitations:**
- ❌ Can't distinguish missing from zero
- ❌ Treats all time gaps equally
- ❌ Loses critical temporal information

### 9.4 Time-Embedded RNN with Masking (Our Approach)

```python
# Explicit time + explicit masking
h_t = rnn(h_{t-1}, values ⊙ masks, time_embed(Δt))
```

**Advantages:**
- ✅ Preserves all information
- ✅ Handles irregularity naturally
- ✅ No artificial data
- ✅ Learns appropriate temporal dynamics

---

## 10. Summary

**Key Takeaways:**

1. **Medical time series are irregular and sparse**
   - Non-uniform sampling intervals
   - Many missing values per timestamp
   - Variable sequence lengths across patients

2. **Our solution: Time-Embedded RNN**
   - Time embeddings capture Δt information
   - Masks distinguish observed from missing
   - Length tracking handles variable sequences

3. **Processing pipeline:**
   ```
   Raw Data → Extract Sequences → Normalize → Batch with Padding
   → RNN with Time Embeddings → Final Representation → Prediction
   ```

4. **Why it works:**
   - Preserves all temporal information
   - No artificial interpolation
   - Learns from observed data only
   - Captures both short and long-term patterns

This approach enables deep learning on realistic medical data while respecting the irregular, sparse nature of clinical measurements.
