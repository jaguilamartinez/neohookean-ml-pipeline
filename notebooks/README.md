# Notebooks Guide

This directory contains Jupyter notebooks for training and analyzing the Neo-Hookean ML models.

## Available Notebooks

### 1. `baseline_analysis.ipynb`
**Baseline Black-Box Neural Network**

Trains and evaluates a standard feedforward neural network (11,843 parameters):
- 3-layer architecture (128 → 64 → 32 neurons)
- Uses scaled data (StandardScaler)
- Achieves R² ≈ 0.9932

**Sections:**
1. Load Data
2. Prepare Data (with scaling)
3. Build Baseline Model
4. Train Model
5. Training History
6. Evaluate Model
7. Predictions
8. Residual Analysis
9. Component Metrics
10. Save Model

### 2. `pann_analysis.ipynb`
**Physics-Augmented Neural Network (PANN)**

Trains and evaluates the physics-informed PANN model:
- Adaptive parameter count (default: n=128, layers=2 → 17,282 params)
- Uses **unscaled physical values** (required for physics computations)
- Achieves R² ≈ 0.9283 (with n=128, layers=2)

**Key Features:**
- Input: Green-Lagrange strain → Deformation gradient F
- Energy-based: Computes P = ∂Ψ/∂F via autodiff
- Output: PK2 stress (converted from P)
- Physics constraints: polyconvexity, zero stress at identity

**Sections:**
1. Load Data
2. Prepare Data (**no scaling** - uses physical values)
3. Build PANN Model (configurable: `n`, `layer_num`)
4. Train PANN Model
5. Training History
6. Evaluate PANN Model
7. PANN Predictions
8. Residual Analysis
9. Component Metrics
10. Save Model

### 3. `comparison.ipynb`
**Baseline vs PANN Comparison**

Side-by-side comparison of both models:
- Performance metrics
- Prediction quality
- Residual distributions
- Error distributions
- Statistical summaries

**Important:** Uses both data preparations:
- `data_baseline`: Scaled data for baseline model
- `data_pann`: Unscaled data for PANN model

**Sections:**
1. Load Data
2. Train or Load Models (Option A: train fresh, Option B: load saved)
3. Comparison Summary
4. Side-by-Side Predictions
5. Residual Distribution Comparison
6. Metrics Comparison
7. Error Distribution Comparison
8. Statistical Summary
9. Key Takeaways

## Usage

### Quick Start

```bash
# Activate virtual environment
source keras_env/bin/activate

# Launch Jupyter
jupyter notebook notebooks/
```

### Individual Notebooks

```bash
# Baseline analysis
make notebook-baseline

# PANN analysis
make notebook-pann

# Comparison
make notebook-comparison
```

## PANN Model Configuration

The PANN model can be configured with different sizes:

| Configuration | Parameters | R² | RMSE (Pa) | MAE (Pa) |
|--------------|------------|-----|-----------|----------|
| n=16, layers=2 | 370 | 0.8997 | 2.13e+05 | 1.19e+05 |
| n=128, layers=2 | 17,282 | 0.9283 | 1.78e+05 | 1.07e+05 |
| n=256, layers=2 | 67,330 | 0.9318 | 1.72e+05 | 1.05e+05 |

**Recommended:** `n=128, layer_num=2` provides good balance of performance and model size.

## Data Requirements

- **Dataset:** `../../neohookean-data-generator/data/consolidated_all.npz`
- **Format:**
  - `strains`: (N, 3) - Green-Lagrange strain [E11, E22, γ12]
  - `stresses`: (N, 3) - PK2 stress [S11, S22, S12]

## Key Differences: Baseline vs PANN

| Aspect | Baseline | PANN |
|--------|----------|------|
| **Data Scaling** | ✅ StandardScaler | ❌ Physical values only |
| **Architecture** | Black-box NN | Physics-informed (F → Ψ → P → S) |
| **Parameters** | 11,843 (fixed) | 370 - 67,330 (configurable) |
| **R² Score** | 0.9932 | 0.8997 - 0.9318 |
| **Physics** | None | Polyconvexity, zero stress at identity |
| **Interpretability** | Low | High (energy-based) |

## Troubleshooting

### Issue: Import errors
**Solution:** Make sure to run `sys.path.append('..')` at the start of each notebook.

### Issue: PANN model not training
**Solution:** 
- Check that `scale=False` in `prepare_data()`
- Verify using physical (unscaled) strain/stress values
- Try smaller learning rate if training is unstable

### Issue: Different results from documented
**Solution:** Results may vary slightly due to random initialization. Use `random_state=42` for reproducibility.

## Notes

- **Training time:** Baseline ~2-3 min, PANN ~3-5 min (depends on model size)
- **GPU:** Not required but recommended for faster training
- **Memory:** ~2GB RAM sufficient for all notebooks
- **Python version:** Tested with Python 3.10+

## Citation

If using these notebooks, please note:
- PANN implementation adapted from Clément Jailin's reference implementation
- Energy-based constitutive modeling with input-convex neural networks (ICNNs)
