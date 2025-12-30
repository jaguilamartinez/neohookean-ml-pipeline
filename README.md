# Neo-Hookean ML Pipeline

ML models for stress-strain prediction in hyperelastic materials.

## Setup

```bash
make setup
source keras_env/bin/activate
```

## Models

**Baseline**: Feedforward network (128→64→32 with BatchNorm, Dropout)  
Direct mapping: strain → stress using StandardScaler

**PANN**: Physics-Augmented Neural Network  
Energy-based: strain → energy (ICNN) → stress (automatic differentiation)  
Ensures zero stress at reference, convexity, and thermodynamic consistency

## Usage

### Training
```bash
make train-baseline
make train-pann
```

### Analysis
Jupyter notebooks in `notebooks/`:
- `baseline_analysis.ipynb` - Baseline model training and evaluation
- `pann_analysis.ipynb` - PANN model training and evaluation  
- `comparison.ipynb` - Side-by-side model comparison
- `generalization_test.ipynb` - Test on unseen scenarios
- `validate_data_splitting.ipynb` - Verify no data leakage

Launch: `make notebook`

## Data

Requires: `../neohookean-data-generator/data/consolidated_all.npz`

Input: Green-Lagrange strain tensor in Voigt notation [E_xx, E_yy, γ_xy]  
Output: PK2 stress tensor in Voigt notation [S_xx, S_yy, S_xy]  
Plane strain condition enforced

## Structure

```
src/
  data.py           - Data loading, splitting, scaling
  visualization.py  - Plotting utilities  
  comparison.py     - Model comparison tools
  models/
    base_model.py   - Baseline feedforward network
    pann_model.py   - ICNN-based energy network
notebooks/          - Analysis and experiments
output/             - Saved models and metrics
```
