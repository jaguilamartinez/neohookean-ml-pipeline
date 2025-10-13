# Neo-Hookean ML Pipeline

Train neural networks to predict stress from strain in Neo-Hookean hyperelastic materials.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Explore data
python main.py --explore

# Train model
python main.py

# Train with custom parameters
python main.py --architecture physics --epochs 150 --batch_size 64
```

## Configuration

Use JSON config files for reproducible experiments:

```json
{
  "architecture": "deep",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "test_size": 0.2,
  "val_size": 0.2,
  "scaler": "standard"
}
```

Then run: `python main.py --config config.json`

## Architecture Options

- `simple`: 2-layer network (64, 32)
- `deep`: 5-layer network (128, 128, 64, 64, 32)
- `wide`: 3-layer wide network (256, 256, 128)
- `physics`: 7-layer network for constitutive modeling

## Results

Training outputs are saved in timestamped directories under `results/`:
- `best_model.h5`: Trained model
- `preprocessor/`: Data preprocessing artifacts
- `results.json`: Training metrics
- `*.png`: Visualization plots

## Data

Expects dataset from neohookean-dataset-generator with:
- `consolidated_dataset.npz`: Main dataset
- `training_data.npz`: Preprocessed training data