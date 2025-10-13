#!/usr/bin/env python3

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.append('src')

from data.loader import DataLoader
from data.preprocessor import Preprocessor
from models.network import NeuralNetwork
from utils.evaluation import calculate_metrics, plot_predictions, plot_residuals, plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network on Neo-Hookean data')
    
    parser.add_argument('--data_path', default='../neohookean-dataset-generator/hyperelastic_comprehensive_experiments',
                       help='Path to dataset')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    
    # Model parameters
    parser.add_argument('--architecture', choices=['simple', 'deep', 'wide', 'physics'], 
                       default='deep', help='Network architecture')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    
    # Data parameters
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size')
    parser.add_argument('--scaler', choices=['standard', 'minmax'], default='standard', 
                       help='Feature scaling method')
    
    parser.add_argument('--config', help='JSON config file')
    parser.add_argument('--explore', action='store_true', help='Just explore data')
    
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def explore_data(data_path):
    loader = DataLoader(data_path)
    dataset = loader.load_dataset()
    df = loader.to_dataframe(dataset)
    
    print(f"Dataset: {dataset['n_samples']} samples, {dataset['n_cases']} cases")
    print(f"Strain components: {list(dataset['strain_components'])}")
    print(f"Stress components: {list(dataset['stress_components'])}")
    print(f"Data shape: {dataset['strains'].shape}")
    
    # Basic statistics
    print("\nStrain statistics:")
    for i, comp in enumerate(dataset['strain_components']):
        data = dataset['strains'][:, i]
        print(f"  {comp}: [{data.min():.6f}, {data.max():.6f}] (mean: {data.mean():.6f})")
    
    print("\nStress statistics:")
    for i, comp in enumerate(dataset['stress_components']):
        data = dataset['stresses'][:, i]
        print(f"  {comp}: [{data.min():.0f}, {data.max():.0f}] Pa (mean: {data.mean():.0f})")


def train_model(args):
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    loader = DataLoader(args.data_path)
    dataset = loader.load_dataset()
    
    X = dataset['strains']
    y = dataset['stresses']
    case_numbers = dataset['case_numbers']
    
    print(f"Loaded dataset: {X.shape[0]} samples")
    
    # Validate case structure
    n_cases, steps_per_case = loader.validate_case_integrity(dataset)
    
    preprocessor = Preprocessor(
        test_size=args.test_size,
        val_size=args.val_size,
        scaler=args.scaler
    )
    
    data = preprocessor.preprocess(X, y, case_numbers)
    
    print(f"\nFinal data splits:")
    print(f"  Train: {data['X_train'].shape[0]} samples")
    print(f"  Validation: {data['X_val'].shape[0]} samples")
    print(f"  Test: {data['X_test'].shape[0]} samples")
    
    # Build and train model
    model = NeuralNetwork(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        architecture=args.architecture,
        learning_rate=args.learning_rate
    )
    
    model.build_model()
    print(f"Model parameters: {model.model.count_params()}")
    
    checkpoint_path = output_dir / 'best_model.h5'
    
    history = model.train(
        data['X_train_scaled'], data['y_train_scaled'],
        data['X_val_scaled'], data['y_val_scaled'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=str(checkpoint_path)
    )
    
    # Save preprocessor and model
    preprocessor.save(output_dir / 'preprocessor')
    model.save(checkpoint_path)
    
    # Evaluate on test set
    y_pred_scaled = model.predict(data['X_test_scaled'])
    y_pred = preprocessor.inverse_transform_y(y_pred_scaled)
    y_test = data['y_test']
    
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2e}")
    print(f"  MAE: {metrics['mae']:.2e}")
    
    # Save results
    results = {
        'metrics': metrics,
        'config': vars(args),
        'model_params': model.model.count_params()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create plots
    plot_training_history(history, output_dir / 'training_history.png')
    plot_predictions(y_test, y_pred, "Test Set", output_dir / 'predictions.png')
    plot_residuals(y_test, y_pred, "Test Set", output_dir / 'residuals.png')
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    args = parse_args()
    
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    if args.explore:
        explore_data(args.data_path)
    else:
        train_model(args)


if __name__ == "__main__":
    main()