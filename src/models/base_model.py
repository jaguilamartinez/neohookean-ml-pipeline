"""Baseline feedforward model for stress prediction."""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime


def build_model(input_dim=3, output_dim=3, learning_rate=0.001):
    """Build 3-layer feedforward network (128→64→32) with BatchNorm and Dropout."""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(output_dim)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(model, data, epochs=100, batch_size=32, verbose=1):
    """Train with early stopping and learning rate reduction."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=verbose
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=verbose
        )
    ]
    
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history


def evaluate_model(model, data):
    """Evaluate on test set, return metrics (R², RMSE, MAE) and predictions."""
    y_pred_model = model.predict(data['X_test'], verbose=0)
    
    if data['y_scaler'] is not None:
        y_pred = data['y_scaler'].inverse_transform(y_pred_model)
    else:
        y_pred = y_pred_model
    
    y_test = data['y_test_original']
    
    metrics = {
        'r2': float(r2_score(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred))
    }
    
    component_metrics = {}
    components = ['S_xx', 'S_yy', 'S_xy']
    for i, comp in enumerate(components):
        component_metrics[comp] = {
            'r2': float(r2_score(y_test[:, i], y_pred[:, i])),
            'rmse': float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))),
            'mae': float(mean_absolute_error(y_test[:, i], y_pred[:, i]))
        }
    
    metrics['components'] = component_metrics
    
    return metrics, y_pred


def save_results(model, metrics, history, output_dir):
    """Save model, metrics, and history to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(output_dir / 'model.h5')
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']]
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


def load_trained_model(model_path):
    """Load trained model from .h5 file."""
    return keras.models.load_model(model_path)


def run_training(data_path, output_dir=None, epochs=100, batch_size=32, 
                 learning_rate=0.001, random_state=42):
    """Complete training pipeline: load data, train, evaluate, save."""
    from ..data import load_dataset, prepare_data
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('output') / f'run_{timestamp}'
    else:
        output_dir = Path(output_dir)
    
    print("Loading data...")
    dataset = load_dataset(data_path)
    print(f"Loaded: {dataset['n_samples']} samples")
    
    print("\nPreparing data...")
    data = prepare_data(
        dataset['strains'],
        dataset['stresses'],
        random_state=random_state,
        scale=True
    )
    print(f"Train: {data['splits']['train']}, Val: {data['splits']['val']}, Test: {data['splits']['test']}")
    
    print("\nBuilding model...")
    model = build_model(learning_rate=learning_rate)
    print(f"Parameters: {model.count_params():,}")
    
    print("\nTraining...")
    history = train_model(model, data, epochs=epochs, batch_size=batch_size)
    
    print("\nEvaluating...")
    metrics, y_pred = evaluate_model(model, data)
    
    print("\nTest Metrics:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2e} Pa")
    print(f"  MAE:  {metrics['mae']:.2e} Pa")
    
    save_results(model, metrics, history, output_dir)
    
    return model, metrics, history, output_dir
