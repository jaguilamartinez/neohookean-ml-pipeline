"""
Physics-Augmented Neural Network (PANN) for plane strain hyperelasticity.

Computes C from Green-Lagrange strain (C = I + 2E), learns strain energy Ψ(C)
using an ICNN over isotropic invariants, and derives PK2 stress as S = 2*dΨ/dC.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
from pathlib import Path
from datetime import datetime


# ==================== Invariants from C (plane strain) ====================

def compute_inv_plane_strain_C(C):
    """
    Compute invariants for isotropic hyperelasticity from C.

    Returns [I1, I2, I3, -2J] where I1 = tr(C), I2 = 0.5*(tr(C)^2 - tr(C^2)),
    I3 = det(C), J = sqrt(I3). The -2J term enables efficient stress correction.
    """
    C = tf.cast(C, tf.float64)

    I1 = tf.linalg.trace(C)

    C2 = tf.linalg.matmul(C, C)
    trC2 = tf.linalg.trace(C2)
    I2 = 0.5 * (tf.square(I1) - trC2)

    I3 = tf.linalg.det(C)

    I3_safe = tf.maximum(I3, 1e-24)
    J = tf.sqrt(I3_safe)

    return tf.stack([I1, I2, I3, -2.0 * J], axis=-1)


def compute_growth(J):
    """Volumetric penalty term: (J + 1/J - 2)^2."""
    J = tf.cast(J, tf.float64)
    J_safe = tf.maximum(J, 1e-12)
    return tf.square(J_safe + 1.0 / J_safe - 2.0)


# ==================== ICNN Energy Model ====================

class ICNNEnergy(layers.Layer):
    """
    ICNN strain energy Ψ_NN over invariants with reference-state corrections.

    Enforces Ψ(I) = 0 and S(I) = 0 via energy shift and stress correction,
    plus a volumetric penalty term.
    """

    def __init__(self, n=16, layer_num=2, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.layer_num = layer_num

        positive_init = tf.keras.initializers.RandomUniform(
            minval=1e-3, maxval=0.1, seed=42
        )

        self.hidden_layers = [
            layers.Dense(
                units=n,
                activation="softplus",
                kernel_initializer=positive_init,
                kernel_constraint=NonNeg(),
                name=f"hidden_{i}",
            )
            for i in range(layer_num)
        ]

        # Non-negative kernel constraint preserves ICNN convexity
        self.final_layer = layers.Dense(
            units=1,
            activation="softplus",
            kernel_initializer=positive_init,
            kernel_constraint=NonNeg(),
            name="final_dense",
        )

        # Learnable energy scale (softplus ensures positivity)
        # Initialized to give ~1e5 after activation
        self.raw_energy_scale = self.add_weight(
            name="raw_energy_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(11.5),
            trainable=True,
        )

    @property
    def energy_scale(self):
        return tf.nn.softplus(self.raw_energy_scale)

    def _psi_from_inv(self, inv_features):
        """Forward pass: invariants → Ψ_NN."""
        h = inv_features
        for layer in self.hidden_layers:
            h = layer(h)
        energy = self.final_layer(h)
        return tf.squeeze(energy, axis=-1)

    def call(self, C):
        """
        Compute Ψ_PANN(C) = Ψ_NN(inv(C)) - Ψ_NN(inv(I)) - (J-1)*c + Ψ_vol(J).
        """
        # Invariants at current state
        inv = compute_inv_plane_strain_C(C)
        inv_f32 = tf.cast(inv, tf.float32)

        psi = self._psi_from_inv(inv_f32)  # (B,)

        # Invariants at reference (identity)
        I_C = tf.eye(3, dtype=tf.float64)[tf.newaxis, ...]  # (1,3,3)
        inv_ref = compute_inv_plane_strain_C(I_C)  # (1,4)
        inv_ref_f32 = tf.cast(inv_ref, tf.float32)

        # Compute stress correction coefficient at reference state
        with tf.GradientTape() as tape:
            tape.watch(inv_ref_f32)
            psi_ref = self._psi_from_inv(inv_ref_f32)  # (1,)
        dpsi_dinv_ref = tape.gradient(psi_ref, inv_ref_f32)  # (1,4)

        # Weighted sum to enforce S(I) = 0
        weights = tf.constant([1.0, 2.0, 1.0, -1.0], dtype=dpsi_dinv_ref.dtype)[tf.newaxis, :]
        stress_correction_coeff = tf.reduce_sum(2.0 * dpsi_dinv_ref * weights, axis=1)  # (1,)

        # Extract J from inv = [I1, I2, I3, -2J]
        J = tf.cast(-inv[:, 3] / 2.0, tf.float64)  # (B,)

        # Broadcast correction coefficient to batch
        batch_size = tf.shape(C)[0]
        stress_correction_coeff = tf.tile(tf.cast(stress_correction_coeff, tf.float64), [batch_size])  # (B,)

        correction = stress_correction_coeff * (J - 1.0)
        volumetric_penalty = compute_growth(J)

        psi_ref_scalar = tf.cast(psi_ref[0], tf.float64)
        psi_corrected = tf.cast(psi, tf.float64) - psi_ref_scalar - correction + volumetric_penalty

        return tf.cast(self.energy_scale, tf.float64) * psi_corrected


# ==================== Main PANN Model ====================

class PANNModel(keras.Model):
    """
    PANN for plane strain hyperelasticity.

    Input: Green-Lagrange strain [E11, E22, gamma12] where gamma12 = 2*E12
    Output: PK2 stress [S11, S22, S12]

    Computes C = I + 2E and derives stress as S = 2*dΨ/dC.
    """

    def __init__(self, n=16, layer_num=2, name="pann_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.icnn_energy = ICNNEnergy(n=n, layer_num=layer_num)

    @staticmethod
    def E_voigt_to_C(E_voigt):
        """Convert Voigt strain to right Cauchy-Green tensor: C = I + 2E."""
        E_voigt = tf.cast(E_voigt, tf.float64)
        E11 = E_voigt[:, 0]
        E22 = E_voigt[:, 1]
        gamma12 = E_voigt[:, 2]
        E12 = gamma12 / 2.0

        B = tf.shape(E_voigt)[0]
        z = tf.zeros((B,), dtype=tf.float64)

        # E tensor (plane strain: E33 = 0)
        row1 = tf.stack([E11, E12, z], axis=1)
        row2 = tf.stack([E12, E22, z], axis=1)
        row3 = tf.stack([z,   z,   z], axis=1)
        E = tf.stack([row1, row2, row3], axis=1)  # (B,3,3)

        I = tf.eye(3, batch_shape=[B], dtype=tf.float64)
        return I + 2.0 * E

    @staticmethod
    def S_tensor_to_voigt(S):
        """Extract [S11, S22, S12] from a (B,3,3) tensor."""
        S = tf.cast(S, tf.float32)
        return tf.stack([S[:, 0, 0], S[:, 1, 1], S[:, 0, 1]], axis=1)

    def call(self, E_voigt, training=None):
        # Build C from strain
        C = self.E_voigt_to_C(E_voigt)  # (B,3,3)

        # Compute S = 2*dΨ/dC via automatic differentiation
        # watch_accessed_variables=False avoids tracking model weights
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(C)
            psi = self.icnn_energy(C)  # (B,)
        dpsi_dC = tape.gradient(psi, C)  # (B,3,3)

        S = 2.0 * dpsi_dC
        return self.S_tensor_to_voigt(S)


# ==================== Training Utilities ====================

def build_pann_model(n=16, layer_num=2, learning_rate=1e-3):
    """Build and compile the PANN model."""
    model = PANNModel(n=n, layer_num=layer_num)

    # Build by calling once
    _ = model(tf.zeros((1, 3), dtype=tf.float32))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_pann_model(model, data, epochs=100, batch_size=32, verbose=1):
    """Train PANN with early stopping and learning rate reduction."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=verbose,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=verbose,
        ),
    ]

    history = model.fit(
        data["X_train"].astype(np.float32),
        data["y_train"].astype(np.float32),
        validation_data=(
            data["X_val"].astype(np.float32),
            data["y_val"].astype(np.float32),
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


def run_pann_training(
    data_path,
    output_dir=None,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    n=16,
    layer_num=2,
    random_state=42,
):
    """End-to-end PANN training pipeline (no scaling for physical consistency)."""
    from ..data import load_dataset, prepare_data
    from .base_model import evaluate_model, save_results

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / f"pann_run_{timestamp}"
    else:
        output_dir = Path(output_dir)

    print("Loading data...")
    dataset = load_dataset(data_path)
    print(f"Loaded: {dataset['n_samples']} samples")

    print("\nPreparing data (no scaling)...")
    data = prepare_data(
        dataset["strains"],
        dataset["stresses"],
        random_state=random_state,
        scale=False,
    )
    print(
        f"Train: {data['splits']['train']}, "
        f"Val: {data['splits']['val']}, "
        f"Test: {data['splits']['test']}"
    )

    print("\nBuilding PANN model...")
    model = build_pann_model(
        learning_rate=learning_rate,
        n=n,
        layer_num=layer_num,
    )
    print(f"Parameters: {model.count_params():,}")

    print("\n" + "=" * 60)
    print("Training PANN model")
    print("=" * 60)
    history = train_pann_model(
        model, data, epochs=epochs, batch_size=batch_size, verbose=1
    )

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    metrics, y_pred = evaluate_model(model, data)

    print("\nTest Metrics:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2e} Pa")
    print(f"  MAE:  {metrics['mae']:.2e} Pa")

    save_results(model, metrics, history, output_dir)
    return model, metrics, history, output_dir