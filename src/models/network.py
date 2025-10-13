import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from pathlib import Path


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, architecture='deep', 
                 dropout=0.2, l2_reg=1e-4, learning_rate=1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.architecture = architecture
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        
        self.architectures = {
            'simple': [64, 32],
            'deep': [128, 128, 64, 64, 32],
            'wide': [256, 256, 128],
            'physics': [128, 128, 96, 96, 64, 64, 32]
        }
    
    def build_model(self):
        layers = self.architectures.get(self.architecture, self.architectures['deep'])
        
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        
        for i, units in enumerate(layers):
            x = keras.layers.Dense(units, 
                                 kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Dropout(self.dropout)(x)
        
        outputs = keras.layers.Dense(self.output_dim,
                                   kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def create_callbacks(self, checkpoint_path):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True, monitor='val_loss'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              checkpoint_path='best_model.h5'):
        
        if self.model is None:
            self.build_model()
        
        callbacks = self.create_callbacks(checkpoint_path)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not built")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        if self.model is None:
            raise ValueError("Model not built")
        return self.model.evaluate(X, y, verbose=0)
    
    def save(self, path):
        if self.model is None:
            raise ValueError("Model not built")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path)
        
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'architecture': self.architecture,
            'dropout': self.dropout,
            'l2_reg': self.l2_reg,
            'learning_rate': self.learning_rate
        }
        
        with open(path.parent / 'model_config.json', 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path):
        path = Path(path)
        
        with open(path.parent / 'model_config.json', 'r') as f:
            config = json.load(f)
        
        nn = cls(**config)
        nn.model = keras.models.load_model(path)
        
        return nn