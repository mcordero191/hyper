'''
Created on 16 Apr 2025

@author: radar
'''

import tensorflow as tf
import keras

# -----------------------------------------------------------------------------
# SIREN Activation: Scales input by w0 and applies sine.
# -----------------------------------------------------------------------------
class SIRENActivation(keras.layers.Layer):
    
    def __init__(self, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        
        return tf.sin(self.w0 * inputs)

# -----------------------------------------------------------------------------
# Custom Weight Initializers following the paper's recommendations
# -----------------------------------------------------------------------------
class SIRENFirstLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, in_features):
        
        self.in_features = in_features

    def __call__(self, shape, dtype=None, **kwargs):
        
        limit = 1.0 / self.in_features
        
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

class SIRENIntermediateLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, in_features, w0=1.0):
        
        self.in_features = in_features
        self.w0 = w0

    def __call__(self, shape, dtype=None, **kwargs):
        
        limit = tf.sqrt(6 / self.in_features) / self.w0
        
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

# -----------------------------------------------------------------------------
# Custom Bias Initializer: initializing biases to zero is recommended.
# -----------------------------------------------------------------------------
class SIRENBiasInitializer(keras.initializers.Initializer):
    
    def __init__(self, in_features):
        
        self.in_features = in_features

    def __call__(self, shape, dtype=None, **kwargs):
        
        limit = 1.0 / self.in_features
        
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

# -----------------------------------------------------------------------------
# Residual Block: uses a Dense layer with SIREN activation and adds a residual.
# -----------------------------------------------------------------------------
class ResidualBlock(keras.layers.Layer):
    
    def __init__(self, units, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            units,
            kernel_initializer=SIRENIntermediateLayerInitializer(units, w0=w0),
            bias_initializer=SIRENBiasInitializer()
        )
        self.activation = SIRENActivation(w0)

    def call(self, inputs):
        
        output = self.activation(self.dense(inputs))
        
        return 0.5 * (inputs + output)

# -----------------------------------------------------------------------------
# ParameterNet: Processes 1D temporal input (t) with an explicit pre-layer.
# -----------------------------------------------------------------------------
class ParameterNet(keras.Model):
    
    def __init__(self, num_blocks, w0=1.0, hidden_units=128, film_units=128, **kwargs):
        
        super().__init__(**kwargs)
        # Pre-layer: project t (1D) to hidden_units.
        self.pre_layer = keras.layers.Dense(
            hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(1),
            bias_initializer=SIRENBiasInitializer()
        )
        self.blocks = [ResidualBlock(hidden_units, w0=w0) for _ in range(num_blocks)]
        # Final layer outputs FiLM parameters (γ and β).
        self.film_layer = keras.layers.Dense(
            2 * film_units,
            kernel_initializer='glorot_uniform',
            bias_initializer=SIRENBiasInitializer()
        )

    def call(self, inputs):
        
        x = self.pre_layer(inputs)
        
        for block in self.blocks:
            x = block(x)
            
        film_params = self.film_layer(x)
        gamma, beta = tf.split(film_params, num_or_size_splits=2, axis=-1)
        
        return gamma, beta

# -----------------------------------------------------------------------------
# ShapeNet: Processes 3D spatial input (x,y,z) with an explicit pre-layer.
# -----------------------------------------------------------------------------
class ShapeNet(keras.Model):
    
    def __init__(self, num_blocks, w0=1.0, hidden_units=128, **kwargs):
        
        super().__init__(**kwargs)
        # Pre-layer: project (x,y,z) to hidden_units.
        self.pre_layer = keras.layers.Dense(
            hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(3),
            bias_initializer=SIRENBiasInitializer()
        )
        self.blocks = [ResidualBlock(hidden_units, w0=w0) for _ in range(num_blocks)]

    def call(self, inputs):
        
        x = self.pre_layer(inputs)
        
        for block in self.blocks:
            x = block(x)
            
        return x

# -----------------------------------------------------------------------------
# SuperNet: Instantiates ParameterNet and ShapeNet internally,
# and combines their outputs using FiLM modulation.
# -----------------------------------------------------------------------------
class SuperNet(keras.Model):
    
    def __init__(self, num_blocks, w0=1.0, hidden_units=128, **kwargs):
        
        super().__init__(**kwargs)
        
        self.shape_net = ShapeNet(num_blocks, w0=w0, hidden_units=hidden_units)
        
        self.parameter_net = ParameterNet(num_blocks, w0=w0, hidden_units=hidden_units, film_units=hidden_units)
        
        self.final_dense = keras.layers.Dense(
            3,
            kernel_initializer='glorot_uniform',
            bias_initializer="zeros",
        )

    def call(self, spatial_inputs, temporal_inputs):
        
        shape_features = self.shape_net(spatial_inputs)
        
        gamma, beta = self.parameter_net(temporal_inputs)
        # FiLM: Modulate shape features using γ and β.
        modulated_features = gamma * shape_features + beta
        
        return self.final_dense(modulated_features)