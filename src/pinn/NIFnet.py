'''
Created on 16 Apr 2025

@author: radar
'''

import tensorflow as tf
import keras

class SIRENActivation(keras.layers.Layer):
    
    def __init__(self, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        
        return tf.sin(self.w0 * inputs)

class SIRENFirstLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0
        
    def __call__(self, shape, dtype=None, **kwargs):
        # Dynamically determine number of input features from shape
        in_features = shape[0]
        limit = 1.0 / in_features / self.w0

        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

class SIRENIntermediateLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0

    def __call__(self, shape, dtype=None, **kwargs):
        
        in_features = shape[1]
        limit = tf.sqrt(6 / in_features) / self.w0
        
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

class SIRENBiasInitializer(keras.initializers.Initializer):

    def __call__(self, shape, dtype=None, **kwargs):
        
        in_features = shape[0]
        
        limit = 1.0 / in_features
        
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

class ResidualBlock(keras.layers.Layer):
    
    def __init__(self, units, nlayers=2, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        
        layers = []
        
        for _ in range(nlayers):
            
            dense = keras.layers.Dense(
                units,
                kernel_initializer=SIRENIntermediateLayerInitializer(w0=w0),
                bias_initializer=SIRENBiasInitializer()
            )
            layers.append(dense)
        
        self.layers = layers
        self.activation = SIRENActivation(w0)

    def call(self, inputs):
        
        x = inputs
        for layer in self.layers:
            
            x = self.activation(layer(x))
        
        return 0.5 * (inputs + x)

class ParameterNet(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, w0=1.0, hidden_units=128, film_units=128, **kwargs):
        
        super().__init__(**kwargs)
        # Pre-layer: project t (1D) to hidden_units.
        self.pre_layer = keras.layers.Dense(
            hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(w0=w0),
            bias_initializer=SIRENBiasInitializer(),
            activation = SIRENActivation(w0=w0),
        )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers=nlayers,w0=w0) for _ in range(num_blocks)]
        # Final layer outputs FiLM parameters (γ and β).
        self.film_layer = keras.layers.Dense(
            2 * film_units,
            kernel_initializer='glorot_normal',
            bias_initializer="zeros",
        )

    def call(self, inputs):
        
        x = self.pre_layer(inputs)
        
        for block in self.blocks:
            x = block(x)
            
        film_params = self.film_layer(x)
        gamma, beta = tf.split(film_params, num_or_size_splits=2, axis=-1)
        
        return gamma, beta

class ShapeNet(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, w0=1.0, hidden_units=128, **kwargs):
        
        super().__init__(**kwargs)
        # Pre-layer: project (x,y,z) to hidden_units.
        self.pre_layer = keras.layers.Dense(
            hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(w0=w0),
            bias_initializer=SIRENBiasInitializer(),
            activation = SIRENActivation(w0=w0),
        )
        self.blocks = [ResidualBlock(hidden_units, nlayers=nlayers, w0=w0) for _ in range(num_blocks)]

    def call(self, inputs):
        
        x = self.pre_layer(inputs)
        
        for block in self.blocks:
            x = block(x)
            
        return x

class NIFNet(keras.Model):
    
    def __init__(self, num_blocks, nlayers=2, noutputs=3, w0=1.0, hidden_units=128, **kwargs):
        
        super().__init__(**kwargs)
        
        self.shape_net = ShapeNet(num_blocks, nlayers=nlayers, w0=w0, hidden_units=hidden_units)
        
        self.parameter_net = ParameterNet(num_blocks, nlayers=nlayers, w0=w0, hidden_units=hidden_units, film_units=hidden_units)
        
        self.final_dense = keras.layers.Dense(
            noutputs,
            kernel_initializer='glorot_normal',
            bias_initializer="zeros",
        )
        
        self.alphas = tf.ones( (nlayers+1, ), name="alphas" )

    def call(self, inputs):
        
        temporal_inputs = inputs[:,0:1]
        spatial_inputs = inputs[:,1:4]
        
        shape_features = self.shape_net(spatial_inputs)
        
        gamma, beta = self.parameter_net(temporal_inputs)
        # FiLM: Modulate shape features using γ and β.
        modulated_features = gamma * shape_features + beta
        
        return self.final_dense(modulated_features)
    
    def update_mask(self, in_shape):
        
        pass