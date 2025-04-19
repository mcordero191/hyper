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

class FiLMNet(keras.Model):
    
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
    
# === Multi-layer FiLM ShapeNet ===
class MultiShapeNet(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, w0=1.0, hidden_units=128):
        
        super().__init__()
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0) for _ in range(num_blocks)]
        
        self.pre_layer = keras.layers.Dense(
            hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(w0),
            bias_initializer=SIRENBiasInitializer(),
            activation=SIRENActivation(w0),
        )

    def call(self, x, gammas, betas):
        
        x = self.pre_layer(x)
        for i, block in enumerate(self.blocks):
            x = gammas[i] * x + betas[i]
            x = block(x)
        return x

# === ParameterNet for Multi-layer FiLM ===
class MultiParameterNet(keras.layers.Layer):
    
    def __init__(self, num_blocks, num_modulations, nlayers=2, w0=1.0, hidden_units=128):
        
        super().__init__()
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0) for _ in range(num_blocks)]
        
        self.pre_layer = keras.layers.Dense(
            hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(w0),
            bias_initializer=SIRENBiasInitializer(),
            activation=SIRENActivation(w0),
        )
        
        self.film_layer = keras.layers.Dense(
            2 * num_modulations * hidden_units,  # gamma + beta for each layer
            kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        )
        self.hidden_units = hidden_units
        self.num_modulations = num_modulations

    def call(self, t):
        
        x = self.pre_layer(t)
        
        for block in self.blocks:
            x = block(x)
            
        film_params = self.film_layer(x)
        film_params = tf.reshape(film_params, (-1, self.num_modulations, 2 * self.hidden_units))
        gamma, beta = tf.split(film_params, num_or_size_splits=2, axis=-1)
        
        return gamma, beta

# === Full Models ===
class MultiFiLMNet(keras.Model):
    
    def __init__(self, num_blocks=4, nlayers=2, hidden_units=128, noutputs=3, w0=1.0):
        
        super().__init__()
        
        self.shapenet = MultiShapeNet(num_blocks, nlayers, w0, hidden_units)
        self.parameternet = MultiParameterNet(num_blocks, num_blocks, nlayers, w0, hidden_units)
        self.final = keras.layers.Dense(noutputs)

    def call(self, inputs):
        
        t = inputs[:, :1]
        x = inputs[:, 1:4]
        
        gamma, beta = self.parameternet(t)
        features = self.shapenet(x, gamma, beta)
        
        return self.final(features)
    
# === Cross-Attention ShapeNet ===
class AttentionShapeNet(keras.layers.Layer):
    
    def __init__(self, hidden_units=128, w0=1.0):
        super().__init__()
        self.spatial_proj = keras.layers.Dense(hidden_units)
        self.temporal_proj = keras.layers.Dense(hidden_units)
        self.attention = keras.layers.Attention(use_scale=True)
        self.final_dense = keras.layers.Dense(hidden_units, activation=SIRENActivation(w0))

    def call(self, spatial, temporal):
        q = self.spatial_proj(spatial)
        k = self.temporal_proj(temporal)
        v = k
        attended = self.attention([q, k, v])
        return self.final_dense(attended)

class AttentionFusionNet(keras.Model):
    
    def __init__(self, hidden_units=128, noutputs=3, w0=1.0):
        super().__init__()
        self.shapenet = keras.layers.Dense(hidden_units, activation=SIRENActivation(w0))
        self.temporalnet = keras.layers.Dense(hidden_units, activation=SIRENActivation(w0))
        self.attn_block = AttentionShapeNet(hidden_units, w0)
        self.final = keras.layers.Dense(noutputs)

    def call(self, inputs):
        t = inputs[:, :1]
        x = inputs[:, 1:4]
        spatial_embed = self.shapenet(x)
        temporal_embed = self.temporalnet(t)
        fused = self.attn_block(spatial_embed, temporal_embed)
        return self.final(fused)