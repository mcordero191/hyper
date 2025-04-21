'''
Created on 16 Apr 2025

@author: radar
'''
import numpy as np
import tensorflow as tf
import keras

from pinn.layers import Scaler

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

class ResidualBlock0(keras.layers.Layer):
    
    def __init__(self, units, nlayers=2, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        
        layers = []
        
        for _ in range(nlayers):
            
            dense = keras.layers.Dense(
                units,
                kernel_initializer="HeNormal", #SIRENIntermediateLayerInitializer(w0=w0),
                bias_initializer="zeros", #SIRENBiasInitializer()
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
            kernel_initializer="HeNormal",#SIRENFirstLayerInitializer(w0=w0),
            bias_initializer="zeros", #SIRENBiasInitializer(),
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
            kernel_initializer="HeNormal", #SIRENFirstLayerInitializer(w0=w0),
            bias_initializer="zeros", #SIRENBiasInitializer(),
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
        
        self.scaler = Scaler()

    def call(self, inputs):
        
        temporal_inputs = inputs[:,0:1]
        spatial_inputs = inputs[:,1:4]
        
        shape_features = self.shape_net(spatial_inputs)
        
        gamma, beta = self.parameter_net(temporal_inputs)
        # FiLM: Modulate shape features using γ and β.
        modulated_features = gamma * shape_features + beta
    
        output = self.final_dense(modulated_features)
        
        return(self.scaler(output))
    
    def update_mask(self, in_shape):
        
        pass
    
class MultiShapeNet0(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, w0=1.0, hidden_units=128):
        
        super().__init__()
        self.pre_layer = keras.layers.Dense(
            hidden_units,
            kernel_initializer="HeNormal",
            bias_initializer="zeros",
            activation=SIRENActivation(w0),
        )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0)
                       for _ in range(num_blocks)]

    def call(self, x, gammas, betas):
        
        # x: (batch, 3)  ->  h: (batch, hidden_units)
        h = self.pre_layer(x)
        
        for i, block in enumerate(self.blocks):
            # Correct indexing: use [:,i,:] not [i]
            h = gammas[:, i, :] * h + betas[:, i, :]
            h = block(h)
            
        return h

class FourierFeatureEmbedding0(keras.layers.Layer):
    
    def __init__(self, num_fourier=4, w0=1.0, period=2.0):
        
        super().__init__()
        self.num_fourier = num_fourier
        self.period = num_fourier
        
        init_freqs = [ (num_fourier-i)*w0 for i in range(num_fourier) ]
        init_freqs = tf.constant(init_freqs, dtype=tf.float32)
        
        self.freqs = tf.Variable(initial_value=init_freqs, trainable=True, name="trainable_freqs")

    def call(self, t):
        
        scaled = t * self.freqs[None, :] * (2 * np.pi / self.period)
        
        sin_feats = tf.sin(scaled)
        cos_feats = tf.cos(scaled)
        
        # Interleave sin and cos: shape -> (batch, num_fourier, 2)
        sin_cos = tf.stack([sin_feats, cos_feats], axis=2)
        
        # Flatten to (batch, 2*num_fourier) as [sin(w1 t), cos(w1 t), sin(w2 t), cos(w2 t), ...]
        interleaved = tf.reshape(sin_cos, (-1, 2 * self.num_fourier))
        
        return interleaved  # (batch, 2*num_fourier)


class MultiParameterNet0(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, hidden_units=128,
                 num_fourier=32, w0=1.0):
        
        super().__init__()
        # Fourier features for time
        self.fourier = FourierFeatureEmbedding(hidden_units)
        
        # self.pre = keras.layers.Dense(hidden_units,
        #                               kernel_initializer="HeNormal", #SIRENFirstLayerInitializer(w0),
        #                               bias_initializer="zeros", #SIRENBiasInitializer(),
        #                               activation=SIRENActivation(w0))
        
        self.blocks = [ResidualBlock(hidden_units*2, nlayers, w0)
                       for _ in range(num_blocks-1)]
        
        # output gamma/beta for each block
        # self.film = keras.layers.Dense(2 * num_blocks * hidden_units,
        #                                kernel_initializer='glorot_normal',
        #                                bias_initializer='zeros')
        
        self.num_blocks = num_blocks
        self.hidden_units = hidden_units
        
    def call(self, t):
        
        params = []
        # embed time -> fourier features
        h = self.fourier(t)
        # h = self.pre(h)
        
        params.append(h)
        
        for blk in self.blocks:
            h = blk(h)
            params.append(h)
            
        # params = self.film(h)
        
        params = tf.reshape(params, (-1, self.num_blocks, 2 * self.hidden_units))
        
        gamma, beta = tf.split(params, 2, axis=-1)
        
        return gamma, beta

# === Full Models ===
class MultiFiLMNet0(keras.Model):
    
    def __init__(self, num_blocks=4, nlayers=2, hidden_units=128, noutputs=3, w0=1.0):
        
        super().__init__()
        
        self.shapenet = MultiShapeNet(num_blocks, nlayers, w0, hidden_units)
        self.parameternet = MultiParameterNet(num_blocks, nlayers=nlayers, w0=w0, hidden_units=hidden_units)
        self.final = keras.layers.Dense(noutputs)
        
        self.scaler = Scaler()

    def call(self, inputs):
        
        t = inputs[:, :1]
        x = inputs[:, 1:4]
        
        gamma, beta = self.parameternet(t)
        features = self.shapenet(x, gamma, beta)
        
        output = self.final(features)
        
        return(self.scaler(output))

# === Fourier Feature Embedding ===
class FourierFeatureEmbedding(keras.layers.Layer):
    
    def __init__(self, num_fourier=16, w0=1.0, period=24.0):
        
        super().__init__()
        self.num_fourier = num_fourier
        self.period = period
        # trainable frequencies initialized as multiples of w0
        init = tf.constant([(num_fourier-i)*w0 for i in range(num_fourier)], dtype=tf.float32)
        
        self.freqs = tf.Variable(init, trainable=True, name='trainable_freqs')

    def call(self, t):
        # t: (batch,1)
        # compute scaled angle: (batch,num_fourier)
        angles = t * self.freqs[None,:] * (2 * np.pi / self.period)
        sin = tf.sin(angles)
        cos = tf.cos(angles)
        
        sin_cos = tf.stack([sin, cos], axis=2)
        
        # interleave sin and cos: stack then reshape
        interleaved = tf.reshape(sin_cos, (-1, 2*self.num_fourier))
        
        return interleaved

# === Residual Block ===
class ResidualBlock(keras.layers.Layer):
    
    def __init__(self, units, nlayers=2, w0=1.0):
        
        super().__init__()
        self.layers_ = [
            keras.layers.Dense(units,
                kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                bias_initializer=SIRENBiasInitializer())
            for _ in range(nlayers)
        ]
        self.act = SIRENActivation(w0)
        
    def call(self, x):
        
        h = x
        for layer in self.layers_:
            h = self.act(layer(h))
            
        return 0.5 * (x + h)

# === Multi-layer FiLM ShapeNet ===
class MultiShapeNet(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, w0=1.0, hidden_units=128):
        
        super().__init__()
        
        self.pre = keras.layers.Dense(hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(w0),
            bias_initializer=SIRENBiasInitializer(),
            activation=SIRENActivation(w0))
        
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0)
                       for _ in range(num_blocks)]

    def call(self, x, gammas, betas):
        # x: (batch,3) -> h: (batch,hidden_units)
        h = self.pre(x)
        
        for i, blk in enumerate(self.blocks):
            # apply per-block FiLM: gamma[:,i,:] and beta[:,i,:]
            h = gammas[:,:,i] * h + betas[:,:,i]
            h = blk(h)
            
        return h

# === ParameterNet with single film_layer ===
class MultiParameterNet(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, hidden_units=128,
                 num_fourier=16, w0=1.0, period=24.0):
        
        super().__init__()
        # Fourier embed -> project to hidden_units
        self.fourier = FourierFeatureEmbedding(num_fourier, w0, period)
        
        self.pre = keras.layers.Dense(hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(w0),
            bias_initializer=SIRENBiasInitializer(),
            activation=SIRENActivation(w0))
        
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0)
                       for _ in range(num_blocks)]
        # film_layer projects final hidden state to gamma+beta
        self.film_layer = keras.layers.Dense(2 * num_blocks * hidden_units,
            kernel_initializer='glorot_normal', bias_initializer='zeros')
        
        self.num_blocks = num_blocks
        
        self.hidden_units = hidden_units

    def call(self, t):
        # t: (batch,1)
        h = self.fourier(t)
        h = self.pre(h)
        
        for blk in self.blocks:
            h = blk(h)
            
        # produce combined gamma & beta
        gb = self.film_layer(h)  # (batch, 2*num_blocks*hidden_units)
        gb = tf.reshape(gb, (-1, self.hidden_units, self.num_blocks, 2))
        
        gamma = gb[:,:,:,0]
        beta  = gb[:,:,:,1]
        
        return gamma, beta

# === Combined MultiFiLMNet ===
class MultiFiLMNet(keras.Model):
    
    def __init__(self, num_blocks=4, nlayers=2, hidden_units=128, noutputs=3,
                 num_fourier=16, w0=30.0, period=24.0):
        
        super().__init__()
        
        self.shape_net = MultiShapeNet(num_blocks, nlayers, w0, hidden_units)
        
        self.param_net = MultiParameterNet(num_blocks, nlayers,
            hidden_units, num_fourier, w0, period)
        
        self.final = keras.layers.Dense(noutputs)
        
        self.scaler = Scaler()

    def call(self, inputs):
        # inputs: (batch,4) = [t, x, y, z]
        t = inputs[:, :1]
        xyz = inputs[:, 1:4]
        
        gamma, beta = self.param_net(t)
        
        h = self.shape_net(xyz, gamma, beta)
        
        h = self.final(h)
        
        return self.scaler(h)
    
    