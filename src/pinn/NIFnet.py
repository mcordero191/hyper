'''
Created on 16 Apr 2025

@author: radar
'''
import numpy as np
import tensorflow as tf
import keras

from pinn.layers import Scaler
from pinn.networks import BaseModel

# === 1) SIREN activation without phase ===
class SIRENActivation(keras.layers.Layer):
    
    def __init__(self, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        
        return tf.sin(inputs)
        # inject high frequencies via sinh
        # return tf.sin(tf.sinh(self.w0 * inputs))


# === 2) SIREN-style initializers ===
class SIRENFirstLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0

    def __call__(self, shape, dtype=None):
        
        in_feats = shape[1]
        limit = tf.sqrt( 2.0 / in_feats) / self.w0
        # limit = tf.sqrt(6.0 / in_feats) / self.w0
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

class SIRENIntermediateLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0

    def __call__(self, shape, dtype=None):
        
        in_feats = shape[1]
        limit = tf.sqrt(6.0 / in_feats) / self.w0
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

class SIRENBiasInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0

    def __call__(self, shape, dtype=None):
        
        limit = np.pi / self.w0
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

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

class FiLMNet(BaseModel):
    
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

class GaussianRFF(keras.layers.Layer):
    """
    Normalized Random Fourier Features generated in build():
      φ(x) = sqrt(2/D) [sin(w_i·x + b_i), cos(w_i·x + b_i)] for i=1..D/2
    where w_i are random directions scaled by learnable σ, and b_i random phases.
    """
    def __init__(self, out_dim: int = 256, init_log_sigma: float = 0.0, **kwargs):
        
        super().__init__(**kwargs)
        
        # assert out_dim % 2 == 0, "out_dim must be even"
        # D = half of final features (sin+cos)
        self.D = out_dim #// 2
        
        self.init_log_sigma = init_log_sigma
        # placeholders for weights
        self.w_dir = None
        self.b = None
        self.log_sigma = None

    def build(self, input_shape):
        # input_shape: (batch_size, d)
        d = input_shape[-1]
        # initialize random directions w_dir: shape (d, D)
        w0 = np.random.randn(d, self.D).astype(np.float32)
        
        self.w_dir = self.add_weight(
            name='w_dir',
            shape=(d, self.D),
            initializer=tf.constant_initializer(w0),
            trainable=False
        )
        # random phase offsets b: shape (D,)
        b0 = np.random.uniform(0, 2*np.pi, size=(self.D,)).astype(np.float32)
        
        self.b = self.add_weight(
            name='b',
            shape=(self.D,),
            initializer=tf.constant_initializer(b0),
            trainable=False
        )
        # learnable log sigma scalar
        self.log_alpha = self.add_weight(
            name='log_sigma',
            shape=(1,),
            initializer=tf.constant_initializer(self.init_log_sigma),
            trainable=True
        )
        
        # normalize to unit variance
        self.coef = np.sqrt(2.0 / self.D)
        
        super().build(input_shape)

    def call(self, inputs):
        # xyz: (batch, d)
        # compute scaled directions
        
        alpha = tf.exp(self.log_alpha)
        
        w = alpha * self.w_dir   # (d, D)
        
        # project input
        proj = tf.matmul(inputs, w) + self.b  # (batch, D)
        
        return self.coef * tf.sin(proj)  # (batch, 2D)
    
# === 3) Residual block with full skip-connection ===
class ResidualBlock(keras.layers.Layer):
    
    def __init__(self, units, nlayers=2, w0=1.0):
        
        super().__init__()
        
        self.nlayers = nlayers
        self.act = SIRENActivation(w0)
        
        self.dense_layers = [
            keras.layers.Dense(
                units,
                kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                bias_initializer="zeros",
            )
            for _ in range(nlayers)
        ]

    def build(self, input_shape):
        # input_shape: (batch_size, d)
        d = input_shape[-1]
        
        # learnable log sigma scalar
        self.log_alpha = self.add_weight(
            name='log_sigma',
            shape=(self.nlayers, ),
            initializer=tf.constant_initializer(0.0),
            trainable=True
        )
        
        super().build(input_shape)
        
    def call(self, inputs):
        
        alpha = tf.exp(self.log_alpha)
        
        h = inputs
        
        for i, layer in enumerate(self.dense_layers):
            h = self.act(layer(alpha[i]*h))
            
        return (inputs + h)

# === Multi-layer FiLM ShapeNet ===
class MultiShapeNet(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, w0=1.0, hidden_units=128):
        
        super().__init__()
        
        # self.fourier = keras.layers.Dense(hidden_units,
        #                             kernel_initializer=SIRENFirstLayerInitializer(w0),
        #                             bias_initializer=SIRENBiasInitializer(w0),
        #                             activation=SIRENActivation(w0),
        #                             )
        
        self.fourier = GaussianRFF(hidden_units)
        # self.fourier = SpatialFourierFeatureEmbedding(hidden_units//6)#num_fourier, period)
        
        # self.mix = keras.layers.Dense(hidden_units,
        #                             kernel_initializer=SIRENIntermediateLayerInitializer(w0),
        #                             bias_initializer=SIRENBiasInitializer(w0),
        #                             activation=SIRENActivation(w0),
        #                             )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0)
                       for _ in range(num_blocks)]

    def call(self, x, gammas, betas):
        # x: (batch,3) -> h: (batch,hidden_units)
        
        h = self.fourier(x)
        # h = self.mix(h)
        
        for i, blk in enumerate(self.blocks):
            # apply per-block FiLM: gamma[:,:,i] and beta[:,:,i]
            h = blk(h)
            h = gammas[:,:,i] * h + betas[:,:,i]
            
        return h

# === ParameterNet with single film_layer ===
class MultiParameterNet(keras.layers.Layer):
    
    def __init__(self, num_blocks, nlayers=2, hidden_units=128,
                 num_fourier=64, w0=1.0, period=2.0):
        
        super().__init__()
        # Fourier embed -> project to hidden_units
        # self.pre = FourierFeatureEmbedding(num_fourier=hidden_units//2)#num_fourier, period)
        
        # self.pre = keras.layers.Dense(hidden_units,
        #                             kernel_initializer=SIRENFirstLayerInitializer(w0),
        #                             bias_initializer=SIRENBiasInitializer(w0),
        #                             activation=SIRENActivation(w0),
        #                             )
        
        self.fourier = GaussianRFF(hidden_units//4)
        
        self.blocks = [ResidualBlock(hidden_units//4, nlayers, w0)
                       for _ in range(num_blocks)]
        
        # film_layer projects final hidden state to gamma+beta
        self.film_layer = keras.layers.Dense(2 * num_blocks * hidden_units,
                                            kernel_initializer="zeros", 
                                            bias_initializer=self._make_film_bias_initializer(num_blocks, hidden_units),
                                            )
        
        self.num_blocks = num_blocks
        
        self.hidden_units = hidden_units
        
    @staticmethod
    def _make_film_bias_initializer(num_blocks, hidden_units):
        # first half = gamma biases, second half = beta biases
        def init(shape, dtype=None):
            # shape = (2*num_blocks*hidden_units,)
            b = np.zeros(shape, dtype=np.float32)/hidden_units
            # set gamma_b’s to 1
            b[: num_blocks * hidden_units] = 1.0
            
            return tf.constant(b, dtype=dtype)
        return init
    
    def call(self, t):
        # t: (batch,1)
        h = self.fourier(t)
        
        for blk in self.blocks:
            h = blk(h)
            
        # produce combined gamma & beta
        gb = self.film_layer(h)  # (batch, 2*num_blocks*hidden_units)
        gb = tf.reshape(gb, (-1, self.hidden_units, self.num_blocks, 2))
        
        gamma = gb[:,:,:,0]
        beta  = gb[:,:,:,1]
        
        return gamma, beta

# === Combined MultiFiLMNet ===
class MultiFiLMNet(BaseModel):
    
    def __init__(self, num_blocks=4, nlayers=2, hidden_units=128, noutputs=3,
                 num_fourier=64, w0=1.0, period=None,
                 output_scaling=[1e0,1e0,1e-2],
                 ):
        
        super().__init__()
        
        if num_fourier is None:
            num_fourier = hidden_units//2
            
        if period is None:
            period = num_fourier
            
        self.shape_net = MultiShapeNet(num_blocks, nlayers, w0, hidden_units)
        
        self.param_net = MultiParameterNet(num_blocks, nlayers,
            hidden_units, num_fourier, w0, period)
        
        self.final = keras.layers.Dense(noutputs)
        
        self.scaler = Scaler(values=output_scaling)

    def call(self, inputs):
        # inputs: (batch,4) = [t, x, y, z]
        t = inputs[:, :1]
        xyz = inputs[:, 1:4]
        
        gamma, beta = self.param_net(t)
        
        h = self.shape_net(xyz, gamma, beta)
        
        h = self.final(h)
        
        return self.scaler(h)
    
# === Combined MultiFiLM Potential Net ===
class MultiFiLMPotentialNet(BaseModel):
    
    def __init__(self, num_blocks=4, nlayers=2, hidden_units=128,
                 num_fourier=64, w0=1.0, period=24.0, use_helmholtz=True,
                 noutputs=3):
        
        super().__init__()
        
        self.use_helmholtz = use_helmholtz
        
        self.shape_net = MultiShapeNet(num_blocks, nlayers, w0, hidden_units)
        
        self.param_net = MultiParameterNet(
            num_blocks, nlayers, hidden_units, num_fourier, w0, period)
        
        if use_helmholtz:
            self.vec_head = keras.layers.Dense(3, name='vector_potential', use_bias=False)
            self.scalar_head = keras.layers.Dense(1, name='scalar_potential', use_bias=False)
            output_scaling=[1e0,1e0,1e0,1e0]
            add_bias = False
        else:
            self.velocity_head = keras.layers.Dense(noutputs, name='direct_velocity', use_bias=True)
            output_scaling=[1e0,1e0,1e-2,1e0]
            add_bias = True
        
        self.scaler = Scaler(output_scaling, add_bias=add_bias)
        
    def call(self, inputs):
        
        t = inputs[:, :1]
        xyz = inputs[:, 1:4]
        
        if self.use_helmholtz:
            
            u = self.compute_velocity_helmholtz(t, xyz)
        else:
            gamma, beta = self.param_net(t)
        
            h = self.shape_net(xyz, gamma, beta)
        
            u = self.velocity_head(h)
            
        return self.scaler(u) 

    def compute_velocity_helmholtz(self, t, xyz):
        """
        Compute u = curl(A) + grad(phi) given A (batch,3), phi (batch,1), and x (batch,3).
        """
        
        z = xyz[:, 1:2]
        x = xyz[:, 2:3]
        y = xyz[:, 3:4]
        
        gamma, beta = self.param_net(t)
        
        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            
            h = self.shape_net(tf.concat([z, x, y], axis=1), gamma, beta)
            
            A = self.vec_head(h)
            phi = self.scalar_head(h)
            
            A1, A2, A3 = A[:,0:1], A[:,1:2], A[:,2:3]
            
        A1_y = tape.gradient(A1, y)
        A1_z = tape.gradient(A1, z)
        
        A2_x = tape.gradient(A2, x)
        A2_z = tape.gradient(A2, z)
        
        A3_x = tape.gradient(A3, x)
        A3_y = tape.gradient(A3, y)
        
        # curl
        u_rot = A3_y - A2_z
        v_rot = A1_z - A3_x
        w_rot = A2_x - A1_y
        
        u_div = tape.gradient(phi, x)
        v_div = tape.gradient(phi, y)
        w_div = tape.gradient(phi, z)
        
        u = tf.concat([u_rot + u_div, v_rot + v_div, w_rot + w_div], axis=1)
        
        del tape
        
        return u
