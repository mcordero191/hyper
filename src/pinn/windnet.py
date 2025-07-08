'''
Created on 28 May 2025

@author: radar
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

from pinn.layers import Scaler
from pinn.networks import BaseModel
from pinn.NIFnet import FiLMNet

def get_activation(name="tanh", w0=1.0):
    if name == "tanh":
        return tf.math.tanh
    elif name == "swish":
        return tf.nn.swish
    elif name == "gelu":
        return tf.nn.gelu
    elif name == "siren":
        return SIRENActivation(w0)
    elif name == "relu":
        return tf.nn.relu
    else:
        raise ValueError(f"Unsupported activation: {name}")
    
class NormalizationLayer(keras.layers.Layer):
    
    def __init__(self, method='rms', epsilon=1e-8, **kwargs):
        """
        Normalizes a tensor along its last dimension.

        Parameters
        ----------
        method : str
            One of 'rms', 'l2', or 'max'.
        epsilon : float
            Small constant to avoid division by zero.
        """
        super().__init__(**kwargs)
        self.method = method
        self.epsilon = epsilon

    def call(self, x):
        if self.method == 'rms':
            norm = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon)
        elif self.method == 'l2':
            norm = tf.norm(x, ord='euclidean', axis=-1, keepdims=True) + self.epsilon
        elif self.method == 'max':
            norm = tf.reduce_max(tf.abs(x), axis=-1, keepdims=True) + self.epsilon
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")
        return x / norm
    
# === SIREN Activation and Initializers ===
class SIRENActivation(keras.layers.Layer):
    
    def __init__(self, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        self.w0 = w0
        self.initial_log_w0 = np.log(w0)
    
    def build(self, input_shape):
    
        self.log_sigma = self.add_weight(
            name="log_sigma",
            shape=(),
            initializer=tf.constant_initializer(self.initial_log_w0),
            trainable=True,
        )
    
        super().build(input_shape)
    
    def call(self, x):
        
        w0 = tf.exp(self.log_sigma)
        
        return tf.sin(w0 * x)

class SIRENFirstLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0
        
    def __call__(self, shape, dtype=None):
        
        in_f = shape[0]
        limit = 1.0 / (in_f * self.w0)
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

class SIRENIntermediateLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=0.0):
        
        self.w0 = w0
        
    def __call__(self, shape, dtype=None):
        
        in_f = shape[1]
        limit = tf.sqrt(6.0 / in_f) / self.w0
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

class SIRENBiasInitializer(keras.initializers.Initializer):
    
    def __call__(self, shape, dtype=None):
        
        in_f = shape[0]
        limit = 1.0 / in_f
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

################################################################################
# TidalEmbedding: fixed periods, trainable spatial phase shifts only
################################################################################
class TidalEmbedding(keras.layers.Layer):
    """
    Computes P fixed‐period “tidal modes”:
       θ_i(t,z,x,y) = 2π*(t_hr/T_i)  +  2π*(φ_z,i * z_km)  +  2π*(φ_x,i * x_km)  +  2π*(φ_y,i * y_km).

    - T_i come from `initial_periods_hours` and remain fixed.
    - φ_z,i, φ_x,i, φ_y,i are trainable “cycles per km” for each mode.
    Returns: (sin(θ), cos(θ)) each of shape (batch,P).
    """
    def __init__(self, initial_periods_hours, **kwargs):
        
        super().__init__(**kwargs)
        
        self.P = len(initial_periods_hours)
        self.ini_periods = initial_periods_hours  # shape (P,)
        
        # placeholders for phase shifts:
        self.phase_z = None
        self.phase_x = None
        self.phase_y = None

    def build(self, input_shape):
        # trainable “cycles per km” for vertical/horiz shifts:
        
        self.periods = self.add_weight(
            name="periods",
            shape=(self.P,),
            initializer=tf.constant_initializer(self.ini_periods),
            trainable=False
        )
        
        self.phase_z = self.add_weight(
            name="phase_z",
            shape=(self.P,),
            initializer="zeros",
            trainable=True
        )
        
        self.phase_x = self.add_weight(
            name="phase_x",
            shape=(self.P,),
            initializer="zeros",
            trainable=True
        )
        
        self.phase_y = self.add_weight(
            name="phase_y",
            shape=(self.P,),
            initializer="zeros",
            trainable=True
        )
        
        super().build(input_shape)

    def call(self, inputs):
        """
        Inputs t,z,x,y each (batch,1) in [-1,1].
        Map to physical units:
           t_hr = (t+1)*12    ∈ [0,24]
           z_km = (z+1)*10    ∈ [0,20]
           x_km = (x+1)*150   ∈ [-150,150]
           y_km = (y+1)*150   ∈ [-150,150]

        θ_i = 2π*(t_hr/T_i) + 2π*(φ_z,i * z_km) + 2π*(φ_x,i * x_km) + 2π*(φ_y,i * y_km).
        Returns sin(θ), cos(θ), each (batch,P).
        """
        
        t = inputs[:, 0:1]
        z = inputs[:, 1:2]
        x = inputs[:, 2:3]
        y = inputs[:, 3:4]
        
        t_hr = (t + 1.0) * 12.0   # [0,24]
        z_km = (z + 1.0) * 10.0   # [0,20]
        x_km = (x + 1.0) * 150.0  # [-150,150]
        y_km = (y + 1.0) * 150.0  # [-150,150]

        # base part: 2π*(t_hr / T_i)
        theta = 2.0 * np.pi * (t_hr / self.periods)  # (batch,P) via broadcasting

        # add trainable spatial phases (each broadcast (batch,1)*(P,)→(batch,P))
        theta = theta + 2.0 * np.pi * (z_km * self.phase_z)
        theta = theta + 2.0 * np.pi * (x_km * self.phase_x)
        theta = theta + 2.0 * np.pi * (y_km * self.phase_y)

        tidal_embed = tf.concat([tf.sin(theta), tf.cos(theta)], axis=-1)  # (batch, 2P)
        
        return tidal_embed   # each (batch,2P)

################################################################################
#  GaussianRFF for “slow” features (Rahimi & Recht 2007)
################################################################################
class GaussianRFF(keras.layers.Layer):
    """
    Gaussian Random Fourier Features:
      φ(x) = sqrt(2/D) * [ sin(σ·w_i · x + b_i), cos(σ·w_i · x + b_i ) ]_{i=1..D},
    where w_i ~ N(0,1), b_i ~ Uniform[0,2π], and σ = exp(log_sigma) controls bandwidth.
    We initialize log_sigma to a large negative value so that φ only captures very low frequencies.
    """
    def __init__(self, out_dim: int = 64, init_log_sigma: float = -2, **kwargs):
        
        super().__init__(**kwargs)
        assert out_dim % 2 == 0, "out_dim must be even"
        
        self.D = out_dim // 2
        self.init_log_sigma = init_log_sigma

    def build(self, input_shape):
        # input_shape = (batch_size, d), here d = 4 (t,z,x,y)
        d = input_shape[-1]
        # 1) w_fixed ~ N(0,1) of shape (d, D)
        ini_w = np.random.randn(d, self.D).astype(np.float32)
        
        self.w_fixed = self.add_weight(
            name="w",
            shape=(d, self.D),
            initializer=tf.constant_initializer(ini_w),
            trainable=True
        )
        
        # 2) b ~ Uniform[0,2π] of shape (D,)
        b0 = np.random.uniform(0, 2*np.pi, size=(self.D,)).astype(np.float32)
        self.b = self.add_weight(
            name="b",
            shape=(self.D,),
            initializer=tf.constant_initializer(b0),
            trainable=True
        )
        
        # 3) trainable log_sigma (scalar), init to a large negative so RFF focuses on very low freq
        self.log_sigma = self.add_weight(
            name="log_sigma",
            shape=(),
            initializer=tf.constant_initializer(self.init_log_sigma),
            # constraint = tf.keras.constraints.NonNeg(),
            trainable=True
        )
        
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch,4) = [t,z,x,y], each ∈[-1,1]
        sigma = tf.exp(self.log_sigma)    # scalar > 0
        
        proj = sigma * tf.matmul(inputs, self.w_fixed) + self.b  # (batch, D)
        
        sin_feats = tf.sin(proj)          # (batch, D)
        cos_feats = tf.cos(proj)          # (batch, D)
        # Concatenate → (batch, 2D)
        return tf.concat([sin_feats, cos_feats], axis=-1)
    
class LowFreqNet(keras.layers.Layer):
    """
    A small MLP that sees only low‐frequency tidal modes from (t,z,x,y).
    Returns u_lf ∈ R^{noutputs}.
    """
    def __init__(self,
                 noutputs=3,
                 hidden_units=32,
                 nlayers=2,
                 nblocks=3,
                 # periods=[24.0, 12.0, 8.0, 6.0, 4.8],
                 w0=1.0,
                init_log_sigma=-1.0,
                activation="tanh",
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.noutputs = noutputs
        # self.P = len(periods)

        # 1) Use the revised TidalEmbedding:
        # self.tidal = TidalEmbedding(periods)

        self.slow_rff = GaussianRFF(out_dim=hidden_units, init_log_sigma=init_log_sigma)
        # self.slow_rff = PositionalEncoding(num_bands=8)
        
        # 2) Simple MLP on those 2P features:
        # self.pre = keras.layers.Dense(
        #     hidden_units,
        #     activation=SIRENActivation(w0),
        #     kernel_initializer=SIRENIntermediateLayerInitializer(w0),
        #     bias_initializer="zeros"
        # )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers=nlayers, w0=w0, activation=activation) for _ in range(nblocks)]
        
        # self.blocks = [
        #     keras.layers.Dense(
        #         hidden_units,
        #         activation=SIRENActivation(w0),
        #         kernel_initializer=SIRENIntermediateLayerInitializer(w0),
        #         bias_initializer="zeros"
        #     ) for _ in range(nlayers)
        # ]
        
        self.out_layer = keras.layers.Dense(
            noutputs,
            kernel_initializer="GlorotNormal",
            bias_initializer="zeros"
        )

    def call(self, inputs):
        
        # tidal_embed = self.tidal(inputs)  # each (batch,2P)

        h = self.slow_rff(inputs)
        
        # h = self.pre(rff_embed)  # (batch, hidden_units)
        
        for layer in self.blocks:
            h = layer(h)
            
        u = self.out_layer(h)  # (batch, noutputs)
        
        return(u, h)
    
class PositionalEncoding(keras.layers.Layer):
    """
    Standard positional encoding: for each coordinate in R^d, we map
      coord → [sin(2^k * π * coord), cos(2^k * π * coord)]_{k=0..L-1},
    so total embedding dimension = d * 2L.
    
    Here we use L frequencies for each of the 4 inputs (t,z,x,y).
    """
    def __init__(self, num_bands=8, **kwargs):
        
        super().__init__(**kwargs)
        
        self.num_bands = num_bands
        # We'll build nothing trainable—always fixed
        bands = 2 ** np.arange(self.num_bands)/2**(4)
        
        self.bands = np.reshape( bands*np.pi, (1, 1, self.num_bands) )
        
    def call(self, inputs):
        """
        inputs: (batch, d)  where d = 4 (t,z,x,y), each in [-1,1].
        Output: (batch, d * 2 * num_bands).
        """
        batch = tf.shape(inputs)[0]
        d = tf.shape(inputs)[1]
        
        # Expand inputs (batch, d) → (batch, d, 1)
        # Multiply by π and each band: (batch, 1, num_bands)
        args = inputs[:,:,tf.newaxis] * self.bands

        # sin and cos: each (batch, d, num_bands)
        sin = tf.sin(args)
        cos = tf.cos(args)

        # Concatenate along last axis → (batch, d, 2*num_bands), then flatten to (batch, d*2*num_bands)
        pe = tf.reshape(tf.concat([sin, cos], axis=-1), (batch, d * 2 * self.num_bands))
        
        return pe  # (batch, 4 * 2 * num_bands)
    
## 2) HighFreqNet now takes both [t,z,x,y] and f_low_hidden, concatenated
#    so that it “knows” what the low-frequency net already predicted.
class HighFreqNet(keras.layers.Layer):
    
    def __init__(self,
                 noutputs=3,
                 hidden_units=64,
                 nlayers=3,
                 nblocks=3,
                 pe_bands=6,
                 w0=1.0,
                 activation="tanh",
                 # low_feat_dim=32,
                init_log_sigma=2.0,
                 **kwargs):
        """
        High-frequency “gravity waves / turbulence” learner.  
        We will concatenate the low-frequency features (dim=low_feat_dim) to the positional encoding
        of (t,z,x,y), so that the HF module can correct whatever was learned by the LF module.
        """
        super().__init__(**kwargs)
        self.noutputs = noutputs
        # self.pe_bands = pe_bands
        
        # 1) Standard positional encoding of (t,z,x,y) → (batch, 4*2*pe_bands)
        # self.four_feat = PositionalEncoding(num_bands=pe_bands)
        
        self.four_feat = GaussianRFF(out_dim=hidden_units, init_log_sigma=init_log_sigma) 
        
        # self.four_feat = keras.layers.Dense(
        #     hidden_units,
        #     activation=SIRENActivation(w0),
        #     kernel_initializer=SIRENFirstLayerInitializer(w0),
        #     bias_initializer=SIRENBiasInitializer()
        # )
        
        # self.pre = keras.layers.Dense(
        #     hidden_units,
        #     activation=SIRENActivation(w0),
        #     kernel_initializer=SIRENIntermediateLayerInitializer(w0),
        #     bias_initializer=SIRENBiasInitializer()
        # )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers=nlayers, w0=w0, activation=activation) for _ in range(nblocks)]
        
        # # 3) A few residual SIREN blocks
        # self.blocks = [
        #     keras.layers.Dense(
        #         hidden_units,
        #         activation=SIRENActivation(w0),
        #         kernel_initializer=SIRENIntermediateLayerInitializer(w0),
        #         bias_initializer="zeros"
        #     )
        #     for _ in range(nblocks)
        # ]

        # 4) Final out → produce u_high (batch, noutputs)
        self.out_layer = keras.layers.Dense(
            noutputs,
            kernel_initializer="GlorotNormal",
            bias_initializer="zeros"
        )
        
        # self.ena_hnet = self.add_weight(
        #     name="ena_hnet",
        #     shape=(),
        #     initializer='zeros',
        #     trainable=False,
        # )

    def call(self, inputs, f_low_hidden=None):
        """
        inputs        : (batch,4) = [t,z,x,y]
        f_low_hidden  : (batch, low_feat_dim)   from LowFreqNet

        We do:
          pe_feats = PE(t,z,x,y)             shape = (batch, pe_dim)
          combine  = tf.concat([pe_feats, f_low_hidden], axis=-1)   shape=(batch, pe_dim+low_feat_dim)
          pass → SIREN residual → out_layer → u_high
        """
        h = self.four_feat(inputs)                    # (batch, pe_dim)
        
        if f_low_hidden is not None:
            h = tf.concat([h, f_low_hidden], axis=-1)  # (batch, pe_dim + low_feat_dim)

        # 1) project down to hidden_units
        # h = self.pre(h)                               # (batch, hidden_units)

        # 2) residual SIREN blocks
        for layer in self.blocks:
            h = layer(h)                 # (batch, hidden_units)

        # 3) final output
        u = self.out_layer(h)                    # (batch, noutputs)
        
        return u, h

# === Residual Block ===
class ResidualBlock(keras.layers.Layer):
    
    def __init__(self, units, nlayers=2, w0=1.0, activation="siren", kernel_initializer="GlorotNormal"):
        
        super().__init__()
        
        self.activation = get_activation(activation, w0)
        self.units = units
        
        inner_layers = []
        norm_layers = []
        
        for _ in range(nlayers):
            
            layer = keras.layers.Dense(
                                        units,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer="zeros",
                                        # activation=SIRENActivation(w0),
                                    ) 
            
            inner_layers.append(layer)
            
            layer = keras.layers.LayerNormalization()
            norm_layers.append(layer)
        
        self.inner_layers = inner_layers
        self.norm_layers = norm_layers
    
        
        self.skip = keras.layers.Dense(
                                            self.units,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer="zeros",
                                            # activation=SIRENActivation(w0),
                                        ) 
        
    # def build(self, input_shape):
    #
    #     in_shape = input_shape[-1]
    #
    #     if True: #in_shape != self.units:
    #         self.proj = keras.layers.Dense(
    #                                         self.units,
    #                                         kernel_initializer="HeNormal",
    #                                         bias_initializer="zeros",
    #                                         # activation=SIRENActivation(w0),
    #                                     ) 
    #     # else:
    #     #     self.proj = keras.layers.Identity()
    #     #
    #
    #     super().build(input_shape) 
        
    def call(self, x):
        
        h = x
        
        for i, layer in enumerate( self.inner_layers[:-1] ):
            h = layer(h)
            h = self.activation(h)
            h = self.norm_layers[i](h)
            
        h = self.inner_layers[-1](h)
        xp = self.skip(x)
        
        h = self.activation(h+xp)
        h = self.norm_layers[-1](h)
        
        return h #0.5*(x + h)

# === BranchNet: accepts (t,) or (t,z) as input ===
class BranchNet(keras.layers.Layer):
    
    def __init__(self, noutputs, num_basis, hidden_units=64, nlayers=2, nblocks=2,
                 w0=1.0,
                 init_log_sigma=0.0,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.noutputs = noutputs
        self.num_basis = num_basis
        self.w0 = w0
        
        # self.fourier = GaussianRFF(out_dim=hidden_units, init_log_sigma=init_log_sigma)
        
        # Pre-layer projects time (and optionally z) to hidden_units
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENFirstLayerInitializer(w0),
            bias_initializer=SIRENBiasInitializer(),
        )
        
        self.mix = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENIntermediateLayerInitializer(w0),
            bias_initializer=SIRENBiasInitializer(),
        )
        
        # Stack of residual blocks
        self.blocks = [ResidualBlock(hidden_units, nlayers=nlayers, w0=w0) for _ in range(nblocks)]
        
        # self.mix = GaussianRFF(out_dim=hidden_units, w0=w0)
        
        # Output layer to produce coefficients for each output and basis
        self.coeff = keras.layers.Dense(
            noutputs * num_basis,
            kernel_initializer='zeros',
            bias_initializer='zeros'
        )
        
    def call(self, inputs, f_low_hidden=None):
        
        # pass through pre-layer
        h = self.pre(inputs)                    # (batch, pe_dim)
        
        if f_low_hidden is not None:
            h = tf.concat([h, f_low_hidden], axis=-1)  # (batch, pe_dim + low_feat_dim)

        # 1) project down to hidden_units
        h = self.mix(h)   
        
        # residual stack
        for blk in self.blocks:
            h = blk(h)
            
        # h = self.mix(h)
        
        # produce coefficients
        c = self.coeff(h)  # (batch, noutputs * num_basis)
        
        # reshape to (batch, noutputs, num_basis)
        c = tf.reshape(c, (-1, self.noutputs, self.num_basis))
        
        return c  # shape (batch, noutputs, num_basis)

# === TrunkNet: accepts (x,y) or (x,y,z) as input ===
class TrunkNet(keras.layers.Layer):
    
    def __init__(self, num_basis, hidden_units=64, nlayers=2, nblocks=3,
                 w0=1.0,
                 init_log_sigma=0.0,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.num_basis = num_basis
        self.w0 = w0
        
        # self.fourier = GaussianRFF(out_dim=hidden_units, init_log_sigma=init_log_sigma)
        
        # Pre-layer projects spatial coords to hidden_units
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENFirstLayerInitializer(w0),
            bias_initializer=SIRENBiasInitializer(),
        )
        
        # Residual stack
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0) for _ in range(nblocks)]
        
        # Output layer to produce basis values
        self.out = keras.layers.Dense(
            num_basis,
            kernel_initializer='GlorotNormal',
            bias_initializer='zeros'
        )
        
        
    def call(self, inputs, f_low_hidden=None):
        
        # pre-layer
        # h = self.fourier(inputs)
        #
        # if f_low_hidden is not None:
        #     h = tf.concat([h, f_low_hidden], axis=-1)  # (batch, pe_dim + low_feat_dim)
            
        h = self.pre(inputs)
        
        # residual stack
        for blk in self.blocks:
            h = blk(h)
        
        # h = self.sc(h)
        # produce basis: shape (batch, num_basis)
        phi = self.out(h)
        
        return phi  # (batch, num_basis)

# === DeepONet: combines BranchNet and TrunkNet ===
class DeepONetHF(keras.layers.Layer):
    
    def __init__(self,
                 noutputs,
                 hidden=64,
                 nlayers=2,
                 nblocks=3,
                 # trunk_hidden=64,
                 # trunk_layers=2,
                 num_basis=256,
                 w0=1.0,
                 init_log_sigma=0.0,
                 # nlow_feat=0,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.num_basis = num_basis
        
        self.branch = BranchNet(
            noutputs=noutputs,
            num_basis=num_basis,
            hidden_units=hidden,
            nlayers=nlayers,
            nblocks=nblocks,
            w0=w0,
            # nlow_feat=nlow_feat,
            # init_log_sigma=init_log_sigma,
        )
        
        self.trunk = TrunkNet(
            num_basis=num_basis,
            hidden_units=hidden,
            nlayers=nlayers,
            nblocks=nblocks,
            w0=w0,
            init_log_sigma=init_log_sigma,
        )
        
    def call(self, inputs, f_low_hidden=None):
        # inputs: (batch, 4) = [t, z, x, y]
        # split into (t,z) for branch and (x,y) or (x,y,z) for trunk
        # If you have all 4 dims, branch gets (t,z), trunk gets (x,y)
        
        input_b = inputs[:, 0:2]   # (batch,2)
        input_t = inputs[:, 2:4]  # (batch,2)
        
        # call branch on (t,z), trunk on (x,y)
        a = self.branch(input_b, f_low_hidden)   # (batch, noutputs, num_basis)
        phi = self.trunk(input_t)  # (batch, num_basis)
        
        # Now combine: for each output c, do dot(a[c,:], phi)
        # a has shape (batch, noutputs, num_basis), phi is (batch, num_basis)
        y = tf.reduce_mean(a * phi[:,tf.newaxis,:], axis=-1)  # (batch, noutputs)
        
        return y

# === MultiScaleDeepONetHF ===
class MultiScaleDeepONetHF(keras.layers.Layer):
    
    def __init__(self, noutputs, num_basis=192, scales=[[0,32],[32,96],[96,192]],
                 hidden=64, 
                 nlayers =1,
                 nblocks = 3,
                 w0 = 1.0,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.noutputs = noutputs
        self.num_basis = num_basis
        self.scales = scales  # list of (start, end) index tuples
        self.num_scales = len(scales)

        # Shared trunk and branch
        self.trunk = TrunkNet(num_basis=num_basis, hidden_units=hidden, nlayers=nlayers, nblocks=nblocks, w0=w0)
        self.branch = BranchNet(noutputs=noutputs, num_basis=num_basis, hidden_units=hidden, nlayers=nlayers, nblocks=nblocks, w0=w0)

        # Learnable gates for each scale
        self.alphas = self.add_weight(name="alphas", shape=(self.num_scales,),
                                      initializer="zeros", trainable=True)

    def call(self, inputs, f_low_hidden=None):
        
        t_z = inputs[:, :2]
        x_y = inputs[:, 2:4]

        a = self.branch(t_z, f_low_hidden=f_low_hidden)          # (batch, noutputs, num_basis)
        phi = self.trunk(x_y)         # (batch, num_basis)

        outputs = []
        for i, (s, e) in enumerate(self.scales):
            a_scale = a[:, :, s:e]        # (batch, noutputs, basis_i)
            phi_scale = phi[:, s:e]       # (batch, basis_i)
            scale = 1.0/(e-s)
            out = scale*tf.reduce_sum(a_scale * phi_scale[:, None, :], axis=-1)  # (batch, noutputs)
            outputs.append(out)

        weights = tf.nn.softmax(self.alphas)  # (num_scales,)
        outputs_stack = tf.stack(outputs, axis=0)  # (num_scales, batch, noutputs)
        combined = tf.tensordot(weights, outputs_stack, axes=([0], [0]))  # (batch, noutputs)

        return combined

class FiLMLayer(keras.layers.Layer):
    
    def __init__(self, units, nlayers=1, nblocks=3, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        self.units = units
        self.activation = SIRENActivation(w0)
        
        # self.pre_spatial = keras.layers.Dense(
        #                                 units,
        #                                 # activation=SIRENActivation(w0),
        #                                 # kernel_initializer=SIRENIntermediateLayerInitializer(w0),
        #                                 bias_initializer="zeros"
        #                             )
        #
        # self.pre_context = keras.layers.Dense(
        #                                 units,
        #                                 # activation=SIRENActivation(w0),
        #                                 # kernel_initializer=SIRENIntermediateLayerInitializer(w0),
        #                                 bias_initializer="zeros"
        #                             )
        
        self.spatial_net = keras.layers.Dense( units,
                                               kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                                               bias_initializer=SIRENBiasInitializer()
                                               )
        
        self.gamma_net = keras.layers.Dense( units,
                                             kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                                             bias_initializer=SIRENBiasInitializer()
                                             )
        
        self.beta_net = keras.layers.Dense( units,
                                            kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                                            bias_initializer=SIRENBiasInitializer()
                                            )

    def call(self, inputs):
        
        spatial_input, context_input = inputs
        
        x = self.spatial_net(spatial_input)
        
        gamma = self.gamma_net(context_input)
        beta = self.beta_net(context_input)
        
        return self.activation(gamma * x + beta)

class PosEnc(keras.layers.Layer):
    
    def __init__(self, num_frequencies=8):
        
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freqs = 2.0 ** tf.range(num_frequencies, dtype=tf.float32)

    def call(self, x):
        
        n = tf.shape(x)[1]
        
        x = tf.expand_dims(x, -1) * self.freqs
        x = tf.reshape(x, (-1, n*self.num_frequencies))
        
        return tf.concat([tf.sin(x), tf.cos(x)], axis=-1)

class FiLM_MLP(keras.Model):
    
    def __init__(self, 
                 hidden_units=64, 
                 nlayers=1,
                 num_blocks=4, 
                 output_dim=3, 
                 w0=6.0, 
                 posenc=True,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.use_posenc = posenc
        if self.use_posenc:
            self.pos_enc = PosEnc(num_frequencies=8)

        self.film_layers = [FiLMLayer(hidden_units, nlayers=1, nblocks=nlayers, w0=w0) for _ in range(num_blocks)]
        self.out_carrier = keras.layers.Dense(output_dim, kernel_initializer="zeros")
    
        self.envelope_net = keras.Sequential([
                keras.layers.Dense(hidden_units, activation="tanh"),
                keras.layers.Dense(output_dim, activation="softplus")
            ])
        
    def call(self, inputs, low_feat=None):
        
        spatial_input = inputs[:, 0:4]  # x, y
        context_input = inputs[:, :2]   # t, z

        if low_feat is not None:
            context_input = tf.concat([context_input, low_feat], axis=1)
            
        if self.use_posenc:
            spatial_input = self.pos_enc(spatial_input)

        x = spatial_input
        out = []
        for film_layer in self.film_layers:
            x = film_layer([x, context_input])
            out.append(x)
            
        out = tf.concat(out, axis=1)
        
        carrier = self.out_carrier(out)
        envelope = self.envelope_net(out)
        
        return carrier * envelope
    
class WindNet(BaseModel):
    """
    Combined = LowFreqNet + HighFreqNet.
    Stage 1: freeze HighFreqNet → train only LowFreqNet (learn tides, planetary waves)
    Stage 2: unfreeze HighFreqNet → train both (capture gravity waves, turbulence).
    """
    def __init__(self,
                 noutputs=3,
                 hf_hidden=64,
                 hf_layers=1,
                 num_blocks=3,
                 lf_hidden=64,
                 lf_layers=3,
                 lf_nblocks=1,
                 num_basis=256,
                 w0=1.0,
                 ena_deeponet=0,
                 ena_film=0,
                 init_log_sigma=0.0,
                 n_subnets=2,
                 activation="swish",
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.low_net = LowFreqNet(
                                    noutputs=noutputs,
                                    hidden_units=lf_hidden,
                                    nlayers=lf_layers,
                                    nblocks=lf_nblocks,
                                    init_log_sigma=0.0,
                                    activation=activation,
                                    # w0=w0,
                                )
        
        self.n_hlayers = 1
        # 2) High‐frequency correction network
        if ena_deeponet:
            
            self.high_nets = [
                                DeepONetHF(
                                    noutputs=noutputs,
                                    hidden=hf_hidden,
                                    nlayers=hf_layers,
                                    nblocks=num_blocks,
                                    w0=w0,
                                    num_basis=num_basis,
                                    # init_log_sigma=init_log_sigma,
                                    # num_scales=3,
                                )
                                ]
        elif ena_film:
            self.high_nets = [
                                FiLMNet(
                                    hidden_units=hf_hidden,
                                    num_blocks=num_blocks,
                                    noutputs=noutputs,
                                    nlayers=hf_layers,
                                    w0=w0,
                                    num_basis=num_basis
                                )
                                ]
            
        else:
            
            blocks = []
            for i in range(n_subnets):
                hf_block = HighFreqNet(
                                            noutputs=noutputs,
                                            hidden_units=hf_hidden,
                                            nlayers=hf_layers,
                                            nblocks=num_blocks,
                                            w0=w0, #2*w0*(i+1),
                                            init_log_sigma=init_log_sigma,
                                            activation=activation,
                                        )
                blocks.append(hf_block)
        
            self.high_nets = blocks
            self.n_hlayers = n_subnets
        
        # self.norm = NormalizationLayer()
        self.scale = Scaler()

    def build(self, input_shape):
        
        super().build(input_shape) 
        
        self.gates = self.add_weight(
            name="gates",
            shape=(self.n_hlayers,),
            initializer="zeros",
            trainable=False,
        )
        
        self.step_count = tf.Variable(0.0, trainable=False)
        
    def call(self, inputs, return_hf=False):
        
                 
        gates = self.gates
        # u_lf_feat = None
        
        u_lf, u_lf_feat = self.low_net(inputs)   # (batch, noutputs)
        # u_lf = self.scale(u_lf)
        
        u_hf = 0
        for i, high_net in enumerate(self.high_nets):
            up, u_lf_feat = high_net(inputs, u_lf_feat)  # (batch, noutputs)
            up = up*gates[i]
            u_hf += up
        
        # u_hf =  self.scale(up)
        
        u = u_hf + u_lf
        
        u =  self.scale(u)
        
        if return_hf:
            return u, u_hf
        
        return u
    
    def update_mask(self, total_epochs):
    
        self.step_count.assign_add(1.0)
        counter = 2.0*( tf.range(self.n_hlayers, dtype=tf.float32) + 1.0 )
        
        # Smooth linear ramp: 0→1 between epochs 0.2*total → 1.0*total
        alpha = (self.step_count - 0.1*total_epochs*counter) / (0.1*total_epochs)
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
        self.gates.assign(alpha)
        
# -----------------------------------------------------------------------------
class ScaleMixer(keras.layers.Layer):
    
    def __init__(self,
                 features,
                 kernel_size=3,
                 dilation_rates=[1,2],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # 1D conv over the 'scale' axis:
        conv_blocks = []
        
        for rate in dilation_rates:
            conv = keras.layers.Conv1D(
                                        filters=features,
                                        kernel_size=kernel_size,
                                        padding='same',
                                        activation=None,
                                        dilation_rate=rate,
                                        kernel_initializer="glorot_normal",
                                    )
            conv_blocks.append(conv)
        
        self.conv_blocks = conv_blocks
        
        self.activation = keras.layers.Activation(tf.tanh)
        
        self.norm = keras.layers.LayerNormalization()
        
        # optional pointwise projection back to your output dim
        # self.project = keras.layers.Dense(features,
        #                                 kernel_initializer="GlorotNormal",
        #                                 bias_initializer="zeros"
        #                                 )
        

    def call(self, x):
        # x: (batch, nstages, features)
        
        mixed = 0
        for conv in self.conv_blocks:
            mixed += conv(x)
            
        y = self.activation(mixed)
        
        # y = tf.reduce_mean(y, axis=1) 
        
        # y = self.project(y)
        
        y = self.norm(y)
        
        return y  # collapse scale axis
    
    
# 1) A single “ScaleNet” that focuses on one band of frequencies
class ScaleNet(keras.layers.Layer):
    """
    A little RFF+SIREN‐MLP that captures one band of scales.
      - init_log_sigma controls the center‐frequency of the RFF.
      - w0 controls the SIREN nonlinearity in the MLP.
    """
    def __init__(self,
                 noutputs: int,
                 hidden_units: int,
                 nlayers: int,
                 nblocks: int,          
                 init_log_sigma: float, #controls the center‐frequency of the RFF.
                 w0: float,             #controls the SIREN nonlinearity in the MLP.
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.rff = GaussianRFF(out_dim=hidden_units, init_log_sigma=init_log_sigma)
        
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENIntermediateLayerInitializer(w0),
            bias_initializer="zeros"
        )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0) for _ in range(nblocks)]
        
        self.norm = keras.layers.LayerNormalization()
        
        # self.out = keras.layers.Dense(
        #     hidden_units,
        #     kernel_initializer="GlorotNormal",
        #     bias_initializer="zeros"
        # )

    def call(self, inputs, low_feat=None):
        
        h = self.rff(inputs)
        
        if low_feat is not None:
            h = tf.concat([h, low_feat], axis=-1)  # (batch, pe_dim + low_feat_dim)
        
        h = self.pre(h)
        
        for blk in self.blocks:
            h = blk(h)
        
        # h = self.out(h)
        
        h = self.norm(h)
        
        return h  # (batch, noutputs)

# 2) Multi‐Scale PINN Model
class MultiScaleWindNet(BaseModel):
    """
    Sum of multiple ScaleNets, each gated on at a different training stage.
    """
    def __init__(self,
                 noutputs=3,
                 scale_params=[     # list of (init_log_sigma, w0) for each stage
                    (-4.0, 1e-2),  # very low freq
                     (-2.0, 1e-1),  # mid-low
                     ( 0.0, 1e0),  # mid
                     ( 2.0, 1e1),  # mid-high
                     ( 4.0, 1e1),  # very high freq
                     # ( 8.0, 6.0),  # ultra high freq
                    # ( 16.0, 6.0),  # extremely high freq
                 ],
                 hidden_units=64,
                 nlayers=1,
                 nblocks=3,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.noutputs = noutputs
        self.nstages   = len(scale_params)
        self.hidden_units = hidden_units

        # build one ScaleNet per frequency‐band
        self.scale_nets = []
        
        for (init_log_sigma, w0) in scale_params:
            
            net = ScaleNet(noutputs, hidden_units, nlayers, nblocks=nblocks,
                           init_log_sigma=init_log_sigma,
                           w0=w0)
            
            self.scale_nets.append(net)
        
        self.mixer = ScaleMixer(hidden_units, kernel_size=3)
        
        # self.out = keras.layers.Dense(
        #     noutputs,
        #     kernel_initializer="GlorotNormal",
        #     bias_initializer="zeros"
        # )
        
        self.norm = keras.layers.LayerNormalization()
        
        blocks = [ResidualBlock(hidden_units, nlayers=nlayers) for _ in range(nblocks)]
        
        self.res_block = keras.Sequential(
                                    [
                                        keras.layers.Dense(hidden_units, kernel_initializer=SIRENIntermediateLayerInitializer(1.0), activation=SIRENActivation(1.0)),
                                        *blocks,
                                    ]
                                    )
        
        self.to_uvw = keras.layers.Dense(noutputs, kernel_initializer="zeros")
        
        self.scaler = Scaler()
        
        # 2) trainable “gate logits” → softmax → one weight per scale
        init = np.array([+5.0] + [-5.0]*(self.nstages-1), dtype=np.float32)
        
        self.gates = self.add_weight(
            name="gates", shape=(self.nstages,),
            initializer=tf.constant_initializer(init),
            trainable=True
        )
        
    def call(self, inputs):
        
        gates = tf.math.sigmoid(self.gates)
        
        # cascade through scales, each sees the previous residual
        low_feat = None
        residuals = []
        
        for i, net in enumerate(self.scale_nets):
            
            h = net(inputs, low_feat=low_feat)   # (batch, hidden_units)
            # low_feat = h                # feed into next band
            residuals.append(gates[i]*h)

        # stack: (batch, nstages, hidden_units)
        # y = tf.stack(residuals, axis=1)
        # y = self.mixer(y) + y
        #
        # y = tf.reduce_mean(y, axis=1)
        
        y = tf.concat(residuals, axis=1)
        
        y = self.res_block(y) #+ y

        y = self.to_uvw(y)
        
        return self.scaler(y)

    def build(self, input_shape):
        
        super().build(input_shape) 
        
        self.step = tf.Variable(0.0, trainable=False)
        
    # def update_mask(self, total_steps):
    #
    #     self.step.assign_add(1.0)
    #
    #     seg = total_steps / float(self.nstages)
    #     s   = self.step
    #     ramp_w = seg * 0.2
    #
    #     gates = []
    #     # stage 0 always fully on
    #     gates.append(7.0)
    #
    #     # for i=1…nstages-1, ramp from 0→1 over steps [i*seg, (i+1)*seg]
    #     for i in range(1, self.nstages):
    #         start = seg * i
    #         # end = seg * (i+1)
    #         g = tf.clip_by_value((s - start) / ramp_w, -1.0, 1.0)
    #         gates.append( 7.0*g )
    #
    #     gates = tf.stack(gates, axis=0)  # shape (nstages,)
    #
    #     self.gates.assign(gates)
    
        
# -----------------------------------------------------------------------------
# === 1) Shared Backbone with Multiple RFF Banks ===
class SharedBackbone(keras.layers.Layer):
    
    def __init__(self,
                 hidden_units: int,
                 init_log_sigmas: list[float],
                 w0: float,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.nstages = len(init_log_sigmas)
        
        self.rffs = [ GaussianRFF(out_dim=hidden_units, init_log_sigma=lg)
                      for lg in init_log_sigmas ]
        
        self.proj = keras.layers.Dense(
            self.nstages*hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENIntermediateLayerInitializer(w0),
            bias_initializer="zeros"
        )
        
        # self.res  = ResidualBlock(hidden_units, nlayers=1, w0=w0)

    def call(self, inputs):
        
        feats = [rff(inputs) for rff in self.rffs]
        
        h = tf.concat(feats, axis=-1)
        
        h = self.proj(h)
        
        return h

# === 2) Per‐Scale MLP Head ===
class ScaleHead(keras.layers.Layer):
    
    def __init__(self,
                 noutputs: int,
                 hidden_units: int,
                 nlayers: int,
                 nblocks: int,
                 w0: float,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENIntermediateLayerInitializer(w0),
            bias_initializer="zeros"
        )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0) for _ in range(nblocks)]
        
        self.norm = keras.layers.LayerNormalization()
        
        # self.out    = keras.layers.Dense(
        #     noutputs,
        #     kernel_initializer="GlorotNormal",
        #     bias_initializer="zeros"
        # )

    def call(self, shared_feat):
        
        h = self.pre(shared_feat)
        
        for blk in self.blocks:
            h = blk(h)
        
        # h = self.out(h)
        
        h = self.norm(h)
        
        return h

# === 3) Multi‐Scale Hybrid WindNet ===
class MultiScaleWindNetShared(BaseModel):
    
    def __init__(self,
                 noutputs=3,
                 init_log_sigmas=[-4,-2,0,2,4],
                 hidden_units=64,
                 nlayers=1,
                 nblocks=3,
                 w0_backbone=1.0,
                 w0_heads=[1e-2, 1e-1, 1e0, 1e1],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.nstages = len(w0_heads)
        self.hidden_units = hidden_units
        # shared stem
        self.backbone = SharedBackbone(
            hidden_units=hidden_units,
            init_log_sigmas=init_log_sigmas,
            w0=w0_backbone
        )
        
        # one head per scale
        self.heads = []
        
        for w0 in w0_heads:
            head = ScaleHead(
                noutputs=noutputs,
                hidden_units=hidden_units,
                nlayers=nlayers,
                nblocks=nblocks,
                w0=w0,
            )
            
            self.heads.append(head)
        
        self.mixer = keras.Sequential(
                                    [
                                        keras.layers.Dense(hidden_units, kernel_initializer=SIRENIntermediateLayerInitializer(1.0), activation=SIRENActivation(1.0)),
                                        keras.layers.Dense(noutputs, kernel_initializer="zeros")
                                    ]
                                    )
        
        
        init = np.array([+6.0] + [-6.0]*(self.nstages-1), dtype=np.float32)
        self.gates = self.add_weight(
            name="gates", shape=(self.nstages,),
            initializer=tf.constant_initializer(init),
            trainable=True
        )
        
        self.scaler = Scaler()

    def call(self, inputs):
        
        gates = tf.sigmoid(self.gates)
        
        shared = self.backbone(inputs)
        
        residuals = []
        
        for i, head in enumerate(self.heads):
            
            h = head(shared)   # (batch, hidden_units)
            # low_feat = h                # feed into next band
            residuals.append(gates[i]*h)
        
        # # stack: (batch, nstages, hidden_units)
        # y = tf.stack(residuals, axis=1)
        #
        # y = tf.reshape(y, (-1, self.nstages*self.hidden_units))
        
        y = tf.concat(residuals, axis=1)
        
        y = self.mixer(y)
        
        # y = tf.stack(residuals, axis=1)
        #
        # y = tf.reduce_mean(y, axis=1)
        
        return self.scaler(y)
    
    def build(self, input_shape):
        
        super().build(input_shape) 
        
        self.step = tf.Variable(0.0, trainable=False)
        
    # def update_mask(self, total_steps):
    #
    #     self.step.assign_add(1.0)
    #
    #     seg = total_steps / float(self.nstages)
    #     s   = self.step
    #     ramp_w = seg * 0.2
    #
    #     gates = []
    #     # stage 0 always fully on
    #     gates.append(7.0)
    #
    #     # for i=1…nstages-1, ramp from 0→1 over steps [i*seg, (i+1)*seg]
    #     for i in range(1, self.nstages):
    #         start = seg * i
    #         # end = seg * (i+1)
    #         g = tf.clip_by_value((s - start) / ramp_w, -1.0, 1.0)
    #         gates.append( 7.0*g )
    #
    #     gates = tf.stack(gates, axis=0)  # shape (nstages,)
    #
    #     self.gates.assign(gates)


# -----------------------------------------------------------------------------
class MultiScaleAttNet(BaseModel):
    """
    Multi‐scale PINN with a self‐attention fusion layer,
    and an extra learnable down‐scale on w to keep PDE‐loss stable.
    """
    def __init__(self,
                 noutputs=3,
                 scale_params=[
                            (-4.0, 1e-2),
                            (-2.0, 1e-1),
                            ( 0.0, 1e0),
                            ( 2.0, 1e0),
                            ( 4.0, 1e0),
                            ( 8.0, 1e0),
                            ],
                 hidden_units=64,
                 nlayers=1,
                 nblocks=3,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.noutputs = noutputs
        self.nstages   = len(scale_params)

        # 1) build one ScaleNet per frequency‐band
        self.scale_nets = []
        for (init_log_sigma, w0) in scale_params:
            
            net = ScaleNet(noutputs, hidden_units, nlayers, nblocks,
                           init_log_sigma=init_log_sigma, w0=w0)
            
            self.scale_nets.append(net)

        # 2) self‐attention fusion across the "stage" axis
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=hidden_units // 4,
            output_shape=hidden_units
        )

        # 3) a tiny feed‐forward on each attended vector
        self.ff = keras.Sequential([
            keras.layers.Dense( hidden_units, activation=SIRENActivation(1.0) ),
            keras.layers.Dense( hidden_units, activation=SIRENActivation(1.0) ),
        ])

        # final projection to (u,v,w)
        self.to_uvw = keras.layers.Dense(self.noutputs)
        
        # 5) final per‐component scaler
        self.scaler = Scaler()

    def call(self, inputs):
        
        # 1) run each ScaleNet → hidden_units features
        hs = [net(inputs) for net in self.scale_nets]  # list of (batch,hidden_units)

        # 2) stack into (batch, nstages, hidden_units)
        x = tf.stack(hs, axis=1)

        # 3) self-attend (with residual)
        y = self.attn(query=x, key=x, value=x) + x

        # 4) feed-forward (applied per-stage, with residual)
        #    note: FF returns (batch, nstages, noutputs)
        y = self.ff(y) + y
        
        uvw = self.to_uvw(y)

        # 5) collapse scales → (batch, noutputs)
        uvw = tf.reduce_mean(uvw, axis=1)

        # 7) re-stack and apply your global scaler
        uvw = self.scaler(uvw)
        
        return uvw