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
            kernel_initializer="zeros",
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

class WindNet(BaseModel):
    """
    Combined = LowFreqNet + HighFreqNet.
    Stage 1: freeze HighFreqNet → train only LowFreqNet (learn tides, planetary waves)
    Stage 2: unfreeze HighFreqNet → train both (capture gravity waves, turbulence).
    """
    def __init__(self,
                 noutputs=3,
                 hf_hidden=64,
                 hf_layers=3,
                 hf_blocks=1,
                 lf_hidden=32,
                 lf_layers=3,
                 lf_nblocks=1,
                 num_basis=256,
                 w0=1.0,
                 ena_deeponet=0,
                 ena_film=0,
                 init_log_sigma=0.0,
                 n_subnets=1,
                 activation="siren",
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.low_net = LowFreqNet(
                                    noutputs=2,
                                    hidden_units=lf_hidden,
                                    nlayers=lf_layers,
                                    nblocks=lf_nblocks,
                                    init_log_sigma=0.0,
                                    activation=activation,
                                    # w0=w0,
                                )
        
        self.n_hlayers = n_subnets
        # 2) High‐frequency correction network
            
        blocks = []
        for i in range(n_subnets):
            hf_block = HighFreqNet(
                                        noutputs=2,
                                        hidden_units=hf_hidden,
                                        nlayers=hf_layers,
                                        nblocks=hf_blocks,
                                        w0=w0, #2*w0*(i+1),
                                        init_log_sigma=init_log_sigma,
                                        activation=activation,
                                    )
            blocks.append(hf_block)
        
            self.highnet_uv = blocks
        
        blocks = []
        for i in range(n_subnets):
            hf_block = HighFreqNet(
                                        noutputs=1,
                                        hidden_units=hf_hidden,
                                        nlayers=hf_layers,
                                        nblocks=hf_blocks,
                                        w0=w0, #2*w0*(i+1),
                                        init_log_sigma=init_log_sigma,
                                        activation=activation,
                                    )
            blocks.append(hf_block)
        
            self.highnet_w = blocks
            
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
        
        uv_lf, u_lf_feat = self.low_net(inputs)   # (batch, noutputs)
        # u_lf = self.scale(u_lf)
        
        uv_hf = 0
        for i, high_net in enumerate(self.highnet_uv):
            up, u_lf_feat = high_net(inputs, u_lf_feat)  # (batch, noutputs)
            up = up*gates[i]
            uv_hf += up
        
        w_hf = 0
        for i, high_net in enumerate(self.highnet_w):
            up, u_lf_feat = high_net(inputs, u_lf_feat)  # (batch, noutputs)
            up = up*gates[i]
            w_hf += up
        
        uv = uv_lf + uv_hf
        
        u = tf.concat([uv, w_hf], axis=-1) 
        
        # u = u_hf + u_lf
        
        u =  self.scale(u)
        
        if return_hf:
            return u, uv_hf
        
        return u
    
    def update_mask(self, total_epochs):
    
        self.step_count.assign_add(1.0)
        counter = 2.0*( tf.range(self.n_hlayers, dtype=tf.float32) + 1.0 )
        
        # Smooth linear ramp: 0→1 between epochs 0.2*total → 1.0*total
        alpha = (self.step_count - 0.15*total_epochs*counter) / (0.01*total_epochs)
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
        self.gates.assign(alpha)
        