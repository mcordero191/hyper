'''
Created on 28 May 2025

@author: radar
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

from pinn.layers import Scaler
from pinn.networks import BaseModel

# === SIREN Activation and Initializers ===
class SIRENActivation(keras.layers.Layer):
    
    def __init__(self, w0=1.0, **kwargs):
        
        super().__init__(**kwargs)
        self.initial_log_w0 = np.log(w0)
    
    def build(self, input_shape):
        
        self.log_sigma = self.add_weight(
            name="log_sigma",
            shape=(),
            initializer=tf.constant_initializer(self.initial_log_w0),
            trainable=True
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
                 # init_log_sigma=0.0,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.noutputs = noutputs
        # self.P = len(periods)

        # 1) Use the revised TidalEmbedding:
        # self.tidal = TidalEmbedding(periods)

        self.slow_rff = GaussianRFF(out_dim=hidden_units, w0=w0)
        
        # 2) Simple MLP on those 2P features:
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENIntermediateLayerInitializer(w0),
            bias_initializer="zeros"
        )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers=nlayers, w0=w0) for _ in range(nblocks)]
        
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

        rff_embed = self.slow_rff(inputs)
        
        h = self.pre(rff_embed)  # (batch, hidden_units)
        
        for layer in self.blocks:
            h = 0.5*(h + layer(h))
            
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
        bands = 2 ** np.arange(self.num_bands)
        
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
                 pe_bands=10,
                 w0=1.0,
                 # low_feat_dim=32,
                init_log_sigma=-2.0,
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
        self.fourier = GaussianRFF(out_dim=hidden_units, init_log_sigma=init_log_sigma) 
        
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENIntermediateLayerInitializer(w0),
            bias_initializer="zeros"
        )
        
        self.blocks = [ResidualBlock(hidden_units, nlayers=nlayers, w0=w0) for _ in range(nblocks)]
        
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

    def call(self, inputs, f_low_hidden):
        """
        inputs        : (batch,4) = [t,z,x,y]
        f_low_hidden  : (batch, low_feat_dim)   from LowFreqNet

        We do:
          pe_feats = PE(t,z,x,y)             shape = (batch, pe_dim)
          combine  = tf.concat([pe_feats, f_low_hidden], axis=-1)   shape=(batch, pe_dim+low_feat_dim)
          pass → SIREN residual → out_layer → u_high
        """
        four_feats = self.fourier(inputs)                    # (batch, pe_dim)
        
        h = tf.concat([four_feats, f_low_hidden], axis=-1)  # (batch, pe_dim + low_feat_dim)

        # 1) project down to hidden_units
        h = self.pre(h)                               # (batch, hidden_units)

        # 2) residual SIREN blocks
        for layer in self.blocks:
            h = 0.5 * (h + layer(h))                 # (batch, hidden_units)

        # 3) final output
        u_high = self.out_layer(h)                    # (batch, noutputs)
        
        return u_high

# === Residual Block ===
class ResidualBlock(keras.layers.Layer):
    
    def __init__(self, units, nlayers=2, w0=1.0):
        
        super().__init__()
        
        self.layers_ = [
            keras.layers.Dense(
                units,
                kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                bias_initializer=SIRENBiasInitializer(),
                activation=SIRENActivation(w0),
            ) for _ in range(nlayers)]
        
        
    def call(self, x):
        
        h = x
        for layer in self.layers_:
            h = layer(h)
            
        return 0.5*(x + h)

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
        
        self.fourier = GaussianRFF(out_dim=hidden_units, init_log_sigma=init_log_sigma)
        
        # Pre-layer projects time (and optionally z) to hidden_units
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENFirstLayerInitializer(w0),
            bias_initializer='zeros'
        )
        
        # Stack of residual blocks
        self.blocks = [ResidualBlock(hidden_units, nlayers=nlayers, w0=w0) for _ in range(nblocks)]
        
        # self.mix = GaussianRFF(out_dim=hidden_units, w0=w0)
        
        # Output layer to produce coefficients for each output and basis
        self.coeff = keras.layers.Dense(
            noutputs * num_basis,
            kernel_initializer='GlorotNormal',
            bias_initializer='zeros'
        )
        
    def call(self, inputs, f_low_hidden=None):
        
        # pass through pre-layer
        four_feats = self.fourier(inputs)                    # (batch, pe_dim)
        
        if f_low_hidden is not None:
            h = tf.concat([four_feats, f_low_hidden], axis=-1)  # (batch, pe_dim + low_feat_dim)
        else:
            h = four_feats

        # 1) project down to hidden_units
        h = self.pre(h)   
        
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
        
        self.fourier = GaussianRFF(out_dim=hidden_units, init_log_sigma=init_log_sigma)
        
        # Pre-layer projects spatial coords to hidden_units
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(w0),
            kernel_initializer=SIRENIntermediateLayerInitializer(w0),
            bias_initializer=SIRENBiasInitializer(),
        )
        
        # Residual stack
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0) for _ in range(nblocks)]
        
        # Output layer to produce basis values
        self.mix = keras.layers.Dense(
            num_basis,
            kernel_initializer='GlorotNormal',
            bias_initializer='zeros'
        )
        
        
    def call(self, inputs, f_low_hidden=None):
        
        # pre-layer
        four_feats = self.fourier(inputs)
        
        if f_low_hidden is not None:
            h = tf.concat([four_feats, f_low_hidden], axis=-1)  # (batch, pe_dim + low_feat_dim)
        else:
            h = four_feats
            
        h = self.pre(h)
        
        # residual stack
        for blk in self.blocks:
            h = blk(h)
        
        # h = self.sc(h)
        # produce basis: shape (batch, num_basis)
        phi = self.mix(h)
        
        return phi  # (batch, num_basis)

# === DeepONet: combines BranchNet and TrunkNet ===
class DeepONet(keras.layers.Layer):
    
    def __init__(self,
                 noutputs,
                 branch_hidden=64,
                 branch_layers=2,
                 trunk_hidden=64,
                 trunk_layers=2,
                 nblocks=3,
                 num_basis=64,
                 w0=1.0,
                 init_log_sigma=0.0,
                 # nlow_feat=0,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.branch = BranchNet(
            noutputs=noutputs,
            num_basis=num_basis,
            hidden_units=branch_hidden,
            nlayers=branch_layers,
            nblocks=nblocks,
            w0=w0,
            # nlow_feat=nlow_feat,
            # init_log_sigma=init_log_sigma,
        )
        
        self.trunk = TrunkNet(
            num_basis=num_basis,
            hidden_units=trunk_hidden,
            nlayers=trunk_layers,
            nblocks=nblocks,
            w0=w0,
            init_log_sigma=init_log_sigma,
        )
        
        # self.ena_hnet = self.add_weight(
        #     name="ena_hnet",
        #     shape=(),
        #     initializer='zeros',
        #     trainable=False,
        # )
        
    def call(self, inputs, f_low_hidden=None):
        # inputs: (batch, 4) = [t, z, x, y]
        # split into (t,z) for branch and (x,y) or (x,y,z) for trunk
        # If you have all 4 dims, branch gets (t,z), trunk gets (x,y)
        
        input_b = inputs[:, 0:2]   # (batch,2)
        input_t = inputs[:, 1:4]  # (batch,2)
        
        # call branch on (t,z), trunk on (x,y)
        a = self.branch(input_b, f_low_hidden)   # (batch, noutputs, num_basis)
        phi = self.trunk(input_t,f_low_hidden)  # (batch, num_basis)
        
        # Now combine: for each output c, do dot(a[c,:], phi)
        # a has shape (batch, noutputs, num_basis), phi is (batch, num_basis)
        y = tf.reduce_sum(a * phi[:,tf.newaxis,:], axis=-1)  # (batch, noutputs)
        
        return y
    
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
                 lf_hidden=32,
                 lf_layers=1,
                 lf_nblocks=3,
                 num_basis=256,
                 w0=6.0,
                 ena_deeponet=0,
                 init_log_sigma=0.0,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.low_net = LowFreqNet(
            noutputs=noutputs,
            hidden_units=lf_hidden,
            nlayers=lf_layers,
            nblocks=lf_nblocks,
            # w0=w0,
        )
        
        # 2) High‐frequency correction network
        if ena_deeponet:
            
            self.high_net = DeepONet(
                                    noutputs=noutputs,
                                    branch_hidden=hf_hidden,
                                    branch_layers=hf_layers,
                                    trunk_hidden=hf_hidden,
                                    trunk_layers=hf_layers,
                                    nblocks=num_blocks,
                                    num_basis=num_basis,
                                    w0=w0,
                                    init_log_sigma=init_log_sigma,
                                )
        else:
            self.high_net = HighFreqNet(
                                        noutputs=noutputs,
                                        hidden_units=hf_hidden,
                                        nlayers=hf_layers,
                                        nblocks=num_blocks,
                                        w0=w0,
                                        init_log_sigma=init_log_sigma,
                                    )
            
        self.scale = Scaler()

    def build(self, input_shape):
        
        super().build(input_shape) 
        
        self.ena_hnet = self.add_weight(
            name="ena_hnet",
            shape=(),
            initializer='zeros',
            trainable=False,
        )
        
        self.step_count = tf.Variable(0.0, trainable=False)
        
    def call(self, inputs, return_hf=False):
        
        u_lf, u_lf_feat = self.low_net(inputs)   # (batch, noutputs)
        u_lf = self.scale(u_lf)
        
        u_hf = self.high_net(inputs, u_lf_feat)  # (batch, noutputs)
        u_hf = u_hf*self.ena_hnet
        u_hf =  self.scale(u_hf)
        
        u = u_lf + u_hf
        # u = u_hf
        
        if return_hf:
            return u, u_hf
        
        return u
    
    def update_mask(self, total_epochs):
    
        self.step_count.assign_add(1.0)
        
        # Smooth linear ramp: 0→1 between epochs 0.2*total → 1.0*total
        alpha = (self.step_count - 0.1*total_epochs) / (0.05*total_epochs)
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
        self.ena_hnet.assign(alpha)
        

# -----------------------------------------------------------------------------
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
        
        self.out = keras.layers.Dense(
            noutputs,
            kernel_initializer="GlorotNormal",
            bias_initializer="zeros"
        )

    def call(self, inputs, low_feat=None):
        
        h = self.rff(inputs)
        
        if low_feat is not None:
            h = tf.concat([h, low_feat], axis=-1)  # (batch, pe_dim + low_feat_dim)
        
        h = self.pre(h)
        
        for blk in self.blocks:
            h = blk(h)
            
        return self.out(h), h  # (batch, noutputs)


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
                     ( 0.0, 1e+0),  # mid
                     ( 2.0, 1e+1),  # mid-high
                     # ( 4.0, 1e+2),  # very high freq
                 ],
                 hidden_units=64,
                 nlayers=1,
                 nblocks=3,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.noutputs = noutputs
        self.nstages   = len(scale_params)

        # build one ScaleNet per frequency‐band
        self.scale_nets = []
        
        for (init_log_sigma, w0) in scale_params:
            
            net = ScaleNet(noutputs, hidden_units, nlayers, nblocks=nblocks,
                           init_log_sigma=init_log_sigma,
                           w0=w0)
            
            self.scale_nets.append(net)

        # gating variables α_i in [0..1], one per stage
        # they start at 0 and will be ramped on in update_mask()
        # self.gates = []
        #
        # for i in range(self.nstages):
        #     g = self.add_weight(
        #         name=f"gate_{i}",
        #         shape=(),
        #         initializer="zeros",
        #         trainable=False
        #     )
        #
        #     self.gates.append(g)
        
        # instead of non‐trainable gates, make raw trainable parameters β_i
        init = np.array([+6.0] + [-6.0]*(self.nstages-1), dtype=np.float32)
        self.gates = self.add_weight(
            name="gates", shape=(self.nstages,),
            initializer=tf.constant_initializer(init),
            trainable=True
        )
        
        self.scaler = Scaler()
        
    def call(self, inputs):
        # inputs: (batch, 4) = [t,z,x,y]
        out = 0.0
        gates = tf.sigmoid(self.gates)  # shape=(nstages,), all in (0,1)
        
        # h, low_feat = self.scale_nets[0](inputs)
        # h = h*gates[0]
        
        for i in range(self.nstages):
            gate = gates[i]
            net = self.scale_nets[i]
            
            h, _ = net(inputs)
            out += gate * h
            
        return self.scaler(out)

    # def build(self, input_shape):
    #
    #     super().build(input_shape) 
    #
    #     self.step_count = tf.Variable(0.0, trainable=False)
    #

    # def update_mask(self, total_steps: float):
    #     """
    #     Call this each training step to ramp gates on in sequence.
    #     E.g. divide timeline into nstages+1 segments,
    #     ramp gate[i] from 0→1 during step ∈ [i/(n+1), (i+1)/(n+1)].
    #     """
    #     self.step_count.assign_add(1.0)
    #
    #     frac = self.step_count / total_steps  # in [0,1]
    #
    #     self.gates[0].assign(1.0) #First stage always enabled
    #
    #     for i in range(1, self.nstages):
    #         # each gate goes from 0→1 over its own window:
    #         start = i / self.nstages
    #         end   = (i+1) / self.nstages
    #         alpha = tf.clip_by_value((frac - start) / (end - start), 0.0, 1.0)
    #         self.gates[i].assign(alpha)

# -----------------------------------------------------------------------------

# === 1) Shared Backbone with Multiple RFF Banks ===
class SharedBackbone(keras.layers.Layer):
    
    def __init__(self,
                 hidden_units: int,
                 init_log_sigmas: list[float],
                 w0: float,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.rffs = [ GaussianRFF(out_dim=hidden_units, init_log_sigma=lg)
                      for lg in init_log_sigmas ]
        
        self.proj = keras.layers.Dense(
            hidden_units,
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
        
        self.blocks = [ResidualBlock(hidden_units, nlayers, w0) for _ in range(nblocks)]
        
        self.out    = keras.layers.Dense(
            noutputs,
            kernel_initializer="zeros",
            bias_initializer="zeros"
        )

    def call(self, shared_feat):
        
        h = shared_feat
        
        for blk in self.blocks:
            h = 0.5 * (h + blk(h))
            
        return self.out(h)

# === 3) Multi‐Scale Hybrid WindNet ===
class MultiScaleWindNetShared(BaseModel):
    
    def __init__(self,
                 noutputs=3,
                 init_log_sigmas=[-4,-2,0,2,4],
                 hidden_layers=64,
                 nlayers=1,
                 nblocks=3,
                 w0_backbone=1.0,
                 w0_heads=[1e-2, 1e-1, 1e0, 1e1,1e2],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.nstages = len(w0_heads)
        
        # shared stem
        self.backbone = SharedBackbone(
            hidden_units=hidden_layers,
            init_log_sigmas=init_log_sigmas,
            w0=w0_backbone
        )
        
        # one head per scale
        self.heads = []
        
        for w0 in w0_heads:
            head = ScaleHead(
                noutputs=noutputs,
                hidden_units=hidden_layers,
                nlayers=nlayers,
                nblocks=nblocks,
                w0=w0,
            )
            
            self.heads.append(head)
            
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
        out = 0.0
        
        for i in range(self.nstages):
            gate = gates[i]
            head = self.heads[i]
            
            out += gate * head(shared)
            
        return self.scaler(out)