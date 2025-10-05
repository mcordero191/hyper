'''
Created on 30 May 2025

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
        self.w0 = w0
        
    def call(self, x):
        
        return tf.sin(x)

class SIRENFirstLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0
    
    def __call__(self, shape, dtype=None):
        
        in_f = shape[0]
        limit = 1.0/(in_f * self.w0)
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

class SIRENIntermediateLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0
        
    def __call__(self, shape, dtype=None):
        
        in_f = shape[1]
        limit = tf.sqrt(6.0/in_f)/self.w0
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

# Example constraint: force log_periods ∈ [ln(0.8*T_i), ln(1.2*T_i)]
class ClipToRange(tf.keras.constraints.Constraint):
    
    def __init__(self, init_vals, frac=0.1):
        # init_vals = initial periods as a 1D NumPy or tf.Tensor of shape (P,)
        self.init_vals = init_vals    # (P,)
        
        self.lower = self.init_vals*(1.0 - frac)
        self.upper = self.init_vals*(1.0 + frac)

    def __call__(self, w):
        # w = current log_periods (shape (P,))
        
        return tf.clip_by_value(w, clip_value_min=self.lower, clip_value_max=self.upper)

class TidalEmbedding(tf.keras.layers.Layer):
    """
    Computes base angle θ_i(t,z,x,y) = 
        2π*(t_hr / T_i) 
        + 2π*(φ_z_i * z_km) 
        + 2π*(φ_x_i * x_km) 
        + 2π*(φ_y_i * y_km)
    and returns (sin(θ_i), cos(θ_i)) each of shape (batch, P).

    - T_i are trainable (via log_periods).
    - φ_z_i, φ_x_i, φ_y_i are trainable “cycles per km”.
    """

    def __init__(self, initial_periods, initial_phase_x=None, initial_phase_y=None, initial_phase_z=None, **kwargs):
        """
        initial_periods_hours: list of tidal periods, e.g. [24.0, 12.0, 8.0, 6.0, 4.8, ...]
        initial_phase_x/y/z: optional arrays (same length) of initial cycles/km
        """
        super().__init__(**kwargs)
        vals = np.array(initial_periods, dtype=np.float32)
        assert vals.ndim == 1, "initial_periods_hours must be 1D"
        self.P = vals.shape[0]
        self._init_values = vals
        self._init_phase_x = np.zeros(self.P, dtype=np.float32) if initial_phase_x is None else np.array(initial_phase_x, dtype=np.float32)
        self._init_phase_y = np.zeros(self.P, dtype=np.float32) if initial_phase_y is None else np.array(initial_phase_y, dtype=np.float32)
        self._init_phase_z = np.zeros(self.P, dtype=np.float32) if initial_phase_z is None else np.array(initial_phase_z, dtype=np.float32)

    def build(self, input_shape):
        
        self.periods = self.add_weight(
            name="periods",
            shape=(self.P,),
            initializer=tf.constant_initializer(self._init_values),
            trainable=False,
            constraint=ClipToRange(self._init_values, frac=0.05)
        )

        self.phase_z = self.add_weight(
            name="phase_z",
            shape=(self.P,),
            initializer=tf.constant_initializer(self._init_phase_z),
            trainable=True
        )
        self.phase_x = self.add_weight(
            name="phase_x",
            shape=(self.P,),
            initializer=tf.constant_initializer(self._init_phase_x),
            trainable=True
        )
        self.phase_y = self.add_weight(
            name="phase_y",
            shape=(self.P,),
            initializer=tf.constant_initializer(self._init_phase_y),
            trainable=True
        )

        super().build(input_shape)

    def call(self, t, z, x, y):
        """
        Inputs (each of shape (batch,1), normalized to [-1,1]):
        - t   → mapped to t_hr ∈ [0,24]
        - z   → mapped to z_km ∈ [0,20]
        - x,y → mapped to ±150 km
        """
        t_hr = (t + 1.0) * 12.0     # [0,24] hours
        z_km = (z + 1.0) * 10.0     # [0,20] km
        x_km = (x + 1.0) * 150.0    # [-150,150] km
        y_km = (y + 1.0) * 150.0

        theta = 2.0 * np.pi * (
            (t_hr / self.periods) +
            (z_km * self.phase_z) +
            (x_km * self.phase_x) +
            (y_km * self.phase_y)
        )

        sin_feats = tf.sin(theta)
        cos_feats = tf.cos(theta)
        
        return sin_feats, cos_feats
    
# === 2) Temporal RFF for turbulence & gravity wave freq modulation ===
class TemporalRFF(keras.layers.Layer):
    
    def __init__(self, num_feats=32, init_log_sigma=0.0, **kwargs):
        super().__init__(**kwargs)
        
        assert num_feats%2==0
        
        self.F = num_feats//2
        self.init_log_sigma = init_log_sigma
        
    def build(self, input_shape):
        
        w0 = np.random.randn(1,self.F).astype(np.float32)
        self.w_dir = self.add_weight(name='w_dir',shape=(1,self.F),
                                     initializer=tf.constant_initializer(w0),trainable=True)
        
        b0 = np.random.uniform(0,2*np.pi,(self.F,)).astype(np.float32)
        self.b = self.add_weight(name='b', shape=(self.F,),
                                 initializer=tf.constant_initializer(b0),trainable=True)
        
        self.log_sigma = self.add_weight(name='log_sigma', shape=(1,),
                                         initializer=tf.constant_initializer(self.init_log_sigma),trainable=True)
        super().build(input_shape)
        
    def call(self,t):
        
        alpha=tf.exp(self.log_sigma)
        proj = alpha*tf.matmul(t,self.w_dir)+self.b
        
        return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)

# === 3) Divergence-free RFF bases ===
class DivFreeRFF(keras.layers.Layer):
    
    def __init__(self, num_vec, init_log_sigma=0.0, **kwargs):
        
        super().__init__(**kwargs)
        assert num_vec%2==0
        
        self.F = num_vec//2
        self.init_log_sigma = init_log_sigma
        
    def build(self, input_shape):
        
        d = 3
        
        # random k vectors
        k0 = np.random.randn(self.F,d).astype(np.float32)
        self.k = self.add_weight(name='k', shape=(self.F,d),initializer=tf.constant_initializer(k0),trainable=True)
        
        b0 = np.random.uniform(0,2*np.pi,(self.F,)).astype(np.float32)
        self.b = self.add_weight(name='b', shape=(self.F,),initializer=tf.constant_initializer(b0),trainable=True)
        
        self.log_sigma = self.add_weight(name='log_sigma', shape=(1,),initializer=tf.constant_initializer(self.init_log_sigma),trainable=True)
        super().build(input_shape)
        
    def call(self, x):
        
        sigma = tf.exp(self.log_sigma)
        proj = sigma * tf.matmul(x, tf.transpose(self.k)) + self.b  # (batch,F)
        
        s = tf.sin(proj)
        # divergence-free basis directions
        k_hat = tf.linalg.l2_normalize(self.k, axis=1)  # (F,3)
        # fixed reference axis, tiled to match k_hat
        axis = tf.constant([1.0, 0.0, 0.0], dtype=k_hat.dtype)
        axis = tf.reshape(axis, (1, 3))
        axis = tf.tile(axis, [tf.shape(k_hat)[0], 1])  # (F,3)
        
        v1 = tf.linalg.l2_normalize(tf.linalg.cross(k_hat, axis), axis=1)
        v2 = tf.linalg.cross(k_hat, v1)
        # combine scalar features with vector directions
        f1 = s[..., None] * v1[None, :, :]  # (batch,F,3)
        f2 = s[..., None] * v2[None, :, :]
        
        return tf.concat([f1, f2], axis=1)  # (batch,2F,3)

# === 4) Curl-free (irrotational) RFF bases ===
class GradRFF(keras.layers.Layer):
    
    def __init__(self, num_sca, init_log_sigma=0.0, **kwargs):
        
        super().__init__(**kwargs)
        self.F = num_sca
        self.init_log_sigma = init_log_sigma
        
    def build(self,input_shape):
        
        w0 = np.random.randn(3,self.F).astype(np.float32)
        self.k = self.add_weight(name='k_sca', shape=(3,self.F),initializer=tf.constant_initializer(w0),trainable=True)
        
        b0 = np.random.uniform(0,2*np.pi,(self.F,)).astype(np.float32)
        self.b = self.add_weight(name='b_sca', shape=(self.F,),initializer=tf.constant_initializer(b0),trainable=True)
        
        self.log_sigma = self.add_weight(name='log_sigma', shape=(1,),initializer=tf.constant_initializer(self.init_log_sigma),trainable=True)
        super().build(input_shape)
        
    def call(self,x):
        
        alpha=tf.exp(self.log_sigma)
        proj = alpha*(tf.matmul(x,self.k)+self.b)  # (batch,F)
        c = tf.cos(proj)
        # gradient = k * cos(proj)
        # expand (batch,F,3)
        grad = c[...,None] * tf.transpose(self.k)[None,:,:]
        
        return grad  # (batch,F,3)
    
class BranchNet(keras.layers.Layer):
    """
    BranchNet with “A cos(θ) + B sin(θ)” tidal head instead of (phase_direct + α_sin + α_cos).
    """

    def __init__(self,
                 noutputs=3,
                 hidden_units=64,
                 nlayers=3,
                 num_vec=64,
                 num_sca=64,
                 gen_feats=32,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.noutputs = noutputs
        self.num_vec  = num_vec
        self.num_sca  = num_sca
        
        initial_periods = [
                            24.0, 12.0, 8.0,     # DW1, SW2, TW3 (migrating)
                            24.0, 24.0, 12.0, 12.0  # DE3, DW2, SE2, SW1 (non-migrating)
                        ]
                        
        # Approximate wave numbers over 300 km domain → k cycles/km = wavenumber / 300
        initial_phase_x = [
                            0.0, 0.0, 0.0,       # migrating: no x-dependence
                            3/300.0, 2/300.0, 2/300.0, 1/300.0  # non-migrating
                        ]

        # (A) TidalEmbedding now just produces (sin_feats, cos_feats, raw_angle)
        self.tidal = TidalEmbedding(initial_periods=initial_periods, initial_phase_x=initial_phase_x)  # P = 5
        self.P = self.tidal.P
        
        # (B) Temporal RFF for the MLP
        self.temp = TemporalRFF(num_feats=gen_feats)

        # (C) MLP that sees only e_temp
        self.pre = keras.layers.Dense(
            hidden_units,
            activation=SIRENActivation(1.0),
            kernel_initializer=SIRENIntermediateLayerInitializer(1.0),
            bias_initializer="zeros"
        )
        
        self.blocks = [
            keras.layers.Dense(
                hidden_units,
                activation=SIRENActivation(1.0),
                kernel_initializer=SIRENIntermediateLayerInitializer(1.0),
                bias_initializer="zeros"
            )
            for _ in range(nlayers)
        ]

        # (D) Coefficient heads for RFF trunk mixing
        self.vec_coeff = keras.layers.Dense(noutputs * num_vec, kernel_initializer="zeros")
        self.sca_coeff = keras.layers.Dense(noutputs * num_sca, kernel_initializer="zeros")
        self.bias_out  = keras.layers.Dense(noutputs,             kernel_initializer="zeros")

        # # (E) NEW: Instead of (phase_direct, α_sin, α_cos), learn two coeffs A, B per (channel, mode)
        #   # number of tidal modes
        # self.A = self.add_weight(
        #     name="A_tide", shape=(1, noutputs, self.P),
        #     initializer="zeros", trainable=True
        # )
        # self.B = self.add_weight(
        #     name="B_tide", shape=(1, noutputs, self.P),
        #     initializer="zeros", trainable=True
        # )
        
        layers = []
        for _ in range(2):
            
            layers.append(
                keras.layers.Dense(
                                    16,
                                    kernel_initializer=SIRENIntermediateLayerInitializer(1.0),
                                    bias_initializer='zeros',
                                    activation=SIRENActivation(1.0)
                                )
            )
            
        layers.append(
            keras.layers.Dense(
                                2*self.noutputs*self.P,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros'
                                )
                        )
        
        self.AB_layer = keras.Sequential(layers)


    def call(self, inputs):
        # inputs: (batch,4) = [t, z, x, y]
        t = inputs[:, 0:1]  # (batch,1)
        z = inputs[:, 1:2]  # (batch,1)
        x = inputs[:, 2:3]  # (batch,1)
        y = inputs[:, 3:4]  # (batch,1)

        # ───────────────────────────────────────────────────
        # 1) TidalEmbedding returns: sin_feats, cos_feats, θ   each (batch,P)
        # ───────────────────────────────────────────────────
        sin_feats, cos_feats = self.tidal(t, z, x, y)
        
        # Now each channel c gets:
        AB = self.AB_layer( tf.concat( [y,z], axis=-1 ) )
        
        A = AB[:,:self.noutputs*self.P]
        B = AB[:,self.noutputs*self.P:]
        
        A_shaped = tf.reshape(A, (-1, self.noutputs, self.P))
        B_shaped = tf.reshape(B, (-1, self.noutputs, self.P))
        
        #   A[c,i]*cos(θ_i) + B[c,i]*sin(θ_i), summed over i
        tide_only = tf.reduce_sum(A_shaped * cos_feats[:, tf.newaxis, :] + B_shaped * sin_feats[:, tf.newaxis, :], axis=-1)
        # tide_only has shape (batch, noutputs)

        # ───────────────────────────────────────────────────
        # 3) Temporal RFF → e_temp  (batch, 2F)
        #    MLP sees only e_temp (no e_tide!)
        # ───────────────────────────────────────────────────
        e_temp = self.temp(t)  # (batch, 2F)
        
        h = tf.concat([1e-2*tide_only, e_temp], axis=-1)
        
        h = self.pre(h)   # (batch, hidden_units)
        
        for layer in self.blocks:
            h = 0.5* ( h + layer(h) )

        vc = self.vec_coeff(h)  # (batch, noutputs * num_vec)
        sc = self.sca_coeff(h)  # (batch, noutputs * num_sca)
        b  = self.bias_out(h)   # (batch, noutputs)

        a_curl = tf.reshape(vc, (-1, self.noutputs, self.num_vec))
        a_grad = tf.reshape(sc, (-1, self.noutputs, self.num_sca))

        return a_curl, a_grad, b, tide_only

# === 6) TrunkNet: x,y,z -> u_curl basis and grad_phi basis ===
class TrunkNet(keras.layers.Layer):
    
    def __init__(self, num_vec, num_sca, mlp_units=64, mlp_layers=1, **kwargs):
        
        super().__init__(**kwargs)
        
        self.num_vec  = num_vec
        self.num_sca  = num_sca
        
        self.div_rff  = DivFreeRFF(num_vec)
        self.grad_rff = GradRFF(num_sca)

        in_dim = num_vec * 3 + num_sca * 3
        
        layers = []
        for _ in range(mlp_layers):
            
            layers.append(keras.layers.Dense(
                mlp_units,
                kernel_initializer=SIRENIntermediateLayerInitializer(1.0),
                bias_initializer='zeros',
                activation=SIRENActivation(1.0)
            ))
            
        layers.append(keras.layers.Dense(
            in_dim,
            kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        ))
        
        self.mixer = keras.Sequential(layers)

    def call(self, x):
        # x: (batch,3)
        u_curl   = self.div_rff(x)    # (batch, num_vec, 3)
        grad_phi = self.grad_rff(x)   # (batch, num_sca, 3)

        flat_c = tf.reshape(u_curl,   (-1, self.num_vec * 3))    # (batch, num_vec*3)
        flat_g = tf.reshape(grad_phi, (-1, self.num_sca * 3))    # (batch, num_sca*3)
        flat   = tf.concat([flat_c, flat_g], axis=-1)            # (batch, in_dim)

        mixed = self.mixer(flat)     # (batch, in_dim)

        mc = mixed[:, : self.num_vec * 3]
        mg = mixed[:, self.num_vec * 3 :]
        
        u_curl   = tf.reshape(mc,   (-1, self.num_vec, 3))
        grad_phi = tf.reshape(mg,   (-1, self.num_sca, 3))

        return u_curl, grad_phi

class DeepONet(BaseModel):
    
    def __init__(self, noutputs,
                 hidden_units=128, nlayers=4,
                 num_vec=64, num_sca=32,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.branch = BranchNet(
            noutputs=noutputs,
            hidden_units=hidden_units, nlayers=nlayers,
            num_vec=num_vec, num_sca=num_sca,
            gen_feats=hidden_units
        )
        
        self.trunk = TrunkNet(
            num_vec=num_vec, num_sca=num_sca,
            mlp_units=hidden_units, mlp_layers=nlayers,
        )
        
        self.scale = Scaler()
        
    
    def build(self, input_shape):
        
        super().build(input_shape) 
        
        self.a = self.add_weight(
            name="mesoscale",
            shape=(),
            initializer='zeros',
            trainable=False,
        )
        
        self.i = tf.Variable(0.0, trainable=False)

    def call(self, inputs):
        # inputs: (batch,4) = [t, z, x, y]
        a_curl, a_grad, b, tide_only = self.branch(inputs)
        u_curl, grad_phi = self.trunk(inputs[:, 1:4])

        term_c = tf.einsum('bci,bic->bc', a_curl, u_curl)   # (batch, noutputs)
        term_g = tf.einsum('bcj,bjc->bc', a_grad, grad_phi) # (batch, noutputs)

        # Final output = spatial mixture + bias + pure‐tide term
        u = self.a*(term_c + term_g + b) + tide_only
        # u = 1e2*tide_only
        
        u = self.scale(u)
        
        return(u)
    
    def update_mask(self, n):
    
        self.i.assign_add(1.0)
        
        if self.i > 0.1*n:
            self.a.assign(1.0)