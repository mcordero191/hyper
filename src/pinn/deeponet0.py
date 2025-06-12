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
        
        return tf.sin(self.w0 * x)

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

# === 1) Temporal Random Fourier Features with trainable log_sigma ===
class TemporalRFF(keras.layers.Layer):
    
    def __init__(self, out_dim: int = 64, init_log_sigma: float = 0.0, **kwargs):
        
        super().__init__(**kwargs)
        
        # assert out_dim % 2 == 0, 'out_dim must be even'
        self.D = out_dim
        self.init_log_sigma = init_log_sigma

    def build(self, input_shape):
        
        w0 = np.random.randn(1, self.D).astype(np.float32)
        
        self.w_dir = self.add_weight(name='w_dir',
                                     shape=(1, self.D),
                                     initializer=tf.constant_initializer(w0),
                                     trainable=True,
                                     )
        
        b0 = np.random.uniform(0, 2*np.pi, size=(self.D,)).astype(np.float32)
        
        self.b = self.add_weight(name='b',
                                 shape=(self.D,),
                                 initializer=tf.constant_initializer(b0),
                                 trainable=True,
                                 )
        
        self.log_sigma = self.add_weight(name='log_sigma',
                                         shape=(1,),
                                         initializer=tf.constant_initializer(self.init_log_sigma),
                                         trainable=True,
                                         )
        
        self.coef = np.sqrt(2.0 / self.D)
        
        super().build(input_shape)

    def call(self, t):
        
        alpha = tf.exp(self.log_sigma)
        proj = tf.matmul(alpha * t, self.w_dir) + self.b  # (batch, D)
        
        return self.coef * tf.sin(proj) #tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)

# === 2) Spatial Random Fourier Features with trainable log_sigma ===
class SpatialRFF(keras.layers.Layer):
    
    def __init__(self, out_dim: int = 64, init_log_sigma: float = 0.0, **kwargs):
        
        super().__init__(**kwargs)
        
        # assert out_dim % 2 == 0, 'out_dim must be even'
        
        self.D = out_dim #// 2
        self.init_log_sigma = init_log_sigma

    def build(self, input_shape):
        
        d = input_shape[-1]
        
        w0 = np.random.randn(d, self.D).astype(np.float32)
        
        self.w_dir = self.add_weight(name='w_dir',
                                     shape=(d, self.D),
                                     initializer=tf.constant_initializer(w0),
                                     trainable=True,
                                     )
        
        b0 = np.random.uniform(0, 2*np.pi, size=(self.D,)).astype(np.float32)
        
        self.b = self.add_weight(name='b',
                                 shape=(self.D,),
                                 initializer=tf.constant_initializer(b0),
                                 trainable=True,
                                 )
        
        self.log_sigma = self.add_weight(name='log_sigma',
                                         shape=(1,),
                                         initializer=tf.constant_initializer(self.init_log_sigma),
                                         trainable=True,
                                         )
        
        self.coef = np.sqrt(2.0 / self.D)
        
        super().build(input_shape)

    def call(self, x):
        
        alpha = tf.exp(self.log_sigma)
        
        proj = tf.matmul(alpha * x, self.w_dir) + self.b
        
        return self.coef * tf.sin(proj) #tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)

# === 3) Branch Net: time → coefficients a_i(t) ===
class BranchNet(keras.layers.Layer):
    
    def __init__(self, num_basis: int, hidden_units: int = 64, nlayers: int = 2,
                 rff_dim: int = 64, w0: float = 30.0, noutputs=3, **kwargs):
        
        super().__init__(**kwargs)
        
        self.noutputs = noutputs
        
        # temporal embedding with trainable log_sigma
        self.rff = TemporalRFF(out_dim=rff_dim, init_log_sigma=0.0)
        
        # initial SIREN pre-layer
        self.pre = keras.layers.Dense(
                                        hidden_units,
                                        kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                                        bias_initializer='zeros',
                                        activation=SIRENActivation(1.0),
                                        )
        
        # residual SIREN blocks
        self.blocks = []
        
        for _ in range(nlayers):
            
            layer = keras.layers.Dense(
                                        hidden_units,
                                        kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                                        bias_initializer='zeros',
                                        activation=SIRENActivation(1.0),
                                        )
            
            self.blocks.append(layer)
        
        # output heads: coefficients and bias
        self.coeff_out = keras.layers.Dense(
            self.noutputs * num_basis, kernel_initializer='zeros', bias_initializer='zeros')
        
        self.bias_out  = keras.layers.Dense(
            self.noutputs, kernel_initializer='zeros', bias_initializer='zeros')
        
        self.num_basis = num_basis

    def call(self, t):
        
        # temporal RFF
        h = self.rff(t)
        # SIREN pre-layer
        h = self.pre(h)
        
        # residual blocks
        for layer in self.blocks:
            h = h + layer(h)
            
        # compute a_i(t)
        coeffs = self.coeff_out(h)  # (batch, 3*p)
        a = tf.reshape(coeffs, (-1, self.noutputs, self.num_basis))  # (batch,3,p)
        
        # compute bias term b(t)
        b = self.bias_out(h)  # (batch,3)
        
        return a, b

# === 4) Trunk Net: x,y,z → basis φ_i(x) ===
class TrunkNet(keras.layers.Layer):
    
    def __init__(self, num_basis: int, hidden_units: int = 128, nlayers: int = 3,
                 rff_dim: int = 64, w0: float = 30.0, **kwargs):
        
        super().__init__(**kwargs)
        
        self.rff = SpatialRFF(out_dim=rff_dim, init_log_sigma=0.0)
        
        self.pre = keras.layers.Dense(
                                    hidden_units,
                                    kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                                    bias_initializer='zeros',
                                    activation=SIRENActivation(w0),
                                    )
        
        self.blocks = []
        
        for _ in range(nlayers):
            layer = keras.layers.Dense(
                                    hidden_units,
                                    kernel_initializer=SIRENIntermediateLayerInitializer(w0),
                                    bias_initializer='zeros',
                                    activation=SIRENActivation(1.0),
                                    )
            
            self.blocks.append(layer)
        
        self.out = keras.layers.Dense(
            num_basis, kernel_initializer='glorot_normal', bias_initializer='zeros')
        
        self.num_basis = num_basis

    def call(self, x):
        
        h = self.rff(x)
        h = self.pre(h)
        
        for layer in self.blocks:
            h = h + layer(h)
            
        return self.out(h)

# === 5) DeepONet: combine branch & trunk ===
class DeepONet(BaseModel):
    
    def __init__(self, noutputs, hidden_units=128, nlayers=4, num_basis: int = 64, **kwargs):
                          
        super().__init__(**kwargs)
        
        self.branch = BranchNet(num_basis, hidden_units=hidden_units, nlayers=nlayers, noutputs=noutputs)
        self.trunk  = TrunkNet(num_basis, hidden_units=hidden_units, nlayers=nlayers)

    def call(self, inputs):
        
        # inputs: (batch,4) = [t, x, y, z]
        t   = inputs[:, :1]    # (batch,1)
        xyz = inputs[:, 1:4]   # (batch,3)
        
        # branch returns coefficients a(t) and bias b(t)
        a, b = self.branch(t)  # a: (batch,3,p), b: (batch,3)
        phi  = self.trunk(xyz)  # (batch,p)
        
        # reconstruct u,v,w = sum_i a[:,c,i] * phi[:,i]
        u = tf.einsum('bci,bi->bc', a, phi)  # (batch,3)
        
        # add temporal bias term b(t)
        return u + b


# === BranchNet: produces a_curl(t), a_grad(t), and bias b(t) ===
class HelmholtzBranch(keras.layers.Layer):
    def __init__(self,num_vec,num_sca,hidden_units=64,nlayers=2,rff_feats=64,tidal_feats=8,**kwargs):
        super().__init__(**kwargs)
        self.temp_rff=TemporalRFF(num_feats=rff_feats,init_log_alpha=-2.0)
        self.tidal_rff=TemporalRFF(num_feats=tidal_feats,init_log_alpha=np.log(2*np.pi/12.0))
        self.pre=keras.layers.Dense(hidden_units,kernel_initializer=SIRENFirstLayerInitializer(30.0),bias_initializer='zeros')
        self.act0=SIRENActivation(30.0)
        self.blocks=[keras.layers.Dense(hidden_units,kernel_initializer=SIRENIntermediateLayerInitializer(30.0),bias_initializer='zeros') for _ in range(nlayers)]
        self.act=SIRENActivation(1.0)
        self.vec_coeff=keras.layers.Dense(3*num_vec,initializer='zeros',use_bias=True)
        self.sca_coeff=keras.layers.Dense(3*num_sca,initializer='zeros',use_bias=True)
        self.bias_out=keras.layers.Dense(3,initializer='zeros',use_bias=True)
        self.num_vec=num_vec
        self.num_sca=num_sca
    def call(self,t):
        h=tf.concat([self.temp_rff(t),self.tidal_rff(t)],axis=-1)
        h=self.act0(self.pre(h))
        for layer in self.blocks:
            h=h+self.act(layer(h))
        vc=self.vec_coeff(h)  # (batch,3*num_vec)
        sc=self.sca_coeff(h)  # (batch,3*num_sca)
        b=self.bias_out(h)    # (batch,3)
        a_curl=tf.reshape(vc,(-1,3,self.num_vec))
        a_grad=tf.reshape(sc,(-1,3,self.num_sca))
        return a_curl,a_grad,b

# === HelmholtzTrunk: produces u_curl(x) and grad_phi(x) bases ===
class HelmholtzTrunk(keras.layers.Layer):
    """
    Produces both divergence-free (curl A) and curl-free (grad phi) bases.
    """
    def __init__(self, num_vec, num_sca,
                 hidden_units=128, nlayers=2, rff_feats=64, w0=30.0, **kwargs):
        super().__init__(**kwargs)
        self.num_vec = num_vec
        self.num_sca = num_sca
        # shared spatial Fourier features
        self.spat_rff = SpatialRFF(num_feats=2*rff_feats, init_log_alpha=-2.0)
        # vector-potential MLP
        self.A_pre = keras.layers.Dense(hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(w0), bias_initializer='zeros')
        self.A_act0 = SIRENActivation(w0)
        self.A_blocks = [keras.layers.Dense(hidden_units,
            kernel_initializer=SIRENIntermediateLayerInitializer(w0), bias_initializer='zeros')
            for _ in range(nlayers)]
        self.A_act = SIRENActivation(1.0)
        self.A_out = keras.layers.Dense(3 * self.num_vec,
            kernel_initializer='glorot_normal', bias_initializer='zeros')
        # scalar-potential MLP
        self.P_pre = keras.layers.Dense(hidden_units,
            kernel_initializer=SIRENFirstLayerInitializer(w0), bias_initializer='zeros')
        self.P_act0 = SIRENActivation(w0)
        self.P_blocks = [keras.layers.Dense(hidden_units,
            kernel_initializer=SIRENIntermediateLayerInitializer(w0), bias_initializer='zeros')
            for _ in range(nlayers)]
        self.P_act = SIRENActivation(1.0)
        self.P_out = keras.layers.Dense(self.num_sca,
            kernel_initializer='glorot_normal', bias_initializer='zeros')

    def call(self, x):
        # Enclose all ops in tape to capture full gradient w.r.t. x
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            # vector potential path
            hA = self.spat_rff(x)
            hA = self.A_act0(self.A_pre(hA))
            for layer in self.A_blocks:
                hA = hA + self.A_act(layer(hA))
            A = self.A_out(hA)
            A = tf.reshape(A, (-1, self.num_vec, 3))
            # scalar potential path
            hP = self.spat_rff(x)
            hP = self.P_act0(self.P_pre(hP))
            for layer in self.P_blocks:
                hP = hP + self.P_act(layer(hP))
            phi = self.P_out(hP)
        # compute curl(A)
        Ax, Ay, Az = A[:,:,0], A[:,:,1], A[:,:,2]
        grad_Ax = tape.gradient(Ax, x)
        grad_Ay = tape.gradient(Ay, x)
        grad_Az = tape.gradient(Az, x)
        u_curl = tf.stack([
            grad_Az[:,:,1] - grad_Ay[:,:,2],
            grad_Ax[:,:,2] - grad_Az[:,:,0],
            grad_Ay[:,:,0] - grad_Ax[:,:,1]
        ], axis=-1)
        # compute grad(phi)
        grad_phi = tape.jacobian(phi, x)
        del tape
        return u_curl, grad_phi

class HelmholtzDeepONet(keras.Model):
    def __init__(self,num_vec=16,num_sca=8,**kwargs):
        super().__init__(**kwargs)
        self.branch=HelmholtzBranch(num_vec,num_sca)
        self.trunk=HelmholtzTrunk(num_vec,num_sca)
    def call(self,inputs):
        t=inputs[:,:1]; x=inputs[:,1:4]
        a_curl,a_grad,b=self.branch(t)
        u_curl,gphi=self.trunk(x)
        # combine: sum_i a_curl_c,i * u_curl_i + sum_j a_grad_c,j * gphi_j + b_c
        term_curl=tf.einsum('bci,bic->bc',a_curl,u_curl)
        term_grad=tf.einsum('bcj,bjc->bc',a_grad,gphi)
        return term_curl+term_grad+b