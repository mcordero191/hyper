'''
Created on 22 Apr 2024

@author: mcordero
'''
import numpy as np
import tensorflow as tf
import keras

data_type     = tf.float32

keras.saving.get_custom_objects().clear()

class SIRENLayerInitializer(keras.initializers.Initializer):
    
    def __init__(self, w0=1.0):
        
        self.w0 = w0

    def __call__(self, shape, dtype=None):
        
        in_feats = shape[1]
        limit = tf.sqrt( 2.0 / in_feats) / self.w0
        # limit = tf.sqrt(6.0 / in_feats) / self.w0
        
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

    
# @keras.saving.register_keras_serializable(package="hyper")
class DropoutLayer(keras.layers.Layer):
    
    def __init__(self, n=5000, **kwargs):
        
        super().__init__(**kwargs)
    
        self.max_n = n
        
    def build(self, input_shape):
    
        in_dim = input_shape[-1]
        
        self.in_dim = in_dim
        
        self.n = tf.Variable(1.0*self.max_n, trainable=False)
        self.i = tf.Variable(0.0, trainable=False)
        
        self.a = self.add_weight(
            shape=(self.in_dim,),
            initializer='zeros',
            trainable=False,
        )
        
    def update_mask(self, n):
        """
        unmask the i/m % of neuros
        """
        #Some are completely deactivated. v214
        # wcut = tf.constant(1.0)*i/n + tf.constant(0.)
        # s    = 10*( tf.linspace(1.0, 0.0, self.in_dim) + wcut - 1.0 )
        
        #all w's  activated but with low weights. v213
        # wcut = tf.constant(1.0)*i/n + tf.constant(0.2)
        # s = ( tf.linspace(1.0, -1.0, self.in_dim) + 2*wcut )
        
        #v250
        # wcut = tf.constant(1.25)*i/n + tf.constant(0.05)
        # s    = 5*( tf.linspace(1.0, 0.0, self.in_dim) + wcut - 1.0 )
        
        #v260: high slope
        wcut = tf.constant(1.1)*self.i/n + tf.constant(0.1)
        s    = 50*(tf.linspace(1.0, 0.0, self.in_dim) + wcut - 1.0)
        
        a = tf.where(s > 1.0, 1.0, s)
        a = tf.where(a < 0.0, 0.0, a)
        
        a = a/tf.reduce_mean(a)
        
        self.a.assign(a)
        
        self.i.assign_add(1.0)
        
    def call(self, inputs):
        
        u = tf.multiply(inputs, self.a)
        
        return u
        
    def get_config(self):
        
        return {"in_dim": self.in_dim}

class GaussianRFF(keras.layers.Layer):
    """
    Normalized Random Fourier Features generated in build():
      φ(x) = sqrt(2/D) [sin(w_i·x + b_i), cos(w_i·x + b_i)] for i=1..D/2
    where w_i are random directions scaled by learnable σ, and b_i random phases.
    """
    def __init__(self, out_dim: int = 256, init_log_sigma: float = 0.0, **kwargs):
        
        super().__init__(**kwargs)
        
        assert out_dim % 2 == 0, "out_dim must be even"
        # D = half of final features (sin+cos)
        self.D = out_dim // 2
        
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
        # self.log_sigma = self.add_weight(
        #     name='log_sigma',
        #     shape=(),
        #     initializer=tf.constant_initializer(self.init_log_sigma),
        #     trainable=True
        # )
        
        # normalize to unit variance
        self.coef = np.sqrt(2.0 / self.D)
        
        super().build(input_shape)

    def call(self, inputs, alpha):
        # xyz: (batch, d)
        # compute scaled directions
        
        sigma = tf.exp(alpha)
        
        w = sigma * self.w_dir   # (d, D)
        
        # project input
        proj = tf.matmul(inputs, w) + self.b  # (batch, D)
        
        return self.coef * tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)  # (batch, 2D)


# === 1) Positional Encoding (NeRF style) ===
class PositionalEncoding(keras.layers.Layer):
    """
    NeRF-style positional encoding: for each input dim, encodes
    [x, sin(2^k π x), cos(2^k π x)] for k=0..num_freqs-1.
    """
    def __init__(self, num_freqs: int = 10, include_input: bool = False, **kwargs):
        
        super().__init__(**kwargs)
        
        self.num_freqs = num_freqs
        self.include_input = include_input
        # frequency bands: [1, 2, 4, ..., 2^(num_freqs-1)]
        self.freq_bands = tf.constant([2.0**(i-num_freqs+5) for i in range(num_freqs)], dtype=tf.float32)
        
        # self.freq_bands = np.logspace(-1, np.log10(2**num_freqs), num_freqs)

    def call(self, inputs, alpha=1.0):
        
        # inputs: (batch, d)
        batch = tf.shape(inputs)[0]
        d = tf.shape(inputs)[1]
        
        out = []
        if self.include_input:
            out.append(inputs)
        # expand for broadcasting: (batch, d, 1)
        x_exp = tf.expand_dims(inputs, -1)
        # (batch, d, num_freqs)
        scaled = x_exp * self.freq_bands[None, None, :] * np.pi
        
        sin = tf.sin(scaled)
        cos = tf.cos(scaled)
        # flatten sin/cos: (batch, d * num_freqs)
        sin_flat = tf.reshape(sin, (batch, d * self.num_freqs))
        cos_flat = tf.reshape(cos, (batch, d * self.num_freqs))
        
        out.extend([sin_flat, cos_flat])
        
        return tf.concat(out, axis=-1)

# === 2) Trainable Fourier Features (spatial only) ===
class TrainableFourierFeatures(keras.layers.Layer):
    """
    Learns a set of spatial frequencies w_i, then embeds xyz via
    [sin(2π w_i x), cos(2π w_i x), ...] across all spatial dims.
    """
    def __init__(self, num_fourier: int = 32, init_scale: float = 10.0, **kwargs):
        
        super().__init__(**kwargs)
        
        self.num_freqs = num_fourier
        
        # initialize frequencies log-uniformly in [1/init_scale, init_scale]
        freqs = np.logspace(-1, np.log10(init_scale), num_fourier)
        
        # self.freqs = tf.Variable(freqs, trainable=True, dtype=tf.float32,
        #                          name='trainable_spatial_freqs')
    
        self.freqs = self.add_weight(
                name="trainable_spatial_freqs",
                shape=(num_fourier,),
                initializer=tf.constant_initializer(freqs),
                trainable=True
                )
        
    def call(self, inputs):
        # xyz: (batch,3)
        batch = tf.shape(inputs)[0]
        d = tf.shape(inputs)[1]
        
        # angles: (batch, 3, num_fourier)
        angles = inputs[..., None] * self.freqs[None, None, :] * np.pi
        
        sin = tf.sin(angles)  # (batch,3,num_fourier)
        cos = tf.cos(angles)
        
        # flatten sin/cos: (batch, d * num_freqs)
        sin_flat = tf.reshape(sin, (batch, d * self.num_freqs))
        cos_flat = tf.reshape(cos, (batch, d * self.num_freqs))
        
        # flatten to (batch, 6*num_fourier)
        feats = tf.concat([sin_flat, cos_flat], axis=-1)
        
        return feats

class HybridEmbedding(keras.layers.Layer):
    """
    Combines NeRF positional encoding (for all dims) with trainable spatial
    Fourier features, then projects to a hidden dimension for a SIREN backbone.
    """
    def __init__(self,
                 hidden_units: int,
                 pe_freqs: int = 10,
                 rff_features: int = 32,
                 rff_scale: float = 10.0,
                 w0: float = 1.0,
                 include_input: bool = False,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.w0 = w0
        self.hidden_units = hidden_units
        self.pe_freqs = pe_freqs
        self.rff_features = rff_features
        self.include_input = include_input
        # layers without project
        self.pos_enc = PositionalEncoding(num_freqs=pe_freqs,
                                          include_input=include_input)
        
        self.rff = TrainableFourierFeatures(num_fourier=rff_features,
                                            init_scale=rff_scale)
        # project will be created in build()
        # self.project = keras.layers.Dense(
        #                                     self.hidden_units,
        #                                     kernel_initializer=SIRENLayerInitializer(self.w0),
        #                                     bias_initializer=keras.initializers.Zeros()
        #                                 )

    # def build(self, input_shape):
    #     # input_shape: (batch, 4)
    #     # number of coordinate dims (t,x,y,z)
    #     d = input_shape[-1]
    #     # compute positional encoding dimension
    #     pe_dim = (d if self.include_input else 0) + 2 * d * self.pe_freqs
    #     # compute trainable Fourier feature dimension for x,y,z
    #     rff_dim = 2 * 3 * self.rff_features
    #     # total embedding dimension
    #     self.embed_dim = pe_dim + rff_dim
    #     # now create projection layer with known input dim
    #
    #     # build the projection to set its weights
    #     self.project.build((None, self.embed_dim))
    #     super().build(input_shape)

    def call(self, inputs, alpha: float = 1.0):
        
        # inputs: (batch,4) = [t, x, y, z]
        pe_feats = self.pos_enc(inputs)         # (batch, pe_dim)
        # xyz = inputs[:, 1:4]                   # (batch,3)
        rff_feats = self.rff(inputs)              # (batch, rff_dim)
        feats = tf.concat([pe_feats, rff_feats], axis=-1)  # (batch, embed_dim)
        
        # project into SIREN hidden_units, including scaling
        return (self.w0 * alpha * feats)


@keras.saving.register_keras_serializable(package="hyper")
class Embedding(keras.layers.Layer):

    def __init__(self,
                 n_neurons,
                 kernel_initializer = 'GlorotNormal',
                 bias_initializer = "zeros",
                 activation  = None, #'LeakyReLU',
                 stddev = 1.0,
                 w0=1.0,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.w0 = w0
        
        if stddev is not None:
            kernel_initializer = keras.initializers.RandomNormal(0.0, stddev)
        
        kernel_initializer = SIRENLayerInitializer(w0)
        
        # bias_initializer = 'GlorotNormal'
        # bias_initializer = keras.initializers.RandomNormal(0.0, np.pi/12)
        
        # self.alpha = tf.constant(stddev, dtype = data_type)
        
        self.layer0 = keras.layers.Dense(n_neurons,
                                      activation = activation,
                                      kernel_initializer = kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      use_bias=False,
                                      trainable=True,
                                      )
        
        # self.layer1 = keras.layers.Dense(n_neurons,
        #                               activation = activation,
        #                               kernel_initializer = kernel_initializer,
        #                               bias_initializer=kernel_initializer,
        #                               )
        #
        # self.layer2 = keras.layers.Dense(n_neurons,
        #                               activation = activation,
        #                               kernel_initializer = kernel_initializer,
        #                               bias_initializer=kernel_initializer,
        #                               )
        
    def call(self, inputs, alpha=1.0 ):
        '''
        Inputs: [npoints, nd]
        
        Outputs: [npoints, n_neurons]
        '''
        #Scaling factor when GlorotNormal distribution is used
        #otherwise the frequency components are too small
        inputs = self.w0*alpha*inputs
        
        x0 = self.layer0(inputs)
        # x1 = self.layer1(inputs)
        # x2 = self.layer2(inputs)
        
        #Fearures
        sinx = tf.sin(x0)
        
        # #version 3.0
        # return(sinx)
    
        cosx = tf.cos(x0)
        
        x0 = tf.concat([sinx, 
                        cosx, 
                       ], 
                        axis=1)
        
        #version 5.03
        return( x0 )
        
        #Fearures
        sinx = tf.sin(x0)
        # cosx = tf.cos(x0)
        
        sin3x = tf.sin(3*x0)
        # cos3x = tf.cos(3*x0)
        #
        sin5x = tf.sin(5*x0)
        # cos5x = tf.cos(9*x0)
        
        # sin_cos3x = tf.sin(3*np.pi*x1)*tf.cos(3*np.pi*x2)
        
        # expx = tf.exp(-x2**2)
        
        x = tf.concat([sinx, 
                        # cosx, 
                        sin3x,
                        # cos3x,
                        sin5x,
                        # cos5x,
                       ], axis=1)
        
        return(x)

# @keras.saving.register_keras_serializable(package="hyper")
class Densenet(keras.layers.Layer):

    def __init__(self,
                 n_neurons,
                 n_layers=1,
                 kernel_initializer = 'LecunNormal',
                 activation  = "sine",
                 trainable=True,
                 skip_connection=False,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        if activation == 'sine':
            self.activation = tf.sin
        elif activation == 'tanh':
            self.activation = tf.tanh
        elif activation == 'swish':
            self.activation = keras.activations.swish
        else:
            self.activation  = keras.activations.linear
        
        self.n_neurons          = n_neurons
        self.n_layers           = n_layers
        self.skip_connection    = skip_connection
        self.activavion         = activation
        
        layers = []
        for _ in range(n_layers):
            layer = keras.layers.Dense(n_neurons,
                                          activation=None,
                                          kernel_initializer=kernel_initializer,
                                          trainable=trainable,
                                          # name='%s_%d' %(name, i),
                                          )
            layers.append(layer)
            
        self.inner_layers = layers
        
    def call(self, u, alphas=None):
        '''
        '''
        if alphas is None:
            alphas = tf.ones(len(self.inner_layers))
            
        for i, layer in enumerate(self.inner_layers):
            
            u = layer(u)
            u = self.activation(alphas[i]*u)
        
        return(u)

@keras.saving.register_keras_serializable(package="hyper")
class LaafLayer(keras.layers.Layer):

    def __init__(self,
                 n_neurons,
                 kernel_initializer = 'GlorotNormal',
                 activation  = None, #'LeakyReLU',
                 w0=1.0,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.w0 = w0
        self.n_neurons = n_neurons
        self.kernel_initializer = SIRENLayerInitializer(w0) #kernel_initializer
        
        if activation == 'sine':
            self.activation = tf.sin
        elif activation == 'tanh':
            self.activation = tf.tanh
        elif activation == 'swish':
            self.activation = keras.activations.swish
        elif activation == 'gelu':
            self.activation = keras.activations.gelu
        else:
            self.activation  = keras.activations.linear
        
    def build(self, input_shape):
        
        nfeatures = input_shape[1]
        
        self.w = self.add_weight(
                                shape = (nfeatures, self.n_neurons),
                                initializer = self.kernel_initializer,
                                trainable = True,
                                )
        
        self.b = self.add_weight(
                                shape = (self.n_neurons, ),
                                initializer = 'zeros',
                                trainable = True,
                                )
        
    def call(self, inputs, alpha=0.0):
        '''
        '''
        
        sigma = tf.exp(alpha)
        
        u = tf.matmul(inputs, sigma*self.w) + self.b
        u = self.activation(u)
        
        return(u)

@keras.saving.register_keras_serializable(package="hyper")
class Linear(keras.layers.Layer):

    def __init__(self,
                 noutputs = 1,
                 kernel_initializer = 'GlorotNormal',
                 constraint = None,
                 add_bias=False,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.kernel_initializer = kernel_initializer
        self.constraint = constraint
        self.noutputs = noutputs
        self.add_bias = add_bias
        
    def build(self, input_shape):
        
        nnodes = input_shape[1]
        
        self.w = self.add_weight(
            shape=(nnodes, self.noutputs),
            initializer = self.kernel_initializer,
            constraint = self.constraint, 
            trainable=True,
        )
        
        # if self.add_bias:
        self.b = self.add_weight(
            shape=(1, self.noutputs),
            initializer='zeros',
            trainable=self.add_bias,
        )
        
        # self.built = True
        
    def call(self, inputs, alpha=1.0):
        '''
        Inputs [npoints, nnodes]
        
        Outputs [npoints, noutputs] 
        '''
        # npoints, nnodes = inputs.shape
        
        u = tf.matmul(inputs, self.w) + self.b
        u = alpha*u
        
        return(u)
    
    # def get_config(self):
    #
    #     config = super().get_config()
    #     config.update({"kernel_initializer": self.kernel_initializer})
    #
    #     return config

@keras.saving.register_keras_serializable(package="hyper")
class EinsumLayer(keras.layers.Layer):

    def __init__(self,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
    def call(self, input0, input1):
        '''
        Inputs [npoints, nnodes]
        
        Outputs [npoints] 
        '''
        
        u = tf.einsum('ik,ik->i', input0, input1)
        
        return(u)

@keras.saving.register_keras_serializable(package="hyper")
class StackLayer(keras.layers.Layer):

    def __init__(self,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)

    def call(self, input0, input1, input2, axis=1):
        '''
        Inputs [npoints, nnodes, nd]
        
        Outputs [npoints, nd] 
        '''
        
        u = tf.stack([ input0, input1, input2], axis=axis)
        
        return(u)
    
@keras.saving.register_keras_serializable(package="hyper")
class ConcatLayer(keras.layers.Layer):

    def __init__(self,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)

    def call(self, input0, input1, input2, axis=1):
        '''
        Inputs [npoints, nnodes, nd]
        
        Outputs [npoints, nd] 
        '''
        
        u = tf.concat([ input0, input1, input2], axis=axis)
        
        return(u)

@keras.saving.register_keras_serializable(package="hyper")
class Scaler(keras.layers.Layer):
    
    def __init__(self,
                 values=[1e0, 1e0, 1e0, 1e0],
                 add_bias=True,
                 # add_nu = False,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.values = values
        self.add_bias = add_bias
        # self.add_nu = add_nu
        
    def build(self, input_shape):
        
        n_in = input_shape[-1]
        
        self.w = self.add_weight(
            shape = (n_in,),
            initializer = 'ones',
            trainable=True,
        )
        
        if self.add_bias:
            self.b = self.add_weight(
                shape = (n_in,),
                initializer = 'zeros',
                trainable=True,
            )
        else:
            self.b = 0.0
        
        # non‑trainable “values” tensor
        self.scaling = self.add_weight(
            shape=(n_in,),
            initializer=keras.initializers.Constant(self.values[:n_in]),
            trainable=False,
            name="output_scaling"
        )
        
        # self.scaling = tf.constant(self.values, name='output_scaling', dtype=data_type)
        
        # self.nu = self.add_weight(name='Nu',
        #                 shape = (1,),
        #                 initializer = 'ones',
        #                 trainable = self.add_nu,
        #                 constraint = keras.constraints.NonNeg())
        
        
        
    def call(self, inputs):
        
        # w = tf.multiply(self.scaling, self.w)
        
        u = tf.multiply(inputs, self.w) + self.b
        
        #2023/11/22 v230.xx 
        u = tf.multiply(u, self.scaling)
        
        return u