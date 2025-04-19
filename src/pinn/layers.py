'''
Created on 22 Apr 2024

@author: mcordero
'''
import tensorflow as tf
import keras

data_type     = tf.float32

keras.saving.get_custom_objects().clear()

class PositionalEncoding(keras.layers.Layer):
    
    def __init__(self,
                 new_dim,
                 kernel_initializer="GlorotNormal",
                 stddev = None,
                 cte = 1e3,
                 width = 40,
                 ):
        
        width = new_dim//4
        
        super(PositionalEncoding, self).__init__()
        
        if stddev is not None:
            # kernel_initializer = keras.initializers.RandomNormal(0.0, stddev)
            width = new_dim*stddev

        self.new_dim = new_dim
        self.kernel_initializer = kernel_initializer
        
        # Define projection layer to map the final positional encoding to new_dim
        self.projection = keras.layers.Dense(self.new_dim,
                                             kernel_initializer=self.kernel_initializer,
                                             use_bias=False,
                                             trainable=True,
                                             )
        
        # Precompute div_term for sinusoidal encoding
        # div_term = tf.exp(tf.range(0, self.new_dim // 2, dtype=tf.float32) * -(np.log(10000.0) / (self.new_dim // 2)))
        div_term = width*tf.pow(cte, -2*tf.range(0, self.new_dim // 2, dtype=tf.float32) / self.new_dim )
        
        # self.div_term = tf.tile(div_term, [n_dim, 1])   # Shape (n_dim, new_dim/2)
        self.div_term = div_term[tf.newaxis,tf.newaxis, :]  # Shape (1, new_dim/2)

        # Define separate position encodings for each dimension
        # self.position_encodings = [keras.layers.Dense(self.new_dim, trainable=False) for _ in range(4)]


    # def build(self, input_shape):
    #
    #     pass
        # n_dim = input_shape[1]
        
        
    def call(self, inputs):
        
        # batch_size = tf.shape(inputs)[0]  # Number of samples
        
        n_dim = inputs.shape[1]  # Number of positional dimensions (4: time, x, y, z)
        
        # Calculate sinusoidal encodings for each positional dimension
        pos_values = inputs[:,:,tf.newaxis]  # Shape (N, n_dim, 1)
        pos_values = pos_values * self.div_term
        
        encoding_sin = tf.sin(pos_values)  # Shape (N, n_dim, new_dim/2)
        encoding_cos = tf.cos(pos_values)  # Shape (N, n_dim, new_dim/2)

        # Concatenate sine and cosine encodings
        pos_encodings = tf.concat([encoding_sin, encoding_cos], axis=-1)  # Shape (batch_size, n_dim, new_dim)
        
        # Aggregate encodings across all positional dimensions
        pos_encodings = tf.reshape(pos_encodings, (-1,n_dim*self.new_dim))
        
        # Apply projection to ensure final output dimension is new_dim
        pos_encodings = self.projection(pos_encodings)  # Shape (batch_size, new_dim)
        
        
        return pos_encodings
    
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
        s    = 20*(tf.linspace(1.0, 0.0, self.in_dim) + wcut - 1.0)
        
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
    

@keras.saving.register_keras_serializable(package="hyper")
class Embedding(keras.layers.Layer):

    def __init__(self,
                 n_neurons,
                 kernel_initializer = 'GlorotNormal',
                 bias_initializer = "zeros",
                 activation  = None, #'LeakyReLU',
                 stddev = 1.0,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        if stddev is not None:
            kernel_initializer = keras.initializers.RandomNormal(0.0, stddev)
        
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
        inputs = alpha*inputs
        
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
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        
        self.n_neurons = n_neurons
        self.kernel_initializer = kernel_initializer
        
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
                                # regularizer = keras.regularizers.L2(0.001),
                                )
        
        self.b = self.add_weight(
                                shape = (self.n_neurons, ),
                                initializer = 'zeros',
                                trainable = True,
                                )
        
    def call(self, inputs, alpha=1.0):
        '''
        '''
        
        u = tf.matmul(inputs, self.w) + self.b
        u = self.activation(alpha*u)
        
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
                 values=[1e0,1e0,1e-1],
                 # add_nu = False,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.values = values
        # self.add_nu = add_nu
        
    def build(self, input_shape):
        
        n_in = input_shape[-1]
        
        self.w = self.add_weight(
            shape = (n_in,),
            initializer = 'ones',
            trainable=True,
        )
        
        self.b = self.add_weight(
            shape = (n_in,),
            initializer = 'zeros',
            trainable=True,
        )
        
        # non‑trainable “values” tensor
        self.scaling = self.add_weight(
            shape=(n_in,),
            initializer=keras.initializers.Constant(self.values),
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
        
        w = tf.multiply(self.scaling, self.w)
        
        u = tf.multiply(inputs, w) + self.b
        
        #2023/11/22 v230.xx 
        # u = tf.multiply(u, self.scaling)
        
        return u

class CustomParameters(keras.layers.Layer):
    
    def __init__(self, add_nu=False):
        
        super().__init__()
        
        self.nu = self.add_weight(
            shape=(1,),
            initializer="ones",
            trainable=add_nu,
            constraint = keras.constraints.NonNeg()
        )