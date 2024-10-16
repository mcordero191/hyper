'''
Created on 22 Apr 2024

@author: mcordero
'''
import numpy as np

import tensorflow as tf
import keras

data_type     = tf.float32
version       = 5.00

keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="hyper")
class DropoutLayer(tf.keras.layers.Layer):
    
    def __init__(self, n=5000):
        
        super().__init__()
    
        self.n = tf.Variable(1.0*n, trainable=False)
        self.i = tf.Variable(0.0, trainable=False)
        
    def build(self, input_shape):
    
        in_dim = input_shape[-1]
        
        self.in_dim = in_dim
        
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
class DeepEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 n_neurons,
                 n_filters,
                 kernel_initializer = 'GlorotNormal',
                 activation  = None, #'LeakyReLU',
                 stddev = None,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        if stddev is not None:
            kernel_initializer = tf.keras.initializers.RandomNormal(0.0, stddev)
        
        self.n_neurons = n_neurons
        self.n_filters = n_filters
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        
        #v11.0
        self.amp = 3*tf.range(1, n_filters+1, delta=1, dtype=np.float32)
        #v12.00
        delta = 2.0
        self.amp = tf.linspace(1.0, delta*n_filters+1, num=n_filters)
        
    def build(self, input_shape):
        
        input_dim = input_shape[-1]
        
        self.w = self.add_weight(
                                shape = (input_dim, self.n_neurons, self.n_filters),
                                initializer = self.kernel_initializer,
                                )
        
        self.b = self.add_weight(
                                shape = (1, self.n_neurons, self.n_filters),
                                initializer = 'zeros',
                                )
        
    def call(self, inputs):
        '''
        Input     :    [..., nd]
        Output    :    [..., nfeatures, nfilters]
        '''
        
        x0 = tf.einsum('mi,ijk,k->mjk', inputs, self.w, self.amp) + self.b
        
        if self.activation is None:
        #Fearures
            sinx = tf.sin(x0)
            cosx = tf.cos(x0)
            
            x = tf.concat([sinx,  cosx,
                           ], axis=1)
        else:
            x = self.activation(x0)
        
        return(x)

@keras.saving.register_keras_serializable(package="hyper")
class DeepDense(tf.keras.layers.Layer):

    def __init__(self,
                 n_neurons,
                 kernel_initializer = 'GlorotNormal',
                 activation  = None, #'LeakyReLU',
                 stddev = None,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        if stddev is not None:
            kernel_initializer = tf.keras.initializers.RandomNormal(0.0, stddev)
        
        self.n_neurons = n_neurons
        self.kernel_initializer = kernel_initializer
        
        if activation is None:
            self.activation = tf.keras.activations.linear
        else:
            self.activation = activation
        
    def build(self, input_shape):
        
        nfilters = input_shape[2]
        nfeatures = input_shape[1]
        
        self.w = self.add_weight(
                                shape = (nfeatures, self.n_neurons, nfilters),
                                initializer = self.kernel_initializer,
                                )
        
        self.b = self.add_weight(
                                shape = (self.n_neurons, nfilters),
                                initializer = 'zeros',
                                )
        
    def call(self, inputs):
        '''
        Inputs    :    [..., nfeatures1, nfilters]
        Output    :    [..., nfeatures2, nfilters]
        '''
        
        x0 = tf.einsum('mik,ijk->mjk', inputs, self.w) + self.b
        
        x = self.activation(x0)
        
        return(x)

@keras.saving.register_keras_serializable(package="hyper")
class DeepMixFilters(tf.keras.layers.Layer):

    def __init__(self,
                 n_features,
                 n_filters,
                 kernel_initializer = 'GlorotNormal',
                 activation  = None, #'LeakyReLU',
                 stddev = None,
                 ravel = False,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        if stddev is not None:
            kernel_initializer = tf.keras.initializers.RandomNormal(0.0, stddev)
        
        self.n_filters = n_filters
        self.n_features = n_features
        self.ravel = ravel
        
        self.kernel_initializer = kernel_initializer
        
        if activation is None:
            self.activation = tf.keras.activations.linear
        else:
            self.activation = activation
        
    def build(self, input_shape):
        
        nfilters = input_shape[2]
        nfeatures = input_shape[1]
        
        self.w = self.add_weight(
                                shape = (nfeatures, nfilters, self.n_features, self.n_filters),
                                initializer = self.kernel_initializer,
                                )
        
        self.b = self.add_weight(
                                shape = (self.n_features, self.n_filters),
                                initializer = 'zeros',
                                )
        
    def call(self, inputs):
        '''
        Inputs    :    [..., nfeatures1, nfilters1]
        Output    :    [..., nfeatures2, nfilters2]
        '''
        
        x0 = tf.einsum('mij,ijkl->mkl', inputs, self.w) + self.b
        
        x = self.activation(x0)
        
        if self.ravel:
            x = tf.reshape(x, (-1, self.n_filters*self.n_features) )
            
        return(x)

@keras.saving.register_keras_serializable(package="hyper")
class DeepBlock(tf.keras.layers.Layer):

    def __init__(self,
                 n_features,
                 n_filters,
                 n_layers,
                 kernel_initializer = 'LecunNormal',
                 activation    = 'sine',
                 **kwargs
                 ):
                 
        super().__init__(**kwargs)
            
        if activation == 'sine':
            activation = tf.sin
        elif activation == 'tanh':
            activation = tf.tanh
        elif activation == 'swish':
            activation = tf.keras.activations.swish
        else:
            raise ValueError
        
        self.n_features  = n_features
        self.n_layers   = n_layers
        self.n_filters   = n_filters
        
        layers = []
        for _ in range(n_layers):
            dense  = DeepDense(n_features, kernel_initializer, activation)
            
        layers.append(dense)
        
        self._layers = layers
        self.mixer = DeepMixFilters(n_features, n_filters, kernel_initializer, activation)
        
    def call(self, x):
        '''
        Inputs    :    [.., nfilters1, nfeatures1]
        Output    :    [..., nfilters2, nfeatures2]
        '''
        
        for layer in self._layers:
            x = layer(x)
        
        x = self.mixer(x)    
        
        return(x)
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class DeepMultiNet(tf.keras.Model):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 n_filters=5,
                 n_blocks=3,
                 kernel_initializer = 'LecunNormal',
                 values = [1e1,1e1,1e0],
                 activation    = 'sine',
                 add_nu=False,
                 **kwargs
                 ):
                 
        super().__init__(**kwargs)
        
        self.n_outs     = n_outs
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        self.n_filters  = n_filters
        self.n_blocks   = n_blocks
        
        self.add_nu     = add_nu
        
        self.emb        = DeepEmbedding(n_neurons, n_filters, kernel_initializer)
        
        blocks = []
        
        for _ in range(n_blocks):
            block     = DeepBlock(n_neurons, n_filters, n_layers, 
                                        kernel_initializer,
                                        activation)
        
            blocks.append(block)
            
        self.blocks = blocks
        self.mixer      = DeepMixFilters(n_outs, 1,
                                         kernel_initializer,
                                         activation=None,
                                         ravel=True)
        
        self.scale = Scaler(np.array(values), add_nu=add_nu)
        
    def call(self, inputs):
        '''
        Inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        
        Output    :    
        
                        [..., nfilters1, nfeatures1]
                        [..., noutputs, 1]
                        [..., noutputs]
        
        '''
        x = self.emb(inputs)
        
        x = self.blocks[0](x)
        
        xout = x
        
        for block in self.blocks[1:]:
            x = block(x)
            xout = xout + x
        
        uvw = self.mixer(xout)
        
        uvw = self.scale(uvw)
        
        return(uvw)
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

@keras.saving.register_keras_serializable(package="hyper")
class Shift(tf.keras.layers.Layer):

    def __init__(self,
                 n_nodes,
                 use_fixed_h=False,
                 reduce=False,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.n_nodes = n_nodes
        self.use_fixed_h = use_fixed_h
        self.reduce = reduce
        
    def build(self, input_shape):
        
        input_dim = input_shape[-1]
        
        self.xc_initializer = tf.keras.initializers.RandomUniform(-1.0, 1.0)
        
        # if version != 4:
        self.h_fixed = tf.constant( 2.0/np.sqrt(self.n_nodes), dtype=data_type)
        # else:
        #     self.h_fixed = tf.constant( 2.0/np.power(self.n_nodes, 1.0/input_dim), dtype=data_type )
        
        self.xc = self.add_weight(
            shape=(self.n_nodes, input_dim),
            initializer=self.xc_initializer,
        )
        
        # if (version < 4) or self.use_fixed_h:
        #     self.h = self.h_fixed
        # else:
        self.h_initializer  = tf.initializers.Constant(self.h_fixed) 
        # self.h_initializer  = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.3)
    
        self.h = self.add_weight(
            shape=(self.n_nodes, input_dim),
            initializer=self.h_initializer,
            constraint = tf.keras.constraints.NonNeg(),
        )
        
        self.built = True
            
        
    def call(self, inputs):
        '''
        '''
        _, nd = inputs.shape
        
        x = tf.reshape(inputs, shape=[-1, 1, nd])
        
        x = (x - self.xc)/(self.h)
        
        if self.reduce:
            return( tf.reduce_sum(x, axis=1) )
            
        return(x)
    
    def get_config(self):
        
        return {"n_nodes": self.n_nodes,
                "use_fixed_h": self.use_fixed_h}

@keras.saving.register_keras_serializable(package="hyper")
class Embedding(tf.keras.layers.Layer):

    def __init__(self,
                 n_neurons,
                 kernel_initializer = 'GlorotNormal',
                 activation  = None, #'LeakyReLU',
                 stddev = None,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        if stddev is not None:
            kernel_initializer = tf.keras.initializers.RandomNormal(0.0, stddev)
        
        # bias_initializer = 'GlorotNormal'
        bias_initializer = tf.keras.initializers.RandomNormal(0.0, np.pi/12)
        
        self.n_neurons = n_neurons
        
        self.layer0 = tf.keras.layers.Dense(n_neurons,
                                      activation = activation,
                                      kernel_initializer = kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      )
        
        # self.layer1 = tf.keras.layers.Dense(n_neurons,
        #                               activation = activation,
        #                               kernel_initializer = kernel_initializer,
        #                               bias_initializer=kernel_initializer,
        #                               )
        #
        # self.layer2 = tf.keras.layers.Dense(n_neurons,
        #                               activation = activation,
        #                               kernel_initializer = kernel_initializer,
        #                               bias_initializer=kernel_initializer,
        #                               )
        
    def call(self, inputs, alpha=tf.constant(1.0) ):
        '''
        '''
        #Scaling factor when GlorotNormal distribution is used
        #otherwise the frequency components are too small
        inputs = tf.constant(10*np.pi)*inputs
        
        x0 = alpha*self.layer0(inputs)
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

@keras.saving.register_keras_serializable(package="hyper")
class NNKernel(tf.keras.layers.Layer):

    def __init__(self,
                 n_outputs=3,
                 n_layers=3,
                 n_kernel=10,
                 kernel_initializer = 'LecunNormal',
                 activation  = tf.sin, #'LeakyReLU',
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.n_kernel = n_kernel
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.kernel_initializer = kernel_initializer
        self.activation = activation
            
        layers = []
        
        layer = tf.keras.layers.Dense(n_kernel,
                                          activation=tf.sin,
                                          kernel_initializer=kernel_initializer,
                                          )
            
        layers.append(layer)
            
        for _ in range(n_layers-2):
            layer = tf.keras.layers.Dense(n_kernel,
                                          activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          )
            
            layers.append(layer)
        
        layer = tf.keras.layers.Dense(n_outputs,
                                      activation=None,
                                      kernel_initializer=kernel_initializer,
                                      )
            
        layers.append(layer)
        
        self.layers = layers
        
    def get_config(self):
        
        return {"n_outputs": self.n_outputs,
                "n_kernel": self.n_kernel,
                "n_layers": self.n_layers,
                "kernel_initializer": self.kernel_initializer,
                "activation": self.activation,
                }
        
    def call(self, inputs, training=False):
        '''
        Inputs [npoints, nnodes, nd]
        
        Outputs [npoints, nnodes, noutputs] 
        '''
        
        shape = inputs.shape
        
        x = tf.reshape(inputs, shape=[-1, shape[2]])
        
        for layer in self.layers:
            x = layer(x)
        
        x = tf.reshape(x, shape=[shape[0], shape[1], -1])
        
        return(x)

@keras.saving.register_keras_serializable(package="hyper")
class Densenet(tf.keras.layers.Layer):

    def __init__(self,
                 n_neurons,
                 n_layers=1,
                 kernel_initializer = 'LecunNormal',
                 activation  = tf.sin,
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
            self.activation = tf.keras.activations.swish
        else:
            self.activation  = tf.keras.activations.linear
        
        self.n_neurons          = n_neurons
        self.n_layers           = n_layers
        self.skip_connection    = skip_connection
        self.activavion         = activation
        
        layers = []
        for i in range(n_layers):
            layer = tf.keras.layers.Dense(n_neurons,
                                          activation=None,
                                          kernel_initializer=kernel_initializer,
                                          trainable=trainable,
                                          # name='%s_%d' %(name, i),
                                          )
            layers.append(layer)
            
        self._layers = layers
        
    def call(self, u, alphas=tf.ones(10)):
        '''
        '''
        for i, layer in enumerate(self.layers):
            
            u = layer(u)
            u = self.activation(alphas[i]*u)
        
        return(u)

@keras.saving.register_keras_serializable(package="hyper")
class LaafLayer(tf.keras.layers.Layer):

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
            self.activation = tf.keras.activations.swish
        else:
            self.activation  = tf.keras.activations.linear
        
    def build(self, input_shape):
        
        nfeatures = input_shape[1]
        
        self.w = self.add_weight(
                                shape = (nfeatures, self.n_neurons),
                                initializer = self.kernel_initializer,
                                )
        
        self.b = self.add_weight(
                                shape = (self.n_neurons, ),
                                initializer = 'zeros',
                                )
        
    def call(self, inputs, alpha=tf.constant(1.0)):
        '''
        '''
        
        u = tf.matmul(inputs, self.w) + self.b
        u = self.activation(alpha*u)
        
        return(u)

@keras.saving.register_keras_serializable(package="hyper")
class Linear(tf.keras.layers.Layer):

    def __init__(self,
                 noutputs = 1,
                 kernel_initializer = 'GlorotNormal',
                 constraint = None,
                 **kwargs
                 ):
        
        super().__init__()
        self.kernel_initializer = kernel_initializer
        self.constraint = constraint
        self.noutputs = noutputs
        
    def build(self, input_shape):
        
        nnodes = input_shape[1]
        # nd     = input_shape[2]
        
        self.w = self.add_weight(
            shape=(nnodes, self.noutputs),
            initializer = self.kernel_initializer,
            constraint = self.constraint, 
            trainable=True,
        )
        
        # self.b = self.add_weight(
        #     shape=(1, self.noutputs),
        #     initializer='zeros',
        #     trainable=True
        # )
        
        # self.built = True
        
    def call(self, inputs, alpha=tf.constant(1.0)):
        '''
        Inputs [npoints, nnodes, nd]
        
        Outputs [npoints, nd] 
        '''
        # npoints, nnodes, nd = inputs.shape
        
        
        u = tf.matmul(inputs, self.w)# + self.b
        u = alpha*u
        
        return(u)
    
    # def get_config(self):
    #
    #     config = super().get_config()
    #     config.update({"kernel_initializer": self.kernel_initializer})
    #
    #     return config

@keras.saving.register_keras_serializable(package="hyper")
class EinsumLayer(tf.keras.layers.Layer):

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
class StackLayer(tf.keras.layers.Layer):

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
class ConcatLayer(tf.keras.layers.Layer):

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
class Scaler(tf.keras.layers.Layer):
    
    def __init__(self,
                 values=[10.,10.,1.],
                 # add_nu = False,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.values = values
        # self.add_nu = add_nu
        self.scaling = tf.constant(self.values, name='output_scaling', dtype=data_type)
        
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
        
        # self.nu = self.add_weight(name='Nu',
        #                 shape = (1,),
        #                 initializer = 'ones',
        #                 trainable = self.add_nu,
        #                 constraint = tf.keras.constraints.NonNeg())
        
        
        
    def call(self, inputs):
        
        u = tf.multiply(inputs, self.w) + self.b
        
        #2023/11/22 v230.xx 
        u = tf.multiply(u, self.scaling)
        
        return u

class iPINN(tf.keras.Model):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 kernel_initializer = 'LecunNormal',
                 values = [1e1,1e1,1e0],
                 activation    = 'sine',
                 add_nu=False,
                 ):
                 
        super().__init__()
        
            
        self.n_out      = n_outs
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        
        self.add_nu     = add_nu
        
        # self.emb = Embedding(n_neurons=n_neurons, kernel_initializer=kernel_initializer)
        
        # kernel_initializer = 'HeNormal'
        
        self.u_fnn = Densenet(n_neurons=n_neurons, n_layers=n_layers-1, activation=activation, name='u_fnn', kernel_initializer=kernel_initializer)
        self.v_fnn = Densenet(n_neurons=n_neurons, n_layers=n_layers-1, activation=activation, name='v_fnn', kernel_initializer=kernel_initializer)
        self.w_fnn = Densenet(n_neurons=n_neurons, n_layers=n_layers-1, activation=activation, name='w_fnn', kernel_initializer=kernel_initializer)
        
        self.u_linear = Densenet(n_neurons=1, n_layers=1, activation=None, name='u_linear', kernel_initializer=kernel_initializer)
        self.v_linear = Densenet(n_neurons=1, n_layers=1, activation=None, name='v_linear', kernel_initializer=kernel_initializer)
        self.w_linear = Densenet(n_neurons=1, n_layers=1, activation=None, name='w_linear', kernel_initializer=kernel_initializer)
        
        self.concat = ConcatLayer()
        
        self.scale = Scaler(values, add_nu=add_nu)
        
    def call(self, x):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        # x = self.emb(inputs)
        
        u = self.u_fnn(x)
        v = self.v_fnn(x)
        w = self.w_fnn(x)
        
        u = self.u_linear(u)
        v = self.v_linear(v)
        w = self.w_linear(w)
        
        uvw = self.concat(u,v,w)
        
        uvw = self.scale(uvw)
        
        return(uvw)
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
class rPINN(tf.keras.Model):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 kernel_initializer = 'LecunNormal',
                 values = [1e0,1e0,1e-1],
                 activation    = 'sine',
                 add_nu=False,
                 ):
                 
        super().__init__()
        
            
        self.n_out      = n_outs
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        
        self.add_nu     = add_nu
        
        self.emb = Embedding(n_neurons=n_neurons, kernel_initializer=kernel_initializer)
        
        kernel_initializer = 'GlorotNormal'
        
        self.fnn = Densenet(n_neurons=n_neurons, n_layers=n_layers-2, activation=activation, name='fnn', kernel_initializer=kernel_initializer)
        self.linear = Densenet(n_neurons=n_outs, n_layers=1, activation=None, name='linear', kernel_initializer=kernel_initializer)
        
        self.scale = Scaler(values, add_nu=add_nu)
        
    def call(self, inputs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        
        u = self.emb(inputs)
        u = self.fnn(u)
        u = self.linear(u)
        u = self.scale(u)
        
        return(u)
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class DeepONetOri(tf.keras.Model):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 n_coeffs,
                 n_neurons_t=None,
                 kernel_initializer = 'LecunNormal',
                 values = [1e0,1e0,1e-1],
                 activation    = 'sine',
                 add_nu=False,
                 ):
                 
        super().__init__()
        
        if n_neurons_t is None:
            n_neurons_t = n_neurons
            
        self.n_out      = tf.constant(n_outs)
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        self.n_coeffs   = n_coeffs
        self.norm       = 1.0/np.sqrt(n_coeffs)
        
        self.add_nu     = add_nu
        
        zero_initializer = kernel_initializer
        
        # self.t_emb = Shift(n_nodes=n_neurons_t, name='t_emb', reduce=True)
        self.t_emb = Embedding(n_neurons=n_neurons, name='t_emb', kernel_initializer=kernel_initializer)
        self.t_fnn = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='t_dense', kernel_initializer=kernel_initializer)
        
        self.t_u = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='t_u', kernel_initializer=zero_initializer)
        self.t_v = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='t_v', kernel_initializer=zero_initializer)
        self.t_w = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='t_w', kernel_initializer=zero_initializer)
        
        # self.x_emb = Densenet(n_neurons=n_neurons, activation=None, name='x_emb', kernel_initializer=kernel_initializer)
        self.x_emb = Embedding(n_neurons=n_neurons, name='x_emb', kernel_initializer=kernel_initializer)
        self.x_fnn = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='x_dense', kernel_initializer=kernel_initializer)
        
        self.x_u = Densenet(n_neurons=n_coeffs, n_layers=1, activation=activation, name='x_u', kernel_initializer=kernel_initializer)
        self.x_v = Densenet(n_neurons=n_coeffs, n_layers=1, activation=activation, name='x_v', kernel_initializer=kernel_initializer)
        self.x_w = Densenet(n_neurons=n_coeffs, n_layers=1, activation=activation, name='x_w', kernel_initializer=kernel_initializer)
        
        self.einsum = EinsumLayer()
        self.stack = StackLayer()
        
        self.scale = Scaler(np.array(values)*self.norm, add_nu=add_nu)
        
    def call(self, inputs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        te = self.t_emb(inputs[:,0:1])
        te = self.t_fnn(te)
        ut = self.t_u(te)
        vt = self.t_v(te)
        wt = self.t_w(te)
        
        xe = self.x_emb(inputs[:,1:4])
        xe = self.x_fnn(xe)
        ux = self.x_u(xe)
        vx = self.x_v(xe)
        wx = self.x_w(xe)
        
        u = self.einsum(ut, ux)
        v = self.einsum(vt, vx)
        w = self.einsum(wt, wx)
        
        uvw = self.stack(u,v,w)
        
        uvw = self.scale(uvw)
        
        return(uvw)
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
class DeepONet(tf.keras.Model):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 n_coeffs,
                 kernel_initializer = 'LecunNormal',
                 values = [1e0,1e0,1e-1],
                 activation    = 'sine',
                 add_nu=False,
                 ):
                 
        super().__init__()
            
        self.n_out      = tf.constant(n_outs)
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        self.n_coeffs   = n_coeffs
        self.norm       = 1.0/np.sqrt(n_coeffs)
        
        self.add_nu     = add_nu
        
        #Good for DNS
        stddev = 0.01*np.sqrt(n_neurons)*np.pi
        
        #ICON?
        # stddev = 0.4*np.sqrt(n_neurons) #v10.07
        # stddev = 0.1*np.sqrt(n_neurons) #v10.08
        # stddev = 0.2*np.sqrt(n_neurons) #v10.09
        # stddev = 0.4*np.sqrt(n_neurons) #v10.10
        # stddev = 0.1*np.sqrt(n_neurons) #v10.11
        # stddev = 0.2*np.sqrt(n_neurons) #v10.12 and v10.13
        # stddev = 0.2*np.sqrt(n_neurons) #v10.14
        
        # stddev = np.sqrt(20.0/(n_neurons))
        # stddev = 0.1*np.sqrt(n_neurons) #v10.30
        
        # self.t_emb = Shift(n_nodes=n_neurons_t, name='t_emb', reduce=True)
        # self.t_emb = Densenet(n_neurons=n_neurons, activation=None, name='t_emb', kernel_initializer=kernel_initializer)
        self.t_emb = Embedding(n_neurons=n_neurons, name='t_emb')#, stddev=stddev)
        # self.t_fnn = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='t_dense', kernel_initializer=kernel_initializer)
        
        zero_initializer = kernel_initializer #'zeros'
        
        self.fnn_u = Densenet(n_neurons=n_coeffs, n_layers=n_layers-1, activation=activation, name='fnn_tu', kernel_initializer=zero_initializer)
        self.fnn_v = Densenet(n_neurons=n_coeffs, n_layers=n_layers-1, activation=activation, name='fnn_tv', kernel_initializer=zero_initializer)
        self.fnn_w = Densenet(n_neurons=n_coeffs, n_layers=n_layers-1, activation=activation, name='fnn_tw', kernel_initializer=zero_initializer)
        
        self.t_u = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='linear_tu', kernel_initializer=zero_initializer)
        self.t_v = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='linear_tv', kernel_initializer=zero_initializer)
        self.t_w = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='linear_tw', kernel_initializer=zero_initializer)
        
        #Good for DNS
        stddev = 0.1*np.sqrt(n_neurons)*np.pi
        
        #ICON?
        # stddev = 0.1*np.sqrt(n_neurons) #v10.07
        # stddev = 0.2*np.sqrt(n_neurons) #v10.08
        # stddev = 0.1*np.sqrt(n_neurons) #v10.09 and v10.10
        # stddev = 0.1*np.sqrt(n_neurons) #v10.11 and v10.12 and v10.13
        # stddev = 0.05*np.sqrt(n_neurons) #v10.14
        #
        # stddev = np.sqrt(2.0/(n_neurons))
        
        # self.x_emb = Shift(n_nodes=n_neurons, name='x_emb', reduce=True)
        # self.x_emb = Densenet(n_neurons=n_neurons, activation=None, name='x_emb', kernel_initializer=kernel_initializer)
        self.x_emb = Embedding(n_neurons=n_neurons, name='x_emb')#, stddev=stddev)
        # self.x_fnn = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='x_dense', kernel_initializer=kernel_initializer)
        
        self.x_u = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='x_u', kernel_initializer=kernel_initializer)
        self.x_v = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='x_v', kernel_initializer=kernel_initializer)
        self.x_w = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='x_w', kernel_initializer=kernel_initializer)
        
        self.einsum = EinsumLayer()
        self.stack = StackLayer()
        
        self.scale = Scaler(np.array(values)*self.norm, add_nu=add_nu)
        
    def call(self, inputs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        te = self.t_emb(inputs[:,0:1])
        
        u_t = self.fnn_u(te)
        v_t = self.fnn_v(te)
        w_t = self.fnn_w(te)
        
        u_t = self.t_u(u_t)
        v_t = self.t_v(v_t)
        w_t = self.t_w(w_t)
        
        # u_t = self.c_u(a_t)
        # v_t = self.c_v(a_t)
        # w_t = self.c_w(a_t)
        
        xe = self.x_emb(inputs[:,1:4])
        
        u_x = self.x_u(xe)
        v_x = self.x_v(xe)
        w_x = self.x_w(xe)
        
        # u = tf.einsum('ik,ik->i', u_x, u_t, name='einsum_u')
        # v = tf.einsum('ik,ik->i', v_x, v_t, name='einsum_v')
        # w = tf.einsum('ik,ik->i', w_x, w_t, name='einsum_w')
        
        
        u = self.einsum(u_x, u_t)
        v = self.einsum(v_x, v_t)
        w = self.einsum(w_x, w_t)
        
        uvw = self.stack(u,v,w)
        
        uvw = self.scale(uvw)
        
        return(uvw)
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class DeepONetOpt(tf.keras.Model):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 n_coeffs,
                 kernel_initializer = 'LecunNormal',
                 values = [1e0,1e0,1e-1],
                 activation    = 'sine',
                 add_nu=False,
                 ):
                 
        super().__init__()
            
        self.n_out      = tf.constant(n_outs)
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        self.n_coeffs   = n_coeffs
        self.norm       = 1.0/np.sqrt(n_coeffs)
        
        self.add_nu     = add_nu
        
        # self.t_emb = Shift(n_nodes=n_neurons_t, name='t_emb', reduce=True)
        self.t_emb = Embedding(n_neurons=n_neurons, name='t_emb', kernel_initializer=kernel_initializer)
        self.t_fnn = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='t_dense', kernel_initializer=kernel_initializer)
        
        zero_initializer = kernel_initializer #'zeros'
        
        # self.t_u = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='t_u', kernel_initializer=zero_initializer)
        # self.t_v = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='t_v', kernel_initializer=zero_initializer)
        # self.t_w = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='t_w', kernel_initializer=zero_initializer)
        
        self.t_u = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='t_u', kernel_initializer=zero_initializer)
        self.t_v = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='t_v', kernel_initializer=zero_initializer)
        self.t_w = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='t_w', kernel_initializer=zero_initializer)
        
        # self.x_emb = Shift(n_nodes=n_neurons, name='x_emb', reduce=True)
        self.x_emb = Embedding(n_neurons=n_neurons, name='x_emb', kernel_initializer=kernel_initializer)
        self.x_fnn = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='x_dense', kernel_initializer=kernel_initializer)
        
        self.x_u = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='x_u', kernel_initializer=kernel_initializer)
        self.x_v = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='x_v', kernel_initializer=kernel_initializer)
        self.x_w = Densenet(n_neurons=n_coeffs, n_layers=3, activation=activation, name='x_w', kernel_initializer=kernel_initializer)
        
        self.einsum = EinsumLayer()
        self.stack = StackLayer()
        
        self.scale = Scaler(np.array(values), add_nu=add_nu)
        
    def call(self, inputs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        te = self.t_emb(inputs[:,0:1])
        a_t = self.t_fnn(te)
        
        u_t = self.t_u(a_t)
        v_t = self.t_v(a_t)
        w_t = self.t_w(a_t)
        
        # u_t = self.c_u(a_t)
        # v_t = self.c_v(a_t)
        # w_t = self.c_w(a_t)
        
        xe = self.x_emb(inputs[:,1:4])
        a_x = self.x_fnn(xe)
        
        u_x = self.x_u(a_x)
        v_x = self.x_v(a_x)
        w_x = self.x_w(a_x)
        
        # u = tf.einsum('ik,ik->i', u_x, u_t, name='einsum_u')
        # v = tf.einsum('ik,ik->i', v_x, v_t, name='einsum_v')
        # w = tf.einsum('ik,ik->i', w_x, w_t, name='einsum_w')
        
        
        u = self.einsum(u_x, u_t)
        v = self.einsum(v_x, v_t)
        w = self.einsum(w_x, w_t)
        
        uvw = self.stack(u,v,w)
        
        uvw = self.scale(uvw)
        
        return(uvw)
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class MultiNet(tf.keras.Model):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 n_coeffs,
                 kernel_initializer = 'LecunNormal',
                 activation = 'sine',
                 add_nu = False,
                 residual_layer = False,
                 ):
        
        super().__init__()
            
        self.n_out  = n_outs
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.n_coeffs = n_coeffs
        self.residual_layer = residual_layer
        
        self.net_mean = iPINN(n_outs,
                              n_neurons=10,
                              n_layers=5,
                              kernel_initializer=kernel_initializer,
                              activation=activation)
        
        if self.residual_layer:
        
            # self.net_residuals = DeepONetOri(n_outs,
            #                                  n_neurons,
            #                                  n_layers,
            #                                  n_coeffs,
            #                                  kernel_initializer=kernel_initializer,
            #                                  activation=activation,
            #                                  add_nu=add_nu)
            
            self.net_residuals = iPINN(n_outs,
                                      n_neurons=n_neurons,
                                      n_layers=n_layers,
                                      kernel_initializer=kernel_initializer,
                                      activation=activation,
                                      add_nu=add_nu)
        
    def call(self, inputs, **kwargs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        uvw0 = self.net_mean(inputs, **kwargs)
        
        if self.residual_layer:
            uvw_res = self.net_residuals(inputs, **kwargs)
        
            uvw = uvw0 + uvw_res
        
            return(uvw)
        
        return(uvw0)
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    
class BaseClass(tf.keras.Model):
    
    def __init__(self,
                 add_nu=False,
                 **kwargs
                 ):
                 
        super().__init__()
        
        self.nu = self.add_weight(name='Nu',
                        shape = (1,),
                        initializer = 'ones',
                        trainable = add_nu,
                        constraint = tf.keras.constraints.NonNeg())
    
                
    def call(self, inputs):
        '''
        inputs    :    [batch_size, dimensions] 
                                    d0    :    time
                                    d1    :    altitude
                                    d2    :    x
                                    d3    :    y
        '''
        
        raise NotImplementedError("This is a base class")
    
    def build_graph(self, in_shape):
        
        x = tf.keras.layers.Input(shape=in_shape)
        
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def update_mask(self, *args):
        
        pass
    
class FCNClass(BaseClass):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 kernel_initializer = 'GlorotNormal',
                 values = [1e0,1e0,1e0],
                 activation    = 'sine',
                 add_nu=False,
                 laaf=0,
                 dropout=0,
                 **kwargs
                 ):
                 
        super().__init__(add_nu=add_nu, **kwargs)
        
        self.n_out      = n_outs
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        
        self.dropout    = dropout
        
        self.i          = 0
            
        self.emb = Embedding(n_neurons=n_neurons, kernel_initializer=kernel_initializer)
        
        layers = []
        for _ in range(n_layers):
            layer = LaafLayer(n_neurons,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              )
            layers.append(layer)
        
        self.laaf_layers = layers
        
        if laaf:
            self.alphas = self.add_weight(
                            name="alphas",
                            shape=(n_layers, ),
                            initializer="ones",
                            # constraint = tf.keras.constraints.NonNeg(),
                            trainable=True,   
                        )
        else:
            self.alphas = tf.ones( (n_layers, ) )
        
        
        if self.dropout:
            
            layers = []
            for _ in range(n_layers+1):
                
                drop_layer = DropoutLayer()
                layers.append(drop_layer)
            
            self.dropout_layers = layers
            
        self.linear_layers = None
        
        self.scale = Scaler(values)
    
    def update_mask(self, n):
    
        if self.dropout:
            for drop_layer in self.dropout_layers:
                drop_layer.update_mask(n)
    
        
class resPINN(FCNClass):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 **kwargs
                 ):
                 
        super().__init__(n_outs,
                        n_neurons,
                        n_layers, **kwargs)
        
        layers = []
        for _ in range(n_layers):
            layer = Linear(n_outs,
                           kernel_initializer = 'zeros',
                           # constraint = tf.keras.constraints.NonNeg()
                           )
            
            layers.append(layer)
        
        self.linear_layers = layers 
        
        
    def call(self, inputs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        
        u = self.emb(inputs)
        
        if self.dropout:
            u = self.dropout_layers[0](u)
        
        # u = self.laaf_layers[0](u, self.alphas[0])
        #
        # if self.dropout:
        #     u = self.dropout_layers[1](u)
        
        output = tf.constant(0.0)#self.linear_layers[0](u)
        
        for i in range(0,self.n_layers):
            
            u = self.laaf_layers[i](u, self.alphas[i])
            
            if self.dropout:
                u = self.dropout_layers[i+1](u)
            
            output = output + self.linear_layers[i](u)
        
        # output = self.linear(output)#, alpha=self.alphas[i+2])
        
        output = self.scale(output)
        
        return(output)
    
class genericPINN(FCNClass):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 **kwargs
                 ):
                 
        super().__init__(n_outs,
                        n_neurons,
                        n_layers, **kwargs)
        
        self.linear_layer = Linear(n_outs,
                                   kernel_initializer = 'zeros',
                                    # constraint = tf.keras.constraints.NonNeg()
                                   )
                
    def call(self, inputs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        
        u = self.emb(inputs)
        
        if self.dropout:
            u = self.dropout_layers[0](u)
        
        for i in range(0,self.n_layers):
            u = self.laaf_layers[i](u, self.alphas[i])
            
            if self.dropout:
                u = self.dropout_layers[i+1](u)
        
        u = self.linear_layer(u)
        
        output = self.scale(u)
        
        return(output)

class sPINN(BaseClass):

    def __init__(self,
                 n_outputs, #3
                 n_kernel,
                 n_layers,
                 n_nodes,
                 kernel_initializer = 'LecunNormal',
                 values = [1e0,1e0,1e0],
                 activation    = 'sine',
                 add_nu=False,
                 **kwargs
                 ):
        
        super().__init__(add_nu=add_nu, **kwargs)
        
        # if activation == 'sine':
        #     activation = tf.sin
        # elif activation == 'tanh':
        #     activation = tf.tanh
        # elif activation == 'swish':
        #     activation = tf.keras.activations.swish
        # else:
        #     raise ValueError
        
        self.n_outputs  = n_outputs
        self.n_nodes = n_nodes
        self.n_kernel = n_kernel
        self.n_layers = n_layers
        self.activation      = activation
        
        self.emb    = Shift(n_nodes=n_nodes)
        
        # self.kernel = NNKernel(n_outputs=n_outputs,
        #                        n_layers=n_layers,
        #                        n_kernel=n_kernel,
        #                        activation=activation)
        
        self.u_kernel = FourierKernel(n_outputs=1,
                                   n_layers=n_layers,
                                   n_kernel=n_kernel,
                                   activation=activation,
                                   kernel_initializer=kernel_initializer)
        
        self.v_kernel = FourierKernel(n_outputs=1,
                                   n_layers=n_layers,
                                   n_kernel=n_kernel,
                                   activation=activation,
                                   kernel_initializer=kernel_initializer)
        
        self.w_kernel = FourierKernel(n_outputs=1,
                                   n_layers=n_layers,
                                   n_kernel=n_kernel,
                                   activation=activation,
                                   kernel_initializer=kernel_initializer)
        
        self.u_linear = Linear(kernel_initializer='zeros')
        
        self.v_linear = Linear(kernel_initializer='zeros')
        
        self.w_linear = Linear(kernel_initializer='zeros')
        
        self.concat = ConcatLayer()
        
        self.scale = Scaler(values)
        
    
    def call(self, inputs, **kwargs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        x   = self.emb(inputs)
        
        u   = self.u_kernel(x)
        v   = self.v_kernel(x)
        w   = self.w_kernel(x)
        
        u = self.u_linear(u)
        v = self.v_linear(v)
        w = self.w_linear(w)
        
        uvw = self.concat(u,v,w, axis=1)
        
        # uvw   = self.linear(uvw, **kwargs)
        
        uvw  = self.scale(uvw)
        
        return(uvw)
