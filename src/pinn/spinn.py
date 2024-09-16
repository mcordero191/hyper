'''
Created on 14 Aug 2024

@author: radar
'''
import numpy as np

import tensorflow as tf
import keras

from pinn.layers import BaseModel, Scaler, Linear

data_type     = tf.float32


@keras.saving.register_keras_serializable(package="hyper")
class Shift(keras.layers.Layer):

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
        
        h_fixed = tf.constant( 2.0/np.sqrt(self.n_nodes), dtype=data_type)
        h_initializer  = keras.initializers.Constant(h_fixed)
        
        #h_initializer  = keras.initializers.RandomNormal(mean=1.0, stddev=0.3) 
        
        xc_initializer = keras.initializers.RandomUniform(-1.0, 1.0)
        
        self.x0 = self.add_weight(
            shape=(1,self.n_nodes, input_dim),
            initializer=xc_initializer,
        )
        
        self.h = self.add_weight(
            shape=(1,self.n_nodes,1),
            initializer= h_initializer,
            constraint = keras.constraints.NonNeg(),
        )
        
        self.built = True
            
        
    def call(self, inputs):
        '''
        Inputs: [npoints, nd]
        
        Outputs: [npoints, n_nodes, (nd) ]
        '''
        _, nd = inputs.shape
        
        x = tf.reshape(inputs, shape=[-1, 1, nd])
        
        x = (x - self.x0)
        
        # x = tf.reduce_sum(tf.square(x), axis=2)
        
        # x = tf.sqrt(x)
        
        x = x/(self.h + tf.constant(1e-6))
        
        if self.reduce:
            return( tf.reduce_sum(x, axis=2) )
            
        return(x)
    
    def get_config(self):
        
        return {"n_nodes": self.n_nodes,
                "use_fixed_h": self.use_fixed_h}
        
@keras.saving.register_keras_serializable(package="hyper")
class FourierLayer(keras.layers.Layer):

    def __init__(self,
                 n_features,
                 kernel_initializer='HeUniform',
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.n_features = n_features
        self.kernel_initializer = kernel_initializer
    
    def get_config(self):
        
        return {"n_features": self.n_features,
                "kernel_initializer": self.kernel_initializer,
                }
    
    def build(self, input_shape):
        
        nd = input_shape[-1]
        
        # self.k = self.add_weight(
        #     shape=(nd, self.n_features),
        #     initializer=self.kernel_initializer,
        # )
        
        k = np.arange(1, self.n_features+1, dtype=np.float32)
        k = np.tile(k,(nd,1))
        
        k_initializer  = keras.initializers.Constant(k)
        
        self.k = self.add_weight(
            shape=(nd, self.n_features),
            initializer=k_initializer,
            trainable=False
        )
        
        # self.ak = self.add_weight(
        #     shape=(1, self.n_features),
        #     initializer=self.kernel_initializer,
        # )
        #
        # self.bk = self.add_weight(
        #     shape=(1, self.n_features),
        #     initializer=self.kernel_initializer,
        # )
        
        #
        # self.b = self.add_weight(
        #     shape=(1,self.n_features),
        #     initializer='zeros',
        # )
        
        self.built = True
        
    def call(self, x):
        '''
        Inputs: [npoints, nd]
        
        Outputs: [npoints, n_features]
        '''
        # nd = inputs.shape[1]
        
        # pi = tf.constant(np.pi)
        #
        kx = np.pi*tf.matmul(x, self.k)
        
        sin_x = tf.sin(kx)
        cos_x = tf.cos(kx)
        
        x = tf.concat([sin_x, cos_x], axis=1)
        
        return(x)

@keras.saving.register_keras_serializable(package="hyper")
class FeatureLayer(keras.layers.Layer):

    def __init__(self,
                n_features,
                 kernel_initializer='HeUniform',
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.n_features = n_features
        self.kernel_initializer = kernel_initializer
    
    def get_config(self):
        
        return {"n_features": self.n_features,
                "kernel_initializer": self.kernel_initializer,
                }
    
    def build(self, input_shape):
        
        nd = input_shape[-1]
        
        self.k = self.add_weight(
            shape=(nd, self.n_features),
            initializer=self.kernel_initializer,
        )
        
        # self.b0 = self.add_weight(
        #     shape=(1,self.n_features),
        #     initializer='zeros',
        # )
        
        # self.w1 = self.add_weight(
        #     shape=(nd, self.n_features),
        #     initializer=self.kernel_initializer,
        # )
        #
        # self.b1 = self.add_weight(
        #     shape=(1,self.n_features),
        #     initializer='zeros',
        # )
        #
        # self.w2 = self.add_weight(
        #     shape=(nd, self.n_features),
        #     initializer=self.kernel_initializer,
        # )
        #
        # self.b2 = self.add_weight(
        #     shape=(1,self.n_features),
        #     initializer='zeros',
        # )
        
        # self.w3 = self.add_weight(
        #     shape=(nd, self.n_features),
        #     initializer=self.kernel_initializer,
        # )
        #
        # self.b3 = self.add_weight(
        #     shape=(1,self.n_features),
        #     initializer='zeros',
        # )
        
        self.built = True
        
    def call(self, inputs):
        '''
        Inputs: [npoints, nd]
        
        Outputs: [npoints, n_features*2]
        '''
        # nd = inputs.shape[1]
        
        # pi = tf.constant(np.pi)
        
        kx0 = tf.constant(np.pi)*tf.matmul(inputs, self.k) #+ self.b0
        # kx1 = tf.matmul(inputs, self.w1) + self.b1
        # kx2 = tf.matmul(inputs, self.w2) + self.b2
        # kx3 = tf.matmul(inputs, self.w3) + self.b3
        
        # kx0 = pi*kx0
        # kx1 = pi*kx1
        # kx2 = pi*kx2
        
        sin_x = tf.sin(kx0)
        cos_x = tf.cos(kx0)
        # sincos_x = tf.sin(kx1)*tf.cos(kx2)
        #
        # sin_2x = tf.sin(2*kx0)
        # cos_2x = tf.cos(2*kx0)
        # sincos_2x = tf.sin(2*kx1)*tf.cos(2*kx2)
        #
        # sin_3x = tf.sin(4*kx0)
        # cos_3x = tf.cos(4*kx0)
        # sincos_3x = tf.sin(4*kx1)*tf.cos(4*kx2)
        #
        # sin_4x = tf.sin(8*kx0)
        # cos_4x = tf.cos(8*kx0)
        # sincos_4x = tf.sin(8*kx1)*tf.cos(8*kx2)
        
        # exp_x = tf.sin(-kx3**2)
        
        x = tf.concat([sin_x,  cos_x, 
                       # sincos_x,
                        # sin_2x, cos_2x, sincos_2x,
                        # sin_3x, cos_3x, sincos_3x,
                        # sin_4x, cos_4x, sincos_4x,
                        # exp_x,
                       ],
                       axis=1)
        
        return(x)

@keras.saving.register_keras_serializable(package="hyper")
class FourierKernel(keras.layers.Layer):

    def __init__(self,
                 n_outputs=1,
                 n_layers=3,
                 n_kernel=10,
                 kernel_initializer = 'LecunNormal',
                 activation  = "sine", #'LeakyReLU',
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        if activation == 'sine':
            activation = tf.sin
        elif activation == 'tanh':
            activation = tf.tanh
        elif activation == 'swish':
            activation = keras.activations.swish
        else:
            raise ValueError
        
        self.n_kernel = n_kernel
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        
        layers = []
        
        layer = FourierLayer(n_kernel,
                             kernel_initializer=kernel_initializer)
        
        layers.append(layer)
            
        for _ in range(n_layers-1):
            layer = keras.layers.Dense(n_kernel,
                                          activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          )
            
            layers.append(layer)
        
        layer = keras.layers.Dense(n_outputs,
                                      activation=None,
                                      kernel_initializer=kernel_initializer,
                                      )
        
        layers.append(layer)
        
        self.inner_layers = layers
        
    def get_config(self):
        
        return {"n_outputs": self.n_outputs,
                "n_kernel": self.n_kernel,
                "n_layers": self.n_layers,
                "kernel_initializer": self.kernel_initializer,
                "activation": self.activation,
                }
        
    def call(self, x):
        '''
        Inputs [npoints, nnodes, nd]
        
        Outputs [npoints, nnodes] 
        '''
        
        _,nnodes,nd = x.shape
        
        x = tf.reshape(x, shape=(-1, nd) )
        
        for layer in self.inner_layers:
            x = layer(x)
        
        x = tf.reshape(x, shape=(-1, nnodes) )
        
        return(x)

class GaussianKernel(keras.layers.Layer):

    def __init__(self,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
    def get_config(self):
        
        return {
                }
        
    def call(self, x):
        '''
        Inputs [npoints, nnodes, nd]
        
        Outputs [npoints, nnodes] 
        '''
        
        _,nnodes,nd = x.shape
        
        x = tf.reshape(x, shape=(-1, nd) )
        
        x = tf.reduce_sum( tf.square(x), axis=1 )
        
        x = tf.exp(-0.5*x)
        
        x = tf.reshape(x, shape=(-1, nnodes) )
        
        return(x)
    
@keras.saving.register_keras_serializable(package="hyper")
class NNKernel(keras.layers.Layer):

    def __init__(self,
                 n_outputs=1,
                 n_layers=4,
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
        
        if activation == 'sine':
            activation = tf.sin
        elif activation == 'tanh':
            activation = tf.tanh
        elif activation == 'swish':
            activation = keras.activations.swish
        else:
            raise ValueError
        
        self.activation = activation
            
        layers = []
        
        layer = FeatureLayer(n_kernel,
                              kernel_initializer=kernel_initializer)
        
        layers.append(layer)
        
        for _ in range(n_layers-1):
            layer = keras.layers.Dense(n_kernel,
                                          activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          )
        
            layers.append(layer)
        
        layer = keras.layers.Dense(n_outputs,
                                      activation=None,
                                      kernel_initializer=kernel_initializer,
                                      )
        
        layers.append(layer)
        
        self.inner_layers = layers
        
    def get_config(self):
        
        return {"n_outputs": self.n_outputs,
                "n_kernel": self.n_kernel,
                "n_layers": self.n_layers,
                "kernel_initializer": self.kernel_initializer,
                "activation": self.activation,
                }
        
    def call(self, x):
        '''
        Inputs [npoints, nnodes, nd]
        
        Outputs [npoints, nnodes] 
        '''
        
        _,nnodes,nd = x.shape
        
        x = tf.reshape(x, shape=(-1, nd) )
        
        for layer in self.inner_layers:
            x = layer(x)
        
        x = tf.reshape(x, shape=(-1, nnodes) )
        
        return(x)
    
class sPINN(BaseModel):

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
        
        # self.u_kernel = GaussianKernel()
        
        # self.u_kernel = FourierKernel(n_outputs=1,
        #                               n_layers=n_layers,
        #                               n_kernel=n_kernel,
        #                               activation=activation,
        #                               kernel_initializer=kernel_initializer)
        
        self.u_kernel = NNKernel(n_outputs=1,
                                   n_layers=n_layers,
                                   n_kernel=n_kernel,
                                   activation=activation,
                                   kernel_initializer=kernel_initializer)
        
        # self.v_kernel = NNKernel(n_outputs=1,
        #                            n_layers=n_layers,
        #                            n_kernel=n_kernel,
        #                            activation=activation,
        #                            kernel_initializer=kernel_initializer)
        #
        # self.w_kernel = NNKernel(n_outputs=1,
        #                            n_layers=n_layers,
        #                            n_kernel=n_kernel,
        #                            activation=activation,
        #                            kernel_initializer=kernel_initializer)
        
        self.u_linear = Linear(kernel_initializer='zeros', noutputs=self.n_outputs)
        
        # self.v_linear = Linear(kernel_initializer='zeros')
        #
        # self.w_linear = Linear(kernel_initializer='zeros')
        #
        # self.concat = ConcatLayer()
        
        self.scale = Scaler(values)
        
    
    def call(self, inputs):
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
        # v   = self.v_kernel(x)
        # w   = self.w_kernel(x)
        
        u = self.u_linear(u)
        # v = self.v_linear(v)
        # w = self.w_linear(w)
        
        # uvw = self.concat(u,v,w, axis=1)
        
        uvw  = self.scale(u)
        
        return(uvw)