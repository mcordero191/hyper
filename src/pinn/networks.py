'''
Created on 14 Aug 2024

@author: radar
'''
# import numpy as np
import tensorflow as tf
import keras

from pinn.layers import GaussianRFF, Embedding, LaafLayer, DropoutLayer, Scaler, Linear

# Define the Gaussian activation function
def gaussian_activation(x):
    
    return tf.exp(-tf.square(x))

class BaseModel(keras.Model):
    
    def __init__(self,
                 add_nu=False,
                 **kwargs
                 ):
                 
        super().__init__()
        self.add_nu = add_nu
        
        # self.parms = FixedParameters(add_nu=add_nu)
        
    def build(self, input_shape):
    
        self.nu = self.add_weight(name='Nu',
                        shape = (),
                        initializer = 'ones',
                        trainable = self.add_nu,
                        constraint = keras.constraints.NonNeg())
    
        super().build(input_shape) 
        
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
        
        x = keras.Input(shape=(in_shape,) )
        
        self.build((None,in_shape))
        
        return keras.Model(inputs=[x], outputs=self.call(x))

    def update_mask(self, *args):
        
        pass

class FixedParameters(keras.layers.Layer):
    
    def __init__(self, add_nu=False):
        
        super().__init__()
        
        self.nu = self.add_weight(
            name="viscosity",
            shape=(),
            initializer="ones",
            trainable=add_nu,
        )
        
class FCNClass(BaseModel):

    def __init__(self,
                 n_neurons,
                 n_layers,
                 kernel_initializer = 'GlorotNormal',
                 values = [1e0]*5,
                 activation    = 'sine',
                 add_nu=False,
                 laaf=0,
                 dropout=0,
                 normalization=1,
                 w0=30.0,
                 stddev=None,
                 **kwargs
                 ):
                 
        super().__init__(add_nu=add_nu, **kwargs)
        
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        
        self.dropout    = dropout
        self.normalization = normalization
        
        self.i          = 0
            
        # self.emb = Embedding(n_neurons=n_neurons,
        #                      kernel_initializer=kernel_initializer,
        #                      w0=w0)
        
        # self.emb = PositionalEncoding(num_freqs=16)
        # self.emb = HybridEmbedding(n_neurons, pe_freqs=32, rff_features=64)
        
        self.emb = GaussianRFF(n_neurons)
        
        layers = []
        for _ in range(n_layers):
            
            layer = LaafLayer(n_neurons,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              w0=1.0,
                              )
            
            layers.append(layer)
        
        self.laaf_layers = layers
        
        if laaf:
            self.alphas = self.add_weight(
                            name="alphas",
                            shape=(n_layers+1, ),
                            initializer="zeros",
                            # constraint = tf.keras.constraints.NonNeg(),
                            trainable=True,   
                        )
        else:
            self.alphas = tf.zeros( (n_layers+1, ), name="alphas" )
        
        
        if self.dropout:
            
            layers = []
            for _ in range(n_layers+1):
                
                drop_layer = DropoutLayer()
                layers.append(drop_layer)
            
            self.dropout_layers = layers
        
        # if self.normalization:
        #
        #     layers = []
        #     for _ in range(n_layers+1):
        #
        #         layer = keras.layers.BatchNormalization(axis=0)
        #         layers.append(layer)
        #
        #     self.norm_layers = layers
        
        self.linear_layers = None
        
        self.scale = None
    
    def update_mask(self, n):
    
        if self.dropout:
            for drop_layer in self.dropout_layers:
                drop_layer.update_mask(n)
    
        
class resPINN(FCNClass):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 use_helmholtz= 1,
                 values = [1e0]*5,
                 **kwargs
                 ):
                 
        super().__init__(n_neurons,
                        n_layers, **kwargs)
        
        self.use_helmholtz = use_helmholtz
        
        add_bias = True
        
        if use_helmholtz:
            n_outs = 4
            add_bias = False
            
        # layers = []
        # for _ in range(n_layers):
        #     layer = Linear(n_outs,
        #                     kernel_initializer = 'zeros',
        #                    )
        #
        #     layers.append(layer)
        #
        # self.linear_layers = layers 
        
        self.linear_layer = Linear(n_outs,
                            kernel_initializer = 'GlorotNormal',
                            add_bias=add_bias,
                           )
        
        self.scale = Scaler(values, add_bias=add_bias)
        
    def _call(self, inputs, training=False):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        
        # inputs = tf.constant(2*np.pi)*inputs
        u = self.emb(inputs, self.alphas[0])
        
        # if self.normalization:
        #     u = self.norm_layers[0](u, training=training)
        
        if self.dropout:
            u = self.dropout_layers[0](u)
        
        u = self.laaf_layers[0](u, self.alphas[1])
        #
        if self.dropout:
            u = self.dropout_layers[1](u)
        
        # output = self.linear_layers[0](u)
        
        # output = 0.0
        
        for i in range(1,self.n_layers):
            
            out = self.laaf_layers[i](u, self.alphas[i+1])
            
            # if self.normalization:
            #     u = self.norm_layers[i+1](u, training=training)
                
            if self.dropout:
                out = self.dropout_layers[i+1](out)
            
            u = (u + out)
            
            # output = ( output + self.linear_layers[i](u) )
        
        output = self.linear_layer(u)
        
        output = self.scale(output)
        
        return(output)
    
    def call(self, inputs, training=False):
        
        if self.use_helmholtz:
            uvw = self.compute_velocity_helmholtz(inputs)
        else:
            uvw = self._call(inputs)
            
        return uvw
            
    def compute_velocity_helmholtz(self, inputs):
        """
        Compute u = curl(A) + grad(phi) given A (batch,3), phi (batch,1), and x (batch,3).
        """
        
        t = inputs[:, :1]
        z = inputs[:, 1:2]
        x = inputs[:, 2:3]
        y = inputs[:, 3:4]
        
        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            
            P = self._call( tf.concat([t, z, x, y], axis=1) )
            
            A1, A2, A3, phi = P[:,0:1], P[:,1:2], P[:,2:3], P[:,3:4]
            
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

class resiPINN(FCNClass):

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
            layer = Linear(n_neurons,
                           )
            
            layers.append(layer)
        
        self.linear_layers = layers 
        
        # self.gaussian_layer  = Linear(n_neurons,
        #                               add_bias=True,
        #                            )
        
        self.linear_output  = Linear(n_outs,
                                     kernel_initializer="zeros",
                                     add_bias=True,
                                   )
    
    def call(self, inputs, training=False):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        u = self.emb(inputs)
            
        if self.dropout:
            u = self.dropout_layers[0](u)
        
        u = self.laaf_layers[0](u, self.alphas[0])
                
        if self.dropout:
            u = self.dropout_layers[1](u)
        
        u = self.linear_layers[0](u)
            
        for i in range(1, self.n_layers):
            
            v = self.laaf_layers[i](u, self.alphas[i])
                
            if self.dropout:
                v = self.dropout_layers[i+1](v)
            
            v = self.linear_layers[i](v)
            
            u = u + v
        
        #Apply a Gaussian kernel
        # u = self.gaussian_layer(u)
        # u = gaussian_activation(u)
        
        output = self.linear_output(u)
        
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

class ResBlock(keras.Model):
    
    def __init__(self):
        
        pass
    
    def build(self, input_shape):
        
        pass
    
    def call(self, inputs):
        
        pass

class iPINN(FCNClass):

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
        
        
    def call(self, inputs, training=False):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        
        # inputs = tf.constant(2*np.pi)*inputs
        
        u = self.emb(inputs)
        
        # if self.normalization:
        #     u = self.norm_layers[0](u, training=training)
            
        if self.dropout:
            u = self.dropout_layers[0](u)
        
        
            
        # u = self.laaf_layers[0](u, self.alphas[0])
        #
        # if self.dropout:
        #     u = self.dropout_layers[1](u)
        
        output = tf.constant(0.0)#self.linear_layers[0](u)
        
        for i in range(0,self.n_layers):
            
            u = self.laaf_layers[i](u, self.alphas[i])
            
            # if self.normalization:
            #     u = self.norm_layers[i+1](u, training=training)
                
            if self.dropout:
                u = self.dropout_layers[i+1](u)
            
            output = output + self.linear_layers[i](u)
        
        # output = self.linear(output)#, alpha=self.alphas[i+2])
        
        output = self.scale(output)
        
        return(output)