'''
Created on 14 Aug 2024

@author: radar
'''
# import numpy as np
import tensorflow as tf
import keras

from pinn.layers import BaseModel, Embedding, LaafLayer, DropoutLayer, Scaler, Linear

# Define the Gaussian activation function
def gaussian_activation(x, alpha=1.0):
    
    return tf.exp(-alpha * tf.square(x))

class FCNClass(BaseModel):

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
                 normalization=1,
                 stddev=None,
                 **kwargs
                 ):
                 
        super().__init__(add_nu=add_nu, **kwargs)
        
        self.n_out      = n_outs
        self.n_neurons  = n_neurons
        self.n_layers   = n_layers
        
        self.dropout    = dropout
        self.normalization = normalization
        
        self.i          = 0
            
        self.emb = Embedding(n_neurons=n_neurons, kernel_initializer=kernel_initializer) #, stddev=stddev)
        # self.emb = PositionalEncoding(n_neurons, kernel_initializer=kernel_initializer, stddev=stddev)
        
        layers = []
        for i in range(n_layers):
            
            if i == (n_layers - 1):
                activation_ = gaussian_activation
            else:
                activation_ = activation
            
            layer = LaafLayer(n_neurons,
                              activation=activation_,
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
        
        self.linear_output  = Linear(n_outs,
                                     kernel_initializer="zeros",
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