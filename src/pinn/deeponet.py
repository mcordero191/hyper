'''
Created on 14 Aug 2024

@author: radar
'''

import numpy as np

import tensorflow as tf
import keras

from pinn.layers import BaseModel, Scaler, Embedding, Densenet, StackLayer, Linear, EinsumLayer
# from pinn.spinn import Shift, GaussianKernel

data_type     = tf.float32

class PCALayer(keras.layers.Layer):

    def __init__(self,
                 kernel_initializer = 'zeros',
                 constraint = None,
                 **kwargs
                 ):
        
        super().__init__()
        self.kernel_initializer = kernel_initializer
        self.constraint = constraint
        
    def build(self, input_shape):
        
        ncoeffs = input_shape[-1]
        
        self.a = self.add_weight(
            shape=(ncoeffs, ),
            initializer = self.kernel_initializer,
            constraint = self.constraint, 
            trainable=True,
        )
        
    def call(self, input0, input1):
        '''
        Inputs [npoints, nnodes]
        
        Outputs [npoints] 
        '''
        
        u = tf.einsum('k,ik,ik->i', self.a, input0, input1)
        
        return(u)
    
class DeepONet(BaseModel):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 n_coeffs,
                 kernel_initializer = 'LecunNormal',
                 zero_initializer = 'zeros',
                 values = [1e0,1e0,1e0],
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
        
        # self.t_emb = Shift(n_nodes=n_neurons, name='t_emb', reduce=False)
        # self.fnn_u = GaussianKernel()
        
        self.t_emb = Embedding(n_neurons=n_neurons, name='t_emb', kernel_initializer=kernel_initializer)#, stddev=stddev)
        
        self.trunk = Densenet(n_neurons=n_neurons, n_layers=n_layers-1, activation=activation, name='fnn_tu', kernel_initializer=kernel_initializer)
        # self.fnn_v = Densenet(n_neurons=n_coeffs, n_layers=n_layers-1, activation=activation, name='fnn_tv', kernel_initializer=kernel_initializer)
        # self.fnn_w = Densenet(n_neurons=n_coeffs, n_layers=n_layers-1, activation=activation, name='fnn_tw', kernel_initializer=kernel_initializer)
        
        # self.t_u = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='linear_tu', kernel_initializer=zero_initializer)
        # self.t_v = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='linear_tv', kernel_initializer=zero_initializer)
        # self.t_w = Densenet(n_neurons=n_coeffs, n_layers=1, activation=None, name='linear_tw', kernel_initializer=zero_initializer)
        
        self.t_u = Linear(n_coeffs, kernel_initializer = zero_initializer, name='linear_tu')
        self.t_v = Linear(n_coeffs, kernel_initializer = zero_initializer, name='linear_tv')
        self.t_w = Linear(n_coeffs, kernel_initializer = zero_initializer, name='linear_tw')
        
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
        
        # activation = "sine"
        
        # self.x_emb = Shift(n_nodes=n_neurons, name='x_emb', reduce=True)
        # self.x_emb = Densenet(n_neurons=n_neurons, activation=None, name='x_emb', kernel_initializer=kernel_initializer)
        self.x_emb = Embedding(n_neurons=n_neurons, name='x_emb', kernel_initializer=kernel_initializer)#, stddev=stddev)
        # self.x_fnn = Densenet(n_neurons=n_coeffs, n_layers=n_layers, activation=activation, name='x_dense', kernel_initializer=kernel_initializer)
        
        self.branch = Densenet(n_neurons=n_neurons, n_layers=n_layers-1, activation=activation, name='x_u', kernel_initializer=kernel_initializer)
        # self.x_v = Densenet(n_neurons=n_coeffs, n_layers=n_layers-1, activation=activation, name='x_v', kernel_initializer=kernel_initializer)
        # self.x_w = Densenet(n_neurons=n_coeffs, n_layers=n_layers-1, activation=activation, name='x_w', kernel_initializer=kernel_initializer)
        
        self.x_u = Linear(n_coeffs, kernel_initializer = kernel_initializer, name='linear_xu')
        self.x_v = Linear(n_coeffs, kernel_initializer = kernel_initializer, name='linear_xv')
        self.x_w = Linear(n_coeffs, kernel_initializer = kernel_initializer, name='linear_xw')
        
        # self.pca_u   = PCALayer()
        # self.pca_v   = PCALayer()
        # self.pca_w   = PCALayer()
        
        self.einsum = EinsumLayer()
        self.stack = StackLayer()
        
        self.scale = Scaler(values)
        
    def call(self, inputs):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
        inputs = tf.constant(np.pi)*inputs
        
        #e = tf.einsum('ij,jk->ik', m0, m1)
        te = self.t_emb(inputs[:,0:2])
        
        trunk_output = self.trunk(te)
        # v_t = self.fnn_v(te)
        # w_t = self.fnn_w(te)
        
        u_t = self.t_u(trunk_output)
        v_t = self.t_v(trunk_output)
        w_t = self.t_w(trunk_output)
        
        xe = self.x_emb(inputs[:,2:4])
        
        branch_output = self.branch(xe)
        # v_x = self.x_v(xe)
        # w_x = self.x_w(xe)
        
        u_x = self.x_u(branch_output)
        v_x = self.x_v(branch_output)
        w_x = self.x_w(branch_output)
        
        # u = tf.einsum('ik,ik->i', u_x, u_t, name='einsum_u')
        # v = tf.einsum('ik,ik->i', v_x, v_t, name='einsum_v')
        # w = tf.einsum('ik,ik->i', w_x, w_t, name='einsum_w')
        
        # u = self.pca_u(u_x, u_t)
        # v = self.pca_v(u_x, u_t)
        # w = self.pca_w(u_x, u_t)
        
        u = self.einsum(u_t, u_x)
        v = self.einsum(v_t, v_x)
        w = self.einsum(w_t, w_x)
        
        uvw = self.stack(u,v,w)
        
        uvw = self.scale(uvw)
        
        return(uvw)