'''
Created on 14 Aug 2024

@author: radar
'''
import numpy as np
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
                        initializer = 'zeros',
                        trainable = self.add_nu,
                        #constraint = keras.constraints.NonNeg()
                        )
    
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
        
        self.emb = GaussianRFF(n_neurons, name="GaussianLayer")
        
        layers = []
        for i in range(n_layers):
            
            layer = LaafLayer(n_neurons,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              w0=1.0,
                              name="laaf%02d" %i,
                              )
            
            layers.append(layer)
        
        self.laaf_layers = layers
        
        ini_alphas = np.zeros(n_layers+1) 
        # ini_alphas = 1.0*np.arange(n_layers+1)/(n_layers+1)
        
        if laaf:
            trainable = True
        else:
            trainable = False
            
        self.alphas = self.add_weight(
                        name="alphas",
                        shape=(n_layers+1, ),
                        initializer=tf.constant_initializer(ini_alphas), 
                        #initializer="zeros",
                        # constraint = keras.constraints.NonNeg(),
                        trainable=trainable,   
                    )
        # else:
        #     self.alphas = tf.cast(ini_alphas, dtype=tf.float32) # tf.zeros( (n_layers+1, ), name="alphas" )
        
        
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
    
class genericPINN(FCNClass):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 use_helmholtz= 0,
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
        
        self.linear_layer = Linear(n_outs,
                            kernel_initializer = 'GlorotNormal',
                            add_bias=add_bias,
                            name="FinalrMixer",
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
        u = self.emb(inputs)

        
        if self.dropout:
            u = self.dropout_layers[0](u)
        
        u = self.laaf_layers[0](u, self.alphas[0])
        #
        if self.dropout:
            u = self.dropout_layers[1](u)
        
        out = 0
        for i in range(1,self.n_layers):
            
            u = self.laaf_layers[i](u, self.alphas[i])
                
            if self.dropout:
                u = self.dropout_layers[i+1](u)
            
            # out = out + u
            
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
    
class resPINN(FCNClass):

    def __init__(self,
                 n_outs,
                 n_neurons,
                 n_layers,
                 use_helmholtz= 0,
                 values=None,
                 **kwargs
                 ):
                 
        super().__init__(n_neurons,
                        n_layers, **kwargs)
        
        self.use_helmholtz = use_helmholtz
        
        add_bias = True
        values = [1e0]*5
        
        if use_helmholtz:
            n_outs = 4
            add_bias = False
            values = [1e-2]*5
            
        layers = []
        for i in range(n_layers):
            
            layer = Linear(n_outs,
                            # activation="sine",
                            # kernel_initializer = 'zeros',
                            # kernel_initializer = "identity",
                            add_bias=True,
                            name = "Linear%02d" %i,
                           )
        
            layers.append(layer)
        
        self.linear_layers = layers 
        
        self.linear_layer = Linear(n_outs,
                            # kernel_initializer = 'GlorotNormal',
                            add_bias=add_bias,
                            name="FinalrMixer",
                           )
        
        self.scale = Scaler(values, add_bias=add_bias)
    
    
    def build(self, input_shape):
        
        super().build(input_shape) 
        
        self.gates = self.add_weight(
            name="gates",
            shape=(self.n_layers+1,),
            initializer="zeros",
            trainable=False,
        )
        
        self.step_count = tf.Variable(0.0, trainable=False)
    
    def update_mask(self, total_epochs):
        
        super().update_mask(total_epochs)
        
        self.step_count.assign_add(1.0)
    
        if True: #self.dropout:
            alpha = tf.ones(self.n_layers+1, dtype=tf.float32)
            self.gates.assign(alpha)
            
            return
        
        # Fraction of training completed
        progress = self.step_count / total_epochs  
    
        # Gates should linearly open between 0.0 → 0.9
        # Each gate i opens at (i / n_layers) * 0.9
        positions = tf.linspace(0.0, 0.8, self.n_layers+1)
    
        # Linear ramp: 0 → 1 over a small window (say 10% / n_layers)
        window = 0.1 / self.n_layers
        alpha = (progress - positions) / window
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
    
        # Force the very first gate to be always open
        alpha = tf.concat([[1.0, 1.0, 1.0], alpha[3:]], axis=0)
    
        self.gates.assign(alpha)
        
    def _call(self, inputs, training=False):
        '''
        inputs    :    [batch_size, dimensions] 
                        d0    :    time
                        d1    :    altitude
                        d2    :    x
                        d3    :    y
        '''
        
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
        
        output = self.linear_layers[0](u)
        
        # output = 0.0
        
        for i in range(1,self.n_layers):
            
            u = self.laaf_layers[i](u, self.alphas[i+1])
            
            # if self.normalization:
            #     u = self.norm_layers[i+1](u, training=training)
                
            if self.dropout:
                u = self.dropout_layers[i+1](u)
            
            output = output + self.gates[i+1]*self.linear_layers[i](u)/self.n_layers
        
        output = self.linear_layer(output)
        
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
        Compute velocity field from Helmholtz decomposition:
        u = curl(A) + grad(phi)
        
        inputs: (batch, 4) [t, z, x, y]
        returns: (batch, 3) [u, v, w]
        """
    
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
    
            # network outputs: [A1, A2, A3, phi]
            P = self._call(inputs)
            A = P[:, 0:3]   # vector potential
            phi = P[:, 3:4] # scalar potential
    
        # compute ∂A/∂x_j (batch, 3, 4)
        J_A = tape.batch_jacobian(A, inputs)  # derivatives of A wrt [t,z,x,y]
    
        # compute ∂phi/∂x_j (batch, 1, 4)
        J_phi = tape.batch_jacobian(phi, inputs)
    
        del tape
    
        # indices: 0=t, 1=z, 2=x, 3=y
        # curl(A) = (∂A3/∂y - ∂A2/∂z,
        #            ∂A1/∂z - ∂A3/∂x,
        #            ∂A2/∂x - ∂A1/∂y)
    
        A1_y = J_A[:,0,3]
        A1_z = J_A[:,0,1]
        A2_x = J_A[:,1,2]
        A2_z = J_A[:,1,1]
        A3_x = J_A[:,2,2]
        A3_y = J_A[:,2,3]
    
        u_rot = A3_y - A2_z
        v_rot = A1_z - A3_x
        w_rot = A2_x - A1_y
    
        # grad(phi)
        u_div = J_phi[:,0,2]
        v_div = J_phi[:,0,3]
        w_div = J_phi[:,0,1]
    
        u = tf.stack([u_rot + u_div,
                      v_rot + v_div,
                      w_rot + w_div], axis=1)
    
        return u*1e3

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
        for i in range(n_layers):
            layer = Linear(n_neurons,
                           name="LinearMixer%02d" %i,
                           )
            
            layers.append(layer)
        
        self.linear_layers = layers 
        
        # self.gaussian_layer  = Linear(n_neurons,
        #                               add_bias=True,
        #                            )
        
        self.linear_output  = Linear(n_outs,
                                     kernel_initializer="zeros",
                                     add_bias=True,
                                     name="FinalMixer"
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


class resPINN_HH(BaseModel):
    """
    Helmholtz-decomposed PINN with two subnetworks:
      - net_A: learns vector potential A = (A1, A2, A3)
      - net_phi: learns scalar potential phi
    Velocity is reconstructed as:
        u = curl(A) + grad(phi)
    """

    def __init__(self,
                 n_neurons=128,
                 n_layers=5,
                 w0_A=1.0,
                 w0_phi=1.0,
                 dropout=0.0,
                 laaf=True,
                 learn_const=False,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.n_layers = n_layers
        self.dropout = dropout
        self.learn_const = learn_const

        # -------------------
        # Net A (rotational)
        # -------------------
        self.emb_A = GaussianRFF(n_neurons, name="Emb_A")
        
        self.laaf_A = [
                        LaafLayer(n_neurons, activation="sine", w0=w0_A, name=f"laafA_{i}")
                        for i in range(n_layers)
                    ]
        
        # self.proj_A = [
        #     Linear(n_neurons, name=f"projA_{i}") for i in range(n_layers)
        # ]
        
        self.res_A = self.add_weight(
                                        name="residual_A",
                                        shape=(self.n_layers,),
                                        initializer=tf.constant_initializer(-2),
                                        trainable=True,
                                        # constraint=keras.constraints.NonNeg,
                                    )
        
        self.out_A = Linear(4, add_bias=False, name="out_A")

        if learn_const:
            # self.c = tf.Variable(tf.zeros(3), trainable=True, name="c_harmonic")
            
            initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)
            
            self.c_model = keras.models.Sequential([
                                              keras.layers.Dense(4, activation=tf.sin),#, kernel_initializer=initializer),
                                              keras.layers.Dense(4, activation=tf.sin),#, kernel_initializer=initializer),
                                              # keras.layers.Dense(8, activation=tf.sin),
                                              keras.layers.Dense(3)
                                            ])
            
        # -------------------
        # # Net φ (irrotational)
        # # -------------------
        # self.emb_phi = GaussianRFF(n_neurons, name="Emb_phi")
        # self.laaf_phi = [
        #     LaafLayer(n_neurons, activation="sine", w0=w0_phi, name=f"laafPhi_{i}")
        #     for i in range(n_layers)
        # ]
        # # self.proj_phi = [
        # #     Linear(n_neurons, name=f"projPhi_{i}") for i in range(n_layers)
        # # ]
        # self.out_phi = Linear(1, add_bias=False, name="out_phi")

        # Dropout layers if needed
        self.dropout_layers_A = None
        if dropout > 0:
            self.dropout_layers_A = [DropoutLayer() for _ in range(n_layers)]
            # self.dropout_layers_phi = [DropoutLayer() for _ in range(n_layers)]

        # Trainable alphas for adaptive scaling of each block (optional)
        self.alphas = self.add_weight(
            name="alphas",
            shape=( (n_layers+1),),
            initializer=tf.constant_initializer(0.0),
            trainable=laaf,
        )
    
    def update_mask(self, total_epochs):
        
        if self.dropout_layers_A is not None:
            for drop_layer in self.dropout_layers_A:
                drop_layer.update_mask(total_epochs)
        
    def forward_A(self, inputs, training=False):
        
        emb = self.emb_A(inputs, self.alphas[0])
        u = emb
        
        alpha = self.res_A #0.5*(tf.tanh(self.res_A) + 1)
        
        for i, laaf in enumerate(self.laaf_A):
            
            # # inject embedding
            # if i > 0:
            #     h = tf.concat([u, emb], axis=-1)
            #     h = self.proj_A[i](h)
            # else:
            #     h = u
        
            h = laaf(u, self.alphas[i+1])
        
            if self.dropout > 0:
                h = self.dropout_layers_A[i](h, training=training)
                
            # skip connection: add input back
            u = u + alpha[i] * h #(h-u)
            
        return self.out_A(u)

    def forward_phi(self, inputs, training=False):
        
        emb = self.emb_phi(inputs, self.alphas[self.n_layers+1])
        u = emb
        
        for i, laaf in enumerate(self.laaf_phi):
            prev_u = u  # save incoming state
        
            h = laaf(u, self.alphas[self.n_layers+2+i])
        
            if self.dropout > 0:
                h = self.dropout_layers_phi[i](h, training=training)
        
            # skip connection: add input back
            u = prev_u + h
            
        return self.out_phi(u)

    def compute_velocity_helmholtz(self, inputs):
        """
        Compute velocity field:
        u = curl(A) + grad(phi)
        """
        
        t, z, x, y = tf.split(inputs, num_or_size_splits=4, axis=1)
        
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            
            A = self.forward_A( tf.concat([t, z, x, y], axis=1) )      # (B,3)
            # phi = self.forward_phi(inputs)  # (B,1)
            # A = A_phi[:,0:3]
            # phi = A_phi[:,3:4]
            
            A1 = A[:,0:1]
            A2 = A[:,1:2]
            A3 = A[:,2:3]

        # J_A = tape.batch_jacobian(A, inputs)    # (B,3,4)
        # J_phi = tape.batch_jacobian(phi, inputs)  # (B,1,4)
        
        # A1_x = tape.gradient(A1, x)
        A1_y = tape.gradient(A1, y)
        A1_z = tape.gradient(A1, z)
        
        A2_x = tape.gradient(A2, x)
        # A2_y = tape.gradient(A2, y)
        A2_z = tape.gradient(A2, z)
        
        A3_x = tape.gradient(A3, x)
        A3_y = tape.gradient(A3, y)
        # A3_z = tape.gradient(A3, z)
        
        # u_div = tape.gradient(phi, x)
        # v_div = tape.gradient(phi, y)
        # w_div = tape.gradient(phi, z)
        
        del tape

        # Indices: 0=t, 1=z, 2=x, 3=y
        # A1_y, A1_z = J_A[:, 0, 3], J_A[:, 0, 1]
        # A2_x, A2_z = J_A[:, 1, 2], J_A[:, 1, 1]
        # A3_x, A3_y = J_A[:, 2, 2], J_A[:, 2, 3]

        # curl(A)
        u_rot = A3_y - A2_z
        v_rot = A1_z - A3_x
        w_rot = A2_x - A1_y

        # # grad(phi)
        # u_div = J_phi[:, 0, 2]   # ∂phi/∂x
        # v_div = J_phi[:, 0, 3]   # ∂phi/∂y
        # w_div = J_phi[:, 0, 1]   # ∂phi/∂z
        
        u = u_rot #+ u_div
        v = v_rot #+ v_div
        w = w_rot #+ w_div
        
        uvw = tf.concat([u, v, w], axis=1)
        
        if self.learn_const:
            uvw_c = self.c_model(inputs[:,0:1])
            uvw = uvw + uvw_c
        
        return uvw

    def call(self, inputs, training=False):
        
        # return self.compute_velocity_helmholtz(inputs)
        
        A_phi = self.forward_A( inputs )
        
        h = tf.constant(0.0)
        if self.learn_const:
            h = self.c_model(inputs[:,0:1])
            
        return A_phi, h
        