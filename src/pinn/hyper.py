"""
PINN multifidelity

********************************************************************************
Author  :    Miguel Urco
Based on: Shota DEGUCHI
        Yosuke SHIBATA
        Structural Analysis Laboratory, Kyushu University (Jul. 19th, 2021)
        
        
********************************************************************************
"""

import os
import time
import datetime
import numpy as np

import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf

data_type     = tf.float32
# tf.keras.backend.set_floatx('float64')

time_base = 1.468e9

from georeference.geo_coordinates import lla2xyh
from utils.histograms import ax_2dhist
from utils.plotting import epoch2num
from pinn.nn_architectures import iPINN, rPINN, resPINN, sPINN, DeepONet, DeepONetOpt, DeepONetOri,\
    MultiNet, DeepMultiNet
    
from pinn.bfgs import LBFGS

class App:
    
    ext_forcing = None
    
    def __init__(self,
                shape_in  = 4,
                shape_out = 3,
                width   = 8,
                depth   = 16, 
                nnodes  = 50,
                w_init  = 'HeNormal',
                b_init  = "zeros",
                act     = "sine",
                init_sigma = 1.0,
                # lr      = 1e-3,
                # opt     = "Adam",
                laaf    = 0,
                # w_data  = 1.,
                # w_div   = 1e0,
                # w_mom   = 1e0,
                # w_temp  = 1e0,
                # w_srt   = 1.0,
                # ns_pde  = 10000,
                NS_type     = 'VP',             #Velocity-Vorticity (VV) or Velocity-Pressure (VP) form
                # dropout     = True,
                r_seed  = 1234,
                f_scl_in    = "minmax",                   #Input feature scaling
                f_scl_out   = 1,                          #Output scaling
                msis        = None,
                lon_ref     = None,
                lat_ref     = None,
                alt_ref     = None,
                nn_type     = 'deeponet',
                nblocks     = 3,
                # filename_mean = '',
                residual_layer=False,
                **kwargs
                ):
        
        """
        Inputs:
            d     :    Doppler frequency
            dd    :    Doppler frequency standard deviation
            k     :    Bragg vector
            
            msis    :    MSIS model outputs as a function of altitude.
                            -Neutral density
                            -Neutral density change rate
                            -Kinematic viscosity
                            -Brunt-Vaisala frequency
        """

        # configuration
        self.shape_in    = shape_in
        self.shape_out   = shape_out
        
        self.width   = width
        self.depth   = depth
        self.nnodes  = nnodes
        self.nblocks = nblocks
        self.nn_type    = str.lower(nn_type)
        
        self.u_width = width//self.shape_out
        
        self.act     = act
            
        self.w_init  = w_init
        self.b_init  = b_init
        
        # self.lr      = lr
        # self.opt     = opt
        
        self.f_scl_in   = f_scl_in
                
        self.laaf       = laaf
        # self.dropout    = dropout
        self.init_sigma = init_sigma
        
        # self.w_data  = w_data
        # self.w_div   = w_div
        # self.w_mom   = w_mom
        # self.w_temp  = w_temp
        # self.w_srt   = w_srt
        #
        # self.N_pde   = ns_pde    #number of samples
        
        self.r_seed  = r_seed
        self.random_seed(self.r_seed)
        
        self.msis       = msis
        
        self.lon_ref    = lon_ref
        self.lat_ref    = lat_ref
        self.alt_ref    = alt_ref
        
        self.NS_type    = NS_type
        
        add_nu = False
        if NS_type in ['VV', 'VV_hydro', 'VP']:
            add_nu = True
        
        self.residual_layer = residual_layer
        
        self.__ini_pde      = True
        
        # dataset
        self.t = None
        self.x = None
        self.y = None
        self.z = None
        
        self.d  = None
        self.k  = None
        
        #Bounds
        self.lbi = None
        self.ubi = None
        self.mn = None
        
        self.X_pde = None
        
        # hiddens_shape = (self.depth-1) * [self.width]
        
        # build
        if self.nn_type == 'rpinn':            
            nn = rPINN( self.shape_out,
                              self.width,
                              self.depth,
                              activation  = act,
                              add_nu     = add_nu,
                              kernel_initializer = w_init
                            )
        
        elif self.nn_type == 'ipinn':            
            nn = iPINN( self.shape_out,
                              self.width,
                              self.depth,
                              activation  = act,
                              add_nu     = add_nu,
                              kernel_initializer = w_init
                            )    
        elif self.nn_type == 'respinn':            
            nn = resPINN( self.shape_out,
                              self.width,
                              self.depth,
                              activation  = act,
                              add_nu     = add_nu,
                              kernel_initializer = w_init,
                              laaf = laaf,
                            )    
        elif self.nn_type == 'spinn':            
            nn = sPINN( self.shape_out,
                              self.width,
                              self.depth,
                              self.nnodes,
                              activation  = act,
                              add_nu     = add_nu,
                              kernel_initializer = w_init
                            )
            
        elif self.nn_type == 'deeponet':
            nn = DeepONet(self.shape_out,
                          self.width,
                          self.depth,
                          self.nnodes,
                          activation  = act,
                          kernel_initializer = w_init,
                          add_nu = add_nu,
                          )
        elif self.nn_type == 'deeponetori':
            nn = DeepONetOri(self.shape_out,
                          self.width,
                          self.depth,
                          self.nnodes,
                          activation  = act,
                          kernel_initializer = w_init,
                          add_nu = add_nu,
                          )
        elif self.nn_type == 'deeponetopt':
            nn = DeepONetOpt(self.shape_out,
                          self.width,
                          self.depth,
                          self.nnodes,
                          activation  = act,
                          kernel_initializer = w_init,
                          add_nu = add_nu,
                          )
        elif self.nn_type == 'multinet':
            nn = MultiNet(self.shape_out,
                          self.width,
                          self.depth,
                          self.nnodes,
                          activation  = act,
                          kernel_initializer = w_init,
                          add_nu = add_nu,
                          residual_layer = residual_layer,
                          )
        elif self.nn_type == 'deepmultinet':
            nn = DeepMultiNet(self.shape_out,
                              n_neurons=self.width,
                              n_layers =self.depth,
                              n_filters=self.nnodes,
                              n_blocks=self.nblocks,
                              activation  = act,
                              kernel_initializer = w_init,
                              add_nu = add_nu,
                              )
            
        else:
            raise ValueError('nn_type not recognized')
        
        model = nn.build_graph( (self.shape_in,) )
        model.summary()
        
        self.model = nn
        
        self.counter = 0
        print("\n************************************************************")
        print("****************     MAIN PROGRAM START     ****************")
        print("************************************************************")
        print(">>>>> start time:", datetime.datetime.now())
        print(">>>>> configuration;")
        print("         Navier-Stokes:", self.NS_type)
        # print("         n_pde_samples:", self.N_pde)
        print("         random seed  :", self.r_seed)
        print("         data type    :", data_type)
        print("         activation   :", self.act)
        print("         weight init  :", self.w_init)
        print("         bias   init  :", self.b_init)
        print("         laaf         :", self.laaf)
        # print("         learning rate:", self.lr)
        # print("         optimizer    :", self.opt)
        print("         width        :", self.width)
        print("         depth        :", self.depth)
        print("         nn_type      :", self.nn_type)
        # print("         total parms  :", n_total_parms)
        print("         feature scaling :", self.f_scl_in)
        

    def random_seed(self, seed = 1234):
        
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def opt_(self, lr, opt, epochs=None):
        
        # if epochs is not None:
        #     lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([int(0.4*epochs),int(0.8*epochs)],
        #                                                           [lr,0.33*lr,0.1*lr])
        
        if opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate = lr, momentum = 0.0, nesterov = False
                )
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False
                )
        elif opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False
                )
        elif opt == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999
                )
        elif opt == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999
                )
        else:
            raise NotImplementedError(">>>>> opt_")
        return optimizer
    
    # @tf.function
    def mean_grads(self, grads):
            
        # average the grads
        n = tf.ones((), dtype=tf.int32)
        mean_grad = tf.zeros((), dtype=data_type)
        
        for grad in grads:
            
            if grad is None:
                continue
        
            n += tf.reduce_prod(grad.shape)
            mean_grad += tf.reduce_sum(tf.abs(grad))
        
        n = tf.cast(n, dtype=data_type)
        mean_grad = tf.truediv(mean_grad, n)
        
        return(mean_grad)
    
    @tf.function
    def forward_pass(self, model, X, training=False, return_all=False):
        
        # print("Tracing model!") 
        
        z = 2. * (X - self.lbi) / (self.ubi - self.lbi)  - 1.
        
        u = model(z, training=training)
        
        # print("Finish tracing model!")
        
        return(u)
    
    # @tf.function
    def pde_div_mean(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        with tf.GradientTape(persistent = False, watch_accessed_variables=False) as tp:
            tp.watch(z)
            
            u, ux, uy = self.forward_pass(model, tf.concat([t, z, x, y], axis=1), return_all=True)
            
            # u = uvw[:,0:1]
            # v = uvw[:,1:2]
            w = u[:,2:3]
        
        u_x = ux[:,0:1]
        v_y = uy[:,1:2]
        
        # w_x = ux[:,2:3]
        # w_y = uy[:,2:3]
        
        w_z = tp.gradient(w, z)
        
        div     = eq_continuity(u_x, v_y, w_z, w=w, rho_ratio=rho_ratio)
        junk     = tf.constant(0.0, dtype=data_type)
            
        return(div, junk, junk, junk, junk, junk)
    
    def pde_div(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
            tp.watch(x)
            tp.watch(y)
            tp.watch(z)
            
            outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
            
            u = outs[:,0:1]
            v = outs[:,1:2]
            w = outs[:,2:3]
        
        u_x = tp.gradient(u, x)
        v_y = tp.gradient(v, y)
        w_z = tp.gradient(w, z)
        
        del tp
        
        div     = eq_continuity(u_x, v_y, w_z, w=w, rho_ratio=rho_ratio)
        junk     = tf.constant(0.0, dtype=data_type)
            
        return(div, junk, junk, junk, junk, junk)
    
    # @tf.function
    def pde_div_mom(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS2:
            
            tpS2.watch(x)
            tpS2.watch(y)
            tpS2.watch(z)
            
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                
                tp.watch(t)
                tp.watch(x)
                tp.watch(y)
                tp.watch(z)
                
                outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                
                u = outs[:,0:1]
                v = outs[:,1:2]
                w = outs[:,2:3]
                
                with tpS2.stop_recording():
                    p = outs[:,3:4]
                
            u_x = tp.gradient(u, x)
            u_y = tp.gradient(u, y)
            u_z = tp.gradient(u, z)
            
            v_x = tp.gradient(v, x)
            v_y = tp.gradient(v, y)
            v_z = tp.gradient(v, z)
            
            with tpS2.stop_recording():
                u_t = tp.gradient(u, t)
                v_t = tp.gradient(v, t)
                
                p_x = tp.gradient(p, x)
                p_y = tp.gradient(p, y)
                
                w_z = tp.gradient(w, z)
            
            del tp
            
        u_xx = tpS2.gradient(u_x, x)
        u_yy = tpS2.gradient(u_y, y)
        u_zz = tpS2.gradient(u_z, z)
        
        v_xx = tpS2.gradient(v_x, x)
        v_yy = tpS2.gradient(v_y, y)
        v_zz = tpS2.gradient(v_z, z)
        
        del tpS2
        
        div     = eq_continuity(u_x, v_y, w_z)#, w=w, rho_ratio=rho_ratio)
        
        mom_x   = eq_horizontal_momentum(u, v, w,
                                       u_t, u_x, u_y, u_z,
                                       u_xx, u_yy, u_zz,
                                       p_x,
                                       F=self.ext_forcing, nu=nu)
        
        mom_y   = eq_horizontal_momentum(u, v, w,
                                       v_t, v_x, v_y, v_z,
                                       v_xx, v_yy, v_zz,
                                       p_y,
                                       F=self.ext_forcing, nu=nu)
        
        # mom = mom_x + mom_y
        
        #Potential temperature
        junk = tf.constant(0.0, dtype=data_type)
            
        return(div, junk, mom_x, mom_y, junk, junk)
    
    # @tf.function
    def pde_div_mom_temp(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS2:
            
            tpS2.watch(x)
            tpS2.watch(y)
            tpS2.watch(z)
            
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                
                tp.watch(t)
                tp.watch(x)
                tp.watch(y)
                tp.watch(z)
                
                outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                
                u = outs[:,0:1]
                v = outs[:,1:2]
                w = outs[:,2:3]
                
                p = outs[:,3:4]
                theta = outs[:,4:5]
                
            u_x = tp.gradient(u, x)
            u_y = tp.gradient(u, y)
            u_z = tp.gradient(u, z)
            
            v_x = tp.gradient(v, x)
            v_y = tp.gradient(v, y)
            v_z = tp.gradient(v, z)
            
            w_x = tp.gradient(w, x)
            w_y = tp.gradient(w, y)
            w_z = tp.gradient(w, z)
                
            theta_x = tp.gradient(theta, x)
            theta_y = tp.gradient(theta, y)
            theta_z = tp.gradient(theta, z)
                
            with tpS2.stop_recording():
                
                u_t = tp.gradient(u, t)
                v_t = tp.gradient(v, t)
                w_t = tp.gradient(w, t)
                
                theta_t = tp.gradient(theta, t)
                
                p_x = tp.gradient(p, x)
                p_y = tp.gradient(p, y)
                p_z = tp.gradient(p, z)
            
            del tp
            
        u_xx = tpS2.gradient(u_x, x)
        u_yy = tpS2.gradient(u_y, y)
        u_zz = tpS2.gradient(u_z, z)
        
        v_xx = tpS2.gradient(v_x, x)
        v_yy = tpS2.gradient(v_y, y)
        v_zz = tpS2.gradient(v_z, z)
        
        w_xx = tpS2.gradient(w_x, x)
        w_yy = tpS2.gradient(w_y, y)
        w_zz = tpS2.gradient(w_z, z)
        
        theta_xx = tpS2.gradient(theta_x, x)
        theta_yy = tpS2.gradient(theta_y, y)
        theta_zz = tpS2.gradient(theta_z, z)
        
        del tpS2
        
        div     = eq_continuity(u_x, v_y, w_z)#, w=w, rho_ratio=rho_ratio)
        
        mom_x   = eq_horizontal_momentum(u, v, w,
                                       u_t, u_x, u_y, u_z,
                                       u_xx, u_yy, u_zz,
                                       p_x = p_x,
                                       F = self.ext_forcing,
                                       nu = nu,
                                       rho = rho)
        
        mom_y   = eq_horizontal_momentum(u, v, w,
                                       v_t, v_x, v_y, v_z,
                                       v_xx, v_yy, v_zz,
                                       p_x = p_y,
                                       F = self.ext_forcing,
                                       nu=nu,
                                       rho = rho)
        
        mom_z = eq_vertical_momentum(u, v, w,
                                   w_t, w_x, w_y, w_z,
                                   w_xx, w_yy, w_zz,
                                   p_z = p_z,
                                   theta = theta,
                                   F = self.ext_forcing,
                                   nu = nu,
                                   N = N,
                                   rho = rho)
        
        # mom = mom_x + mom_y + mom_z
        
        #Potential temperature
        temp = eq_temperature(u, v, w,
                              theta_t, theta_x, theta_y, theta_z,
                              theta_xx, theta_yy, theta_zz,
                              N=N, k=nu)
        
        junk = tf.constant(0.0, dtype=data_type)
        
        return(div, temp, mom_x, mom_y, mom_z, junk)
    
    # @tf.function
    def pde_vorticity_3O_div(self, model, t, z, x, y, nu, rho, rho_ratio, N):
                
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS:
            
            tpS.watch(x)
            tpS.watch(y)
            tpS.watch(z)
            
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                
                tp.watch(x)
                tp.watch(y)
                tp.watch(z)
                
                outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                
                u = outs[:,0:1]
                v = outs[:,1:2]
                w = outs[:,2:3]
            
            #### Divergence ###############
            # with tpS.stop_recording():
            u_x = tp.gradient(u, x)
            u_y = tp.gradient(u, y)
            u_z = tp.gradient(u, z)
            
            v_x = tp.gradient(v, x)
            v_y = tp.gradient(v, y)
            v_z = tp.gradient(v, z)
            
            w_x = tp.gradient(w, x)
            w_y = tp.gradient(w, y)
            w_z = tp.gradient(w, z)
            
            #Vorticity equals curl of velocity
            omegax  = w_y - v_z
            omegay  = u_z - w_x
            omegaz  = v_x - u_y
            
            del tp
        
        
        omegax_x = tpS.gradient(omegax, x)
        omegay_y = tpS.gradient(omegay, y)
        omegaz_z = tpS.gradient(omegaz, z)
        
        # omegax_y = tpS.gradient(omegax, y)
        # omegax_z = tpS.gradient(omegax, z)
        #
        # omegay_x = tpS.gradient(omegay, x)
        #
        # omegay_z = tpS.gradient(omegay, z)
        #
        # omegaz_x = tpS.gradient(omegaz, x)
        # omegaz_y = tpS.gradient(omegaz, y)
        #
        #
        # u_xx = tpS.gradient(u_x, x)
        # u_yy = tpS.gradient(u_y, y)
        # u_zz = tpS.gradient(u_z, z)
        #
        # v_xx = tpS.gradient(v_x, x)
        # v_yy = tpS.gradient(v_y, y)
        # v_zz = tpS.gradient(v_z, z)
        #
        # w_xx = tpS.gradient(w_x, x)
        # w_yy = tpS.gradient(w_y, y)
        # w_zz = tpS.gradient(w_z, z)
        
        ############################
        
        del tpS
        
        #Divergence
        div = eq_continuity(u_x, v_y, w_z)#, w=w, rho_ratio=rho_ratio)

        div_w = 1e6*eq_continuity(omegax_x, omegay_y, omegaz_z)
        
        # poi_x     = eq_poisson(u_xx, u_yy, u_zz, omegaz_y, omegay_z)
        # poi_y     = eq_poisson(v_xx, v_yy, v_zz, omegax_z, omegaz_x)
        # poi_z     = eq_poisson(w_xx, w_yy, w_zz, omegay_x, omegax_y)
        
        junk = tf.constant(0.0, dtype=data_type)
        
        return(div, div_w, junk, junk, junk, junk)
    
    # @tf.function
    def pde_vorticity_3O_noNu(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        # div     = tf.constant(0.0, dtype=data_type)
        # mom     = tf.constant(0.0, dtype=data_type)
        # div_w    = tf.constant(0.0, dtype=data_type)
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS:
            tpS.watch(t)
            
            tpS.watch(x)
            tpS.watch(y)
            # tpS.watch(z)
                
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                
                tp.watch(x)
                tp.watch(y)
                tp.watch(z)
                
                outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                
                u = outs[:,0:1]
                v = outs[:,1:2]
                w = outs[:,2:3]
            
            with tpS.stop_recording():
                u_x = tp.gradient(u, x)
                v_y = tp.gradient(v, y)
                w_z = tp.gradient(w, z)
            
            #Vorticity components
            u_y = tp.gradient(u, y)
            u_z = tp.gradient(u, z)
            v_x = tp.gradient(v, x)
            v_z = tp.gradient(v, z)
            w_x = tp.gradient(w, x)
            w_y = tp.gradient(w, y)
            
                
            del tp
            
            #Vorticity equals curl of velocity
            omegax  = w_y - v_z
            omegay  = u_z - w_x
            omegaz  = v_x - u_y
            
            a   = w*omegay - v*omegaz
            b   = u*omegaz - w*omegax
            
            Fx  = a #+ nu*p
            Fy  = b #+ nu*q
        
        #### Divergence ###############
        
        # # omegax_x = tpS.gradient(omegax, x)
        # omegax_y = tpS.gradient(omegax, y)
        # omegax_z = tpS.gradient(omegax, z)
        #
        # omegay_x = tpS.gradient(omegay, x)
        # # omegay_y = tpS.gradient(omegay, y)
        # omegay_z = tpS.gradient(omegay, z)
        #
        # omegaz_x = tpS.gradient(omegaz, x)
        # omegaz_y = tpS.gradient(omegaz, y)
        # # omegaz_z = tpS.gradient(omegaz, z)
        
        # u_xx = tpS.gradient(u_x, x)
        # u_yy = tpS.gradient(u_y, y)
        # u_zz = tpS.gradient(u_z, z)
        #
        # v_xx = tpS.gradient(v_x, x)
        # v_yy = tpS.gradient(v_y, y)
        # v_zz = tpS.gradient(v_z, z)
        #
        # w_xx = tpS.gradient(w_x, x)
        # w_yy = tpS.gradient(w_y, y)
        # w_zz = tpS.gradient(w_z, z)

        ############################
        omegaz_t    = tpS.gradient(omegaz, t)
                
        ###############################
        ####Rotational formulation #####
        ###############################
        Fx_y =   tpS.gradient(Fx, y)
        Fy_x =   tpS.gradient(Fy, x)
        
        del tpS
        
        junk = tf.constant(0.0, dtype=data_type)
        
        #Divergence
        div = eq_continuity(u_x, v_y, w_z, w=w, rho_ratio=rho_ratio)
        
        div_w = junk#eq_continuity(omegax_x, omegay_y, omegaz_z)
        
        poi_x     = junk#eq_poisson(u_xx, u_yy, u_zz, omegaz_y, omegay_z)
        poi_y     = junk#eq_poisson(v_xx, v_yy, v_zz, omegax_z, omegaz_x)
        poi_z     = junk#eq_poisson(w_xx, w_yy, w_zz, omegay_x, omegax_y)
        
        ## Rotational formulation
        mom_z   = eq_vorticity_RF_F(omegaz_t, Fy_x, Fx_y)
        
        return(div, mom_z, poi_x, poi_y, poi_z, junk)

    # @tf.function
    def pde_vorticity_3O(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        # div     = tf.constant(0.0, dtype=data_type)
        # mom     = tf.constant(0.0, dtype=data_type)
        # div_w    = tf.constant(0.0, dtype=data_type)
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS2:
            tpS2.watch(x)
            tpS2.watch(y)
            tpS2.watch(z)
                
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS:
                
                tpS.watch(t)
                tpS.watch(x)
                tpS.watch(y)
                tpS.watch(z)
                
                with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                    
                    tp.watch(x)
                    tp.watch(y)
                    tp.watch(z)
                    
                    outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                    
                    u = outs[:,0:1]
                    v = outs[:,1:2]
                    w = outs[:,2:3]
                    
                u_y = tp.gradient(u, y)
                u_z = tp.gradient(u, z)
                
                v_x = tp.gradient(v, x)
                v_z = tp.gradient(v, z)
                
                w_x = tp.gradient(w, x)
                w_y = tp.gradient(w, y)
                
                #Vorticity equals curl of velocity
                omegax  = w_y - v_z
                omegay  = u_z - w_x
                omegaz  = v_x - u_y
                
                del tp
            
            omegax_z = tpS.gradient(omegax, z)
            omegay_z = tpS.gradient(omegay, z)
            
            omegaz_x = tpS.gradient(omegaz, x)
            omegaz_y = tpS.gradient(omegaz, y)
            
            a   = w*omegay - v*omegaz
            b   = u*omegaz - w*omegax
            
            p   = omegaz_y - omegay_z
            q   = omegax_z - omegaz_x
            
            Fx  = a + nu*p
            Fy  = b + nu*q
            
            with tpS2.stop_recording():
                
                #### Divergence ###############
                u_x = tpS.gradient(u, x)
                v_y = tpS.gradient(v, y)
                w_z = tpS.gradient(w, z)
                
                # omegax_x = tpS.gradient(omegax, x)
                # omegay_y = tpS.gradient(omegay, y)
                # omegaz_z = tpS.gradient(omegaz, z)
        
                ############################
                
                omegaz_t    = tpS.gradient(omegaz, t)
            
            del tpS
        
        ###############################
        ####Rotational formulation #####
        ###############################
        Fx_y =   tpS2.gradient(Fx, y)
        Fy_x =   tpS2.gradient(Fy, x)
        
        del tpS2
        
        #Divergence
        div = eq_continuity(u_x, v_y, w_z)#, w=w, rho_ratio=rho_ratio)
        
        # div_w = 1e9*eq_continuity(omegax_x, omegay_y, omegaz_z)
        
        ## Rotational formulation
        mom_z   = eq_vorticity_RF_F(omegaz_t,
                                  Fy_x, Fx_y,
                                  tf.constant(0.0, dtype=data_type),
                                  tf.constant(0.0, dtype=data_type),
                                  )
        
        junk = tf.constant(0.0, dtype=data_type)
        
        return(div, junk, junk, junk, mom_z, junk)

    # @tf.function
    def pde_vorticity_3O_hydrostatic(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        # print("Tracing pde hydro!")
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS2:
            
            tpS2.watch(x)
            tpS2.watch(y)
            tpS2.watch(z)
                
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS:
                
                tpS.watch(t)
                tpS.watch(x)
                tpS.watch(y)
                tpS.watch(z)
                
                with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                    
                    tp.watch(x)
                    tp.watch(y)
                    tp.watch(z)
                    
                    outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                    
                    u = outs[:,0:1]
                    v = outs[:,1:2]
                    w = outs[:,2:3]
                    
                # u_x = tp.gradient(u, x)
                u_y = tp.gradient(u, y)
                u_z = tp.gradient(u, z)
                
                v_x = tp.gradient(v, x)
                # v_y = tp.gradient(v, y)
                v_z = tp.gradient(v, z)
                
                w_x = tp.gradient(w, x)
                w_y = tp.gradient(w, y)
                # w_z = tp.gradient(w, z)
                
                #Vorticity equals curl of velocity
                omegax  = w_y - v_z
                omegay  = u_z - w_x
                omegaz  = v_x - u_y
            
                del tp
            
            # omegax_x = tpS.gradient(omegax, x)
            omegax_y = tpS.gradient(omegax, y)
            omegax_z = tpS.gradient(omegax, z)
            
            omegay_x = tpS.gradient(omegay, x)
            # omegay_y = tpS.gradient(omegay, y)
            omegay_z = tpS.gradient(omegay, z)
            
            omegaz_x = tpS.gradient(omegaz, x)
            omegaz_y = tpS.gradient(omegaz, y)
            # omegaz_z = tpS.gradient(omegaz, z)
            
            a   = w*omegay - v*omegaz
            b   = u*omegaz - w*omegax
            c   = v*omegax - u*omegay
                    
            p   = omegaz_y - omegay_z
            q   = omegax_z - omegaz_x
            r   = omegay_x - omegax_y
            
            Fx  = a + nu*p
            Fy  = b + nu*q
            Fz  = c + nu*r
            
            with tpS2.stop_recording():
                
                u_x = tpS.gradient(u, x)
                v_y = tpS.gradient(v, y)
                w_z = tpS.gradient(w, z)
                
                # omegax_x = tpS.gradient(omegax, x)
                # omegay_y = tpS.gradient(omegay, y)
                # omegaz_z = tpS.gradient(omegaz, z)
                
                ###############################
                # u_xx = tpS.gradient(u_x, x)
                # u_yy = tpS.gradient(u_y, y)
                # u_zz = tpS.gradient(u_z, z)
                #
                # v_xx = tpS.gradient(v_x, x)
                # v_yy = tpS.gradient(v_y, y)
                # v_zz = tpS.gradient(v_z, z)
                #
                # w_xx = tpS.gradient(w_x, x)
                # w_yy = tpS.gradient(w_y, y)
                # w_zz = tpS.gradient(w_z, z)
                
                ############################
                # a_y =   tpS.gradient(a, y)
                # a_z =   tpS.gradient(a, z)
                #
                # b_x =   tpS.gradient(b, x)
                # b_z =   tpS.gradient(b, z)
                #
                # c_x =   tpS.gradient(c, x)
                # c_y =   tpS.gradient(c, y)
        
                ############################
                omegax_t    = tpS.gradient(omegax, t)
                omegay_t    = tpS.gradient(omegay, t)
                omegaz_t    = tpS.gradient(omegaz, t)
            
            del tpS
        
        # omegax_xx = tpS2.gradient(omegax_x, x)
        # omegax_yy = tpS2.gradient(omegax_y, y)
        # omegax_zz = tpS2.gradient(omegax_z, z)
        #
        # omegay_xx = tpS2.gradient(omegay_x, x)
        # omegay_yy = tpS2.gradient(omegay_y, y)
        # omegay_zz = tpS2.gradient(omegay_z, z)
        #
        # omegaz_xx = tpS2.gradient(omegaz_x, x)
        # omegaz_yy = tpS2.gradient(omegaz_y, y)
        # omegaz_zz = tpS2.gradient(omegaz_z, z)
        
        ############################
        # p_y =   tpS2.gradient(p, y)
        # p_z =   tpS2.gradient(p, z)
        #
        # q_x =   tpS2.gradient(q, x)
        # q_z =   tpS2.gradient(q, z)
        #
        # r_x =   tpS2.gradient(r, x)
        # r_y =   tpS2.gradient(r, y)
        
        Fx_y =   tpS2.gradient(Fx, y)
        Fx_z =   tpS2.gradient(Fx, z)
        
        Fy_x =   tpS2.gradient(Fy, x)
        Fy_z =   tpS2.gradient(Fy, z)
        
        Fz_x =   tpS2.gradient(Fz, x)
        Fz_y =   tpS2.gradient(Fz, y)
        
        del tpS2
        
        #Divergence
        div = eq_continuity(u_x, v_y, w_z)#, w=w, rho_ratio=rho_ratio)
        # div_w = eq_continuity(omegax_x, omegay_y, omegaz_z)
        
        # poi_x   = eq_poisson(u_xx, u_yy, u_zz, omegaz_y, omegay_z)
        # poi_y   = eq_poisson(v_xx, v_yy, v_zz, omegax_z, omegaz_x)
        # poi_z   = eq_poisson(w_xx, w_yy, w_zz, omegay_x, omegax_y)
        #
        # poisson = poi_x + poi_y + poi_z
        
        
        
        ## Laplacian formulation
        # mom_x   = eq_vorticity_LF(omegax_t,
        #                           omegax_xx, omegax_yy, omegax_zz,
        #                           c_y, b_z,
        #                           theta_y,
        #                           nu,
        #                           N)
        #
        # mom_y   = eq_vorticity_LF(omegay_t,
        #                          omegay_xx, omegay_yy, omegay_zz,
        #                          a_z, c_x,
        #                          -theta_x,
        #                          nu,
        #                          N)
        #
        # mom_z   = eq_vorticity_LF(omegaz_t,
        #                          omegaz_xx, omegaz_yy, omegaz_zz,
        #                          b_x, a_y,
        #                          tf.constant(0.0, dtype=data_type),
        #                          nu,
        #                          tf.constant(0.0, dtype=data_type))
        
        ## Rotational formulation
        mom_x   = 1e-2*eq_vorticity_RF_F(omegax_t,
                                    Fz_y, Fy_z,
                                    )
        
        mom_y   = 1e-2*eq_vorticity_RF_F(omegay_t,
                                    Fx_z, Fz_x,
                                    )
        
        mom_z   = eq_vorticity_RF_F(omegaz_t,
                                    Fy_x, Fx_y,
                                    )
        
        # mom = mom_x + mom_y + mom_z
        
        # #Potential temperature
        # heat        = eq_temperature(u, v, w,
        #                              theta_t, theta_x, theta_y, theta_z,
        #                              theta_xx, theta_yy, theta_zz,
        #                              N, k=nu)
        #
        # theta_zero  =  tf.reduce_sum(theta) #Temperature perturbations must have zero-mean
        #
        # thermal     = heat + theta_zero
        junk = tf.constant(0.0, dtype=data_type)
        
        # print("Finish tracing pde hydro!")
        
        return(div, junk, mom_x, mom_y, mom_z, junk)
    
    # @tf.function
    def pde_vorticity_3O_hydro_noNu(self, model, t, z, x, y, nu, rho, rho_ratio, N):
                
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS:
            
            tpS.watch(t)
            tpS.watch(x)
            tpS.watch(y)
            tpS.watch(z)
                
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                
                tp.watch(x)
                tp.watch(y)
                tp.watch(z)
                
                outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                
                u = outs[:,0:1]
                v = outs[:,1:2]
                w = outs[:,2:3]
                
            # u_x = tp.gradient(u, x)
            u_y = tp.gradient(u, y)
            u_z = tp.gradient(u, z)
            
            v_x = tp.gradient(v, x)
            # v_y = tp.gradient(v, y)
            v_z = tp.gradient(v, z)
            
            w_x = tp.gradient(w, x)
            w_y = tp.gradient(w, y)
            # w_z = tp.gradient(w, z)   
            
            del tp 
            
            #Vorticity equals curl of velocity
            omegax  = w_y - v_z
            omegay  = u_z - w_x
            omegaz  = v_x - u_y
            
            a   = w*omegay - v*omegaz
            b   = u*omegaz - w*omegax
            c   = v*omegax - u*omegay
            
            Fx  = a #+ nu*p
            Fy  = b #+ nu*q
            Fz  = c #+ nu*r
            
        u_x = tpS.gradient(u, x)
        v_y = tpS.gradient(v, y)
        w_z = tpS.gradient(w, z)
        
        # omegax_x    = tpS.gradient(omegax, x)
        # omegay_y    = tpS.gradient(omegay, y)
        # omegaz_z    = tpS.gradient(omegaz, z)
        
        ############################
        omegax_t    = tpS.gradient(omegax, t)
        omegay_t    = tpS.gradient(omegay, t)
        omegaz_t    = tpS.gradient(omegaz, t)
        
        # del tpS
        
        Fx_y =   tpS.gradient(Fx, y)
        Fx_z =   tpS.gradient(Fx, z)
        
        Fy_x =   tpS.gradient(Fy, x)
        Fy_z =   tpS.gradient(Fy, z)
        
        Fz_x =   tpS.gradient(Fz, x)
        Fz_y =   tpS.gradient(Fz, y)
        
        del tpS
        
        #Divergence
        div = eq_continuity(u_x, v_y, w_z, w=w, rho_ratio=rho_ratio)
        # div_w = eq_continuity(omegax_x, omegay_y, omegaz_z)
        
        # poi_x   = eq_poisson(u_xx, u_yy, u_zz, omegaz_y, omegay_z)
        # poi_y   = eq_poisson(v_xx, v_yy, v_zz, omegax_z, omegaz_x)
        # poi_z   = eq_poisson(w_xx, w_yy, w_zz, omegay_x, omegax_y)
        #
        # poisson = poi_x + poi_y + poi_z
        
        
        
        ## Laplacian formulation
        # mom_x   = eq_vorticity_LF(omegax_t,
        #                           omegax_xx, omegax_yy, omegax_zz,
        #                           c_y, b_z,
        #                           theta_y,
        #                           nu,
        #                           N)
        #
        # mom_y   = eq_vorticity_LF(omegay_t,
        #                          omegay_xx, omegay_yy, omegay_zz,
        #                          a_z, c_x,
        #                          -theta_x,
        #                          nu,
        #                          N)
        #
        # mom_z   = eq_vorticity_LF(omegaz_t,
        #                          omegaz_xx, omegaz_yy, omegaz_zz,
        #                          b_x, a_y,
        #                          tf.constant(0.0, dtype=data_type),
        #                          nu,
        #                          tf.constant(0.0, dtype=data_type))
        
        ## Rotational formulation
        mom_x   = 1e-1*eq_vorticity_RF_F(omegax_t,
                                    Fz_y, Fy_z,
                                    )
        
        mom_y   = 1e-1*eq_vorticity_RF_F(omegay_t,
                                    Fx_z, Fz_x,
                                    )
        
        mom_z   = eq_vorticity_RF_F(omegaz_t,
                                    Fy_x, Fx_y,
                                    )
        
        # mom = mom_x + mom_y + mom_z
        
        # #Potential temperature
        # heat        = eq_temperature(u, v, w,
        #                              theta_t, theta_x, theta_y, theta_z,
        #                              theta_xx, theta_yy, theta_zz,
        #                              N, k=nu)
        #
        # theta_zero  =  tf.reduce_sum(theta) #Temperature perturbations must have zero-mean
        #
        # thermal     = heat + theta_zero
        junk = tf.constant(0.0, dtype=data_type)
        
        return(div, junk, mom_x, mom_y, mom_z, junk)
    
    # @tf.function
    def pde_vorticity_4O_noNu(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        # div     = tf.constant(0.0, dtype=data_type)
        # mom     = tf.constant(0.0, dtype=data_type)
        # div_w    = tf.constant(0.0, dtype=data_type)
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS2:
            tpS2.watch(t)
            
            tpS2.watch(x)
            tpS2.watch(y)
            tpS2.watch(z)
                
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                
                tp.watch(x)
                tp.watch(y)
                tp.watch(z)
                
                outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                
                u = outs[:,0:1]
                v = outs[:,1:2]
                w = outs[:,2:3]
                
                with tp.stop_recording():
                    rho = outs[:,3:4]   #Normalized density perturbations p'/p_0
                
            u_y = tp.gradient(u, y)
            u_z = tp.gradient(u, z)
            
            v_x = tp.gradient(v, x)
            v_z = tp.gradient(v, z)
            
            w_x = tp.gradient(w, x)
            w_y = tp.gradient(w, y)
            
            del tp
            
            #Vorticity equals curl of velocity
            omegax  = w_y - v_z
            omegay  = u_z - w_x
            omegaz  = v_x - u_y
            
            a   = w*omegay - v*omegaz
            b   = u*omegaz - w*omegax
            
            Fx  = a #+ nu*p
            Fy  = b #+ nu*q
        
        #### Density perturbations ###############
        # rho_t = tpS2.gradient(rho, t)
        #
        # rho_x = tpS2.gradient(rho, x)
        # rho_y = tpS2.gradient(rho, y)
        # rho_z = tpS2.gradient(rho, z)
        
        #### Divergence ###############
        u_x = tpS2.gradient(u, x)
        v_y = tpS2.gradient(v, y)
        w_z = tpS2.gradient(w, z)
        
        omegax_x = tpS2.gradient(omegax, x)
        omegay_y = tpS2.gradient(omegay, y)
        omegaz_z = tpS2.gradient(omegaz, z)

        ############################
        omegaz_t    = tpS2.gradient(omegaz, t)
                
        ###############################
        ####Rotational formulation #####
        ###############################
        Fx_y =   tpS2.gradient(Fx, y)
        Fy_x =   tpS2.gradient(Fy, x)
        
        del tpS2
        
        #Divergence
        div = eq_continuity(u_x, v_y, w_z)#, u, v, w, rho, rho_t, rho_x, rho_y, rho_z)
        
        div_w = eq_continuity(omegax_x, omegay_y, omegaz_z)
        
        ## Rotational formulation
        mom_z   = eq_vorticity_RF_F(omegaz_t,
                                  Fy_x, Fx_y
                                  )
        
        rho_mean  =  tf.reduce_sum(rho)
        
        junk = tf.constant(0.0, dtype=data_type)
        
        return(div, div_w, junk, junk, mom_z, rho_mean)

    # @tf.function
    def pde_vorticity_4O(self, model, t, z, x, y, nu, rho, rho_ratio, N):
        
        # div     = tf.constant(0.0, dtype=data_type)
        # mom_x   = tf.constant(0.0, dtype=data_type)
        # mom_y   = tf.constant(0.0, dtype=data_type)
        # mom_z   = tf.constant(0.0, dtype=data_type)
        # temp    = tf.constant(0.0, dtype=data_type)
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS2:
            tpS2.watch(x)
            tpS2.watch(y)
            tpS2.watch(z)
                
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tpS:
                
                tpS.watch(t)
                tpS.watch(x)
                tpS.watch(y)
                tpS.watch(z)
                
                with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                    
                    tp.watch(x)
                    tp.watch(y)
                    tp.watch(z)
                    
                    outs = self.forward_pass(model, tf.concat([t, z, x, y], axis=1))
                    
                    u = outs[:,0:1]
                    v = outs[:,1:2]
                    w = outs[:,2:3]
                    
                    with tp.stop_recording():
                        theta = outs[:,3:4]
                    
                # u_x = tp.gradient(u, x)
                u_y = tp.gradient(u, y)
                u_z = tp.gradient(u, z)
                
                v_x = tp.gradient(v, x)
                # v_y = tp.gradient(v, y)
                v_z = tp.gradient(v, z)
                
                w_x = tp.gradient(w, x)
                w_y = tp.gradient(w, y)
                # w_z = tp.gradient(w, z)
                
                #Vorticity equals curl of velocity
                omegax  = w_y - v_z
                omegay  = u_z - w_x
                omegaz  = v_x - u_y
            
                del tp
            
            # omegax_x = tpS.gradient(omegax, x)
            omegax_y = tpS.gradient(omegax, y)
            omegax_z = tpS.gradient(omegax, z)
            
            omegay_x = tpS.gradient(omegay, x)
            # omegay_y = tpS.gradient(omegay, y)
            omegay_z = tpS.gradient(omegay, z)
            
            omegaz_x = tpS.gradient(omegaz, x)
            omegaz_y = tpS.gradient(omegaz, y)
            # omegaz_z = tpS.gradient(omegaz, z)
            
            a   = w*omegay - v*omegaz
            b   = u*omegaz - w*omegax
            c   = v*omegax - u*omegay
                    
            p   = omegaz_y - omegay_z
            q   = omegax_z - omegaz_x
            r   = omegay_x - omegax_y
            
            Fx  = a + nu*p
            Fy  = b + nu*q
            Fz  = c + nu*r
            
            theta_x     = tpS.gradient(theta, x)
            theta_y     = tpS.gradient(theta, y)
            theta_z     = tpS.gradient(theta, z)
            
            with tpS2.stop_recording():
                
                u_x = tpS.gradient(u, x)
                v_y = tpS.gradient(v, y)
                w_z = tpS.gradient(w, z)
                
                omegax_x = tpS.gradient(omegax, x)
                omegay_y = tpS.gradient(omegay, y)
                omegaz_z = tpS.gradient(omegaz, z)
                
                ###############################
                # u_xx = tpS.gradient(u_x, x)
                # u_yy = tpS.gradient(u_y, y)
                # u_zz = tpS.gradient(u_z, z)
                #
                # v_xx = tpS.gradient(v_x, x)
                # v_yy = tpS.gradient(v_y, y)
                # v_zz = tpS.gradient(v_z, z)
                #
                # w_xx = tpS.gradient(w_x, x)
                # w_yy = tpS.gradient(w_y, y)
                # w_zz = tpS.gradient(w_z, z)
                
                ############################
                # a_y =   tpS.gradient(a, y)
                # a_z =   tpS.gradient(a, z)
                #
                # b_x =   tpS.gradient(b, x)
                # b_z =   tpS.gradient(b, z)
                #
                # c_x =   tpS.gradient(c, x)
                # c_y =   tpS.gradient(c, y)
        
                ############################
                omegax_t    = tpS.gradient(omegax, t)
                omegay_t    = tpS.gradient(omegay, t)
                omegaz_t    = tpS.gradient(omegaz, t)
                
                theta_t     = tpS.gradient(theta, t)
            
            del tpS
        
        # omegax_xx = tpS2.gradient(omegax_x, x)
        # omegax_yy = tpS2.gradient(omegax_y, y)
        # omegax_zz = tpS2.gradient(omegax_z, z)
        #
        # omegay_xx = tpS2.gradient(omegay_x, x)
        # omegay_yy = tpS2.gradient(omegay_y, y)
        # omegay_zz = tpS2.gradient(omegay_z, z)
        #
        # omegaz_xx = tpS2.gradient(omegaz_x, x)
        # omegaz_yy = tpS2.gradient(omegaz_y, y)
        # omegaz_zz = tpS2.gradient(omegaz_z, z)
        
        ############################
        
        theta_xx = tpS2.gradient(theta_x, x)
        theta_yy = tpS2.gradient(theta_y, y)
        theta_zz = tpS2.gradient(theta_z, z)
        
        ############################
        # p_y =   tpS2.gradient(p, y)
        # p_z =   tpS2.gradient(p, z)
        #
        # q_x =   tpS2.gradient(q, x)
        # q_z =   tpS2.gradient(q, z)
        #
        # r_x =   tpS2.gradient(r, x)
        # r_y =   tpS2.gradient(r, y)
        
        Fx_y =   tpS2.gradient(Fx, y)
        Fx_z =   tpS2.gradient(Fx, z)
        
        Fy_x =   tpS2.gradient(Fy, x)
        Fy_z =   tpS2.gradient(Fy, z)
        
        Fz_x =   tpS2.gradient(Fz, x)
        Fz_y =   tpS2.gradient(Fz, y)
        
        del tpS2
        
        #Divergence
        div = eq_continuity(u_x, v_y, w_z)#, w=w, rho_ratio=rho_ratio)
        div_w = eq_continuity(omegax_x, omegay_y, omegaz_z)
        
        # poi_x   = eq_poisson(u_xx, u_yy, u_zz, omegaz_y, omegay_z)
        # poi_y   = eq_poisson(v_xx, v_yy, v_zz, omegax_z, omegaz_x)
        # poi_z   = eq_poisson(w_xx, w_yy, w_zz, omegay_x, omegax_y)
        #
        # poisson = poi_x + poi_y + poi_z
        
        
        
        ## Laplacian formulation
        # mom_x   = eq_vorticity_LF(omegax_t,
        #                           omegax_xx, omegax_yy, omegax_zz,
        #                           c_y, b_z,
        #                           theta_y,
        #                           nu,
        #                           N)
        #
        # mom_y   = eq_vorticity_LF(omegay_t,
        #                          omegay_xx, omegay_yy, omegay_zz,
        #                          a_z, c_x,
        #                          -theta_x,
        #                          nu,
        #                          N)
        #
        # mom_z   = eq_vorticity_LF(omegaz_t,
        #                          omegaz_xx, omegaz_yy, omegaz_zz,
        #                          b_x, a_y,
        #                          tf.constant(0.0, dtype=data_type),
        #                          nu,
        #                          tf.constant(0.0, dtype=data_type))
        
        ## Rotational formulation
        mom_x   = eq_vorticity_RF_F(omegax_t,
                                    Fz_y, Fy_z,
                                    theta_y,
                                    N)
        
        mom_y   = eq_vorticity_RF_F(omegay_t,
                                    Fx_z, Fz_x,
                                    -theta_x,
                                    N)
        
        mom_z   = eq_vorticity_RF_F(omegaz_t,
                                    Fy_x, Fx_y,
                                    tf.constant(0.0, dtype=data_type),
                                    tf.constant(0.0, dtype=data_type),
                                    )
        
        # mom = mom_x + mom_y + mom_z
        
        #Potential temperature
        heat        = eq_temperature(u, v, w,
                                     theta_t, theta_x, theta_y, theta_z,
                                     theta_xx, theta_yy, theta_zz,
                                     N, k=nu)
        
        theta_zero  =  tf.reduce_sum(theta) #Temperature perturbations must have zero-mean
        
        return(div, div_w, mom_x, mom_y, mom_z, heat, theta_zero)
    
    def _select_pde_function(self, NS='VP_div'):
        
        NS_type = NS[:2].upper()
        variant = NS[3:].lower()
        
        if NS_type.upper() == 'ME':
            print("Using [mean] div-free variant ...")
            func = self.pde_div
        elif NS_type == 'VP':
            
            if self.shape_out == 3:
                print("Using div-free variant ...")
                func = self.pde_div
            elif self.shape_out == 4:
                print("Using div-free and mom variant ...")
                func = self.pde_div_mom
            elif self.shape_out == 5:
                print("Using div-free, mom and temp variant ...")
                func = self.pde_div_mom_temp
            else:
                raise NotImplementedError
        
        elif NS_type[:2] == 'VV':
            if self.shape_out == 3:
                if variant == '':
                    print("Using the Vorticity-z PDE variant ...")
                    func = self.pde_vorticity_3O
                elif variant == 'div':
                    print("Using the DIV PDE variant  ...")
                    func = self.pde_vorticity_3O_div
                elif variant == 'hydro_nonu':
                    print("Using the Hydrostatic noNu PDE variant ...")
                    func = self.pde_vorticity_3O_hydro_noNu
                elif variant == 'hydro':
                    print("Using the Hydrostatic PDE variant ...")
                    func = self.pde_vorticity_3O_hydrostatic
                elif variant == 'nonu':
                    print("Using the no viscosity PDE variant ...")
                    func = self.pde_vorticity_3O_noNu
                else:
                    raise ValueError('PDE variant unrecognized')
                    
            elif self.shape_out == 4:
                if variant == 'nonu':
                    print("Using the no viscosity PDE variant ...")
                    func = self.pde_vorticity_4O_noNu
                else:
                    print("Using the full PDE variant ...")
                    func = self.pde_vorticity_4O
            else:
                raise NotImplementedError
        else:
            print("PDE type not implemented: %s" %NS_type)
            raise NotImplementedError
        
        self.pde_func = func
        
    # @tf.function
    def pde(self, model, X):
        '''
        X = [t, z, x, y, nu, rho, rho_ratio, N]
        '''
        # print("Tracing pde!")
        
        t = X[:,0:1]
        z = X[:,1:2]
        x = X[:,2:3]
        y = X[:,3:4]
        nu = X[:,4:5]
        rho = X[:,5:6]
        rho_ratio = X[:,6:7]
        N = X[:,7:8]
        
        # t, z, x, y, nu, rho, rho_ratio, N = tf.split(X, num_or_size_splits=8, axis=1)
        
        nu_scaling = self.model.scale.nu
        nu          = nu*nu_scaling #scaling
        
        output = self.pde_func(model, t, z, x, y, nu, rho, rho_ratio, N)
        
        # print("Finish tracing pde!")
        
        return output
    
    # @tf.function
    def loss_pde(self, model, X):
        '''
        X = [t, z, x, y, nu, rho, rho_ratio, N]
        '''
        
        # print("Tracing loss_pde!")
        
        #div, mom, temp, theta
        outputs = self.pde(model, X)
        
        loss_div        = tf.reduce_mean(tf.square(outputs[0]))
        loss_div_vor    = tf.reduce_mean(tf.square(outputs[1]))
        loss_mom        = tf.reduce_mean(tf.square(outputs[2])) + tf.reduce_mean(tf.square(outputs[3])) + tf.reduce_mean(tf.square(outputs[4]))
        loss_temp       = tf.reduce_mean(tf.square(outputs[5]))  #Zero mean temperature changes
        
        # loss_srt = tf.constant(0.0, dtype=data_type)
        
        # print("Finish tracing loss_pde!")
        
        return (loss_div, loss_mom, loss_div_vor, loss_temp)
    
    # @tf.function
    def loss_data(self, model, X):
        '''
        X = [t, z, x, y, d, kx, ky, kz]
        
        d = u*k_x + w*k_z + n
        
        u = u_0 + u'
        w = w_0
        
        || d - u_0*k_x - w_0*k_z - u'*k_x ||
        
        Inputs:
            t   :        time coordinate . Dimension (N,1)
            x
            y
            z
            d   :      #Doppler. Dimension (N,1)
            k   :      #Bragg coefficients. Dimension (N,3)
        
        
        '''
        # print("Tracing loss_data!")
        
        f = X[:,4:5]
        k = X[:,5:8]
        f_err = X[:,8:9]
        
        #Standard
        u0 = self.forward_pass(model, X[:,0:4], training=True)
        
        nn0 = tf.reduce_sum(u0*k, axis=1, keepdims=True)
        
        df0 = f  + nn0
        
        # loss = tf.square(tf.reduce_mean(tf.abs(df0/f_err)))
        loss = tf.reduce_mean(tf.square(df0/f_err))
        
        return (loss)
    
    def loss_slope_recovery_term(self):
        '''
        Jagtap AD, Kawaguchi K,Karniadakis GE. 2020
        Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks.
        Proc.R.Soc.A476: 20200334.
        http://dx.doi.org/10.1098/rspa.2020.0334
        '''
        if self.laaf:
            # slt = tf.constant(0.0, dtype=data_type)
            slt = tf.constant(1.0, dtype=data_type)/tf.reduce_mean(tf.exp(self.model.alphas))
        else:
            slt = tf.constant(0.0, dtype=data_type)
        
        return slt
    
    # @tf.function
    def loss_glb(self, 
                 model,
                X_data,
                X_pde,
                w_data, w_pde, w_srt,
                ):
        
        # print("Tracing loss_glb!") 
        
        loss_data = self.loss_data(model, X_data)
        
        loss_div, loss_mom, loss_div_vor, _ = self.loss_pde(model, X_pde)
        
        loss_pde = loss_div + loss_mom +loss_div_vor
        
        loss_srt = self.loss_slope_recovery_term()
        
        # loss_total = tf.square(w_data)*loss_data + tf.square(w_div)*loss_div + tf.square(w_mom)*loss_mom + tf.square(w_temp)*loss_div_vor + tf.square(w_srt)*loss_srt
        
        loss_total = w_data*loss_data + w_pde*loss_pde + w_srt*loss_srt
        
        
        # print("Finish tracing loss_glb!") 
        
        return loss_total, loss_data, loss_pde, loss_srt
    
    @tf.function
    def loss_glb_lbfgs(self, 
                 model,
                X_data,
                X_pde,
                w_data, w_pde, w_srt,
                ):
        
        # print("Tracing loss_glb!") 
        
        loss_data = self.loss_data(model, X_data)
        
        loss_div, loss_mom, loss_div_vor, _ = self.loss_pde(model, X_pde)
        
        loss_pde = loss_div + loss_mom +loss_div_vor
        
        loss_srt = tf.constant(0.0)
        
        # loss_total = tf.square(w_data)*loss_data + tf.square(w_div)*loss_div + tf.square(w_mom)*loss_mom + tf.square(w_temp)*loss_div_vor + tf.square(w_srt)*loss_srt
        
        loss_total = w_data*loss_data + w_pde*loss_pde + w_srt*loss_srt
        
        
        # print("Finish tracing loss_glb!") 
        
        return loss_total
    
    @tf.function
    def grad_desc(self, *args):
        # print("\nTracing grad_desc!") 
        
        # print("Tracing Grad_desc!!") 
        trainable_variables = self.model.trainable_variables
        
        with tf.GradientTape(persistent=False) as tp:
            
            losses = self.loss_glb(*args)
        
            loss_total  = losses[0]
            
        grad        = tp.gradient(loss_total, trainable_variables)
        
        # del tp
        self.optimizer.apply_gradients(zip(grad, trainable_variables))
        
        # print("Finish tracing grad_desc!") 
        
        return losses
    
    @tf.function
    def grad_desc_plus(self, *args):
        
        # print("\nTracing grad_desc_plus!") 
        
        # print("Tracing Grad_desc_plus!") 
        trainable_params = self.model.trainable_variables
        
        with tf.GradientTape(persistent=True) as tp:
            
            losses = self.loss_glb(*args)
        
            loss_total, loss_data, loss_pde, _ = losses
        
        grad      = tp.gradient(loss_total, trainable_params)
        
        grad_data = tp.gradient(loss_data, trainable_params)
        grad_pde  = tp.gradient(loss_pde, trainable_params)
        # grad_mom  = tp.gradient(loss_mom, trainable_params)
        # grad_temp = tp.gradient(loss_temp, trainable_params)
        # grad_srt  = tf.constant(0.0, dtype=data_type) #tp.gradient(loss_srt, trainable_params)
        
        del tp
        
        self.optimizer.apply_gradients(zip(grad, trainable_params))
        
        grad_mean      = self.mean_grads(grad)
        grad_data_mean = self.mean_grads(grad_data)
        grad_pde_mean  = self.mean_grads(grad_pde)
        # grad_mom_mean  = self.mean_grads(grad_mom)
        # grad_temp_mean = self.mean_grads(grad_temp)
        grad_srt_mean  = tf.constant(0.0, dtype=data_type) #self.mean_grads(grad_srt)
        
        grads  = [grad_mean, grad_data_mean,  grad_pde_mean, grad_srt_mean]
        
        # print("Finish tracing grad_desc_plus!") 
        
        return losses, grads
    
    def train_epoch(self,
                  X_data,
                  X_pde,
                  w_data=tf.constant(1.0),
                  w_pde=tf.constant(1e4),
                  # w_mom=tf.constant(1e1),
                  # w_temp=tf.constant(1e4),
                  w_srt=tf.constant(1e3),
                  calc_gradients=False
                  ):
        
        #Calculate only losses, no grads (faster)
        losses = self.grad_desc(self.model,
                                X_data,
                                X_pde,
                                w_data, w_pde, w_srt
                                )
        
        grads = None
        
        if calc_gradients:
            #Calculate losses and losses and gradients
            # losses, grads = self.grad_desc_plus( self.model,X_data,
            #                                      X_pde,
            #                                      w_data, w_pde, w_srt
            #                                      # tf.constant(1.0, dtype=data_type),
            #                                      )
            
            ## grads  = [grad_mean, grad_data_mean,  grad_div_mean, grad_mom_mean, grad_temp_mean, grad_srt_mean]
            grads  = [1.0, 1e0,  1e-2, 1e-2, 1e-2, 1e-2]
            
        return(losses, grads)
             
    def train(self,
              t, lon, lat, alt, d, k,
              t_val, lon_val, lat_val, alt_val,
              u_val, v_val, w_val,
              d_err = None,
              epochs = 10 ** 4,
              batch_size = None,
              tol = 1e-5,
              print_rate    = 200,
              saving_rate   = 500,
              resampling_rate = 200,
              grad_upd_rate = 200,
              filename = None,
              w_pde_update_rate = 1e-4,
              tmin=None,
              tmax=None,
              lr      = 1e-3,
              opt     = "Adam",
              w_data  = 1.,
              w_div   = 1e0,
              w_mom   = 1e0,
              w_temp  = 1e0,
              w_srt   = 1.0,
              ns_pde  = 10000,
              # NS_type     = 'VP',             #Velocity-Vorticity (VV) or Velocity-Pressure (VP) form
              dropout     = True,
              ):
            
        t = t - time_base 
        t_val = t_val - time_base 
            
        print("\n>>>>> training setting;")
        print("         # of epoch     :", epochs)
        print("         batch size     :", batch_size)
        print("         convergence tol:", tol)
        
        self.lr      = lr
        self.opt     = opt
        self.dropout = dropout
        
        self.w_data  = w_data
        self.w_div   = w_div
        self.w_mom   = w_mom
        self.w_temp  = w_temp
        self.w_srt   = w_srt
        
        self.N_pde   = ns_pde    #number of samples
        
        # self.NS_type    = NS_type
        
        # optimization
        self.optimizer = self.opt_(self.lr, self.opt, epochs)
        
        x, y, z = lla2xyh(lat, lon, alt, self.lat_ref, self.lon_ref, self.alt_ref)
        x_val, y_val, z_val = lla2xyh(lat_val, lon_val, alt_val, self.lat_ref, self.lon_ref, self.alt_ref)
        
        # Training dataset
        t = tf.convert_to_tensor(t.reshape(-1,1), data_type)
        x = tf.convert_to_tensor(x.reshape(-1,1), data_type)
        y = tf.convert_to_tensor(y.reshape(-1,1), data_type)
        z = tf.convert_to_tensor(z.reshape(-1,1), data_type)
        
        d  = tf.convert_to_tensor(d.reshape(-1,1), data_type)
        k  = tf.convert_to_tensor(k.reshape(-1,3), data_type)
        
        if d_err is None:
            d_err = np.ones_like(d)
            
        d_err  = tf.convert_to_tensor(d_err.reshape(-1,1), data_type)
        
        #Validation dataset
        t_val = tf.convert_to_tensor(t_val.reshape(-1,1), data_type)
        x_val = tf.convert_to_tensor(x_val.reshape(-1,1), data_type)
        y_val = tf.convert_to_tensor(y_val.reshape(-1,1), data_type)
        z_val = tf.convert_to_tensor(z_val.reshape(-1,1), data_type)
        
        u_val  = tf.convert_to_tensor(u_val.reshape(-1,1), data_type)
        v_val  = tf.convert_to_tensor(v_val.reshape(-1,1), data_type)
        w_val  = tf.convert_to_tensor(w_val.reshape(-1,1), data_type)
        
        # bounds (for feature scaling)
        X_data  = tf.concat([t, z, x, y, d, k, d_err], axis=1)
        lb = tf.reduce_min (X_data, axis = 0)[:4]
        ub = tf.reduce_max (X_data, axis = 0)[:4]
        
        self.lbi = lb #+ rb*0.1
        self.ubi = ub #- rb*0.1
        
        self.mn = tf.reduce_mean(X_data, axis = 0)[:4]
        
        print("         lower bounds    :", self.lbi.numpy())
        print("         upper bounds    :", self.ubi.numpy())
        
        w_data  = tf.convert_to_tensor(self.w_data, data_type)
        w_div   = tf.convert_to_tensor(self.w_div, data_type)
        w_mom   = tf.convert_to_tensor(self.w_mom, data_type)
        w_temp  = tf.convert_to_tensor(self.w_temp, data_type)
        w_srt   = tf.convert_to_tensor(self.w_srt, data_type)
        
        w_pde_update_rate = tf.convert_to_tensor(w_pde_update_rate, data_type)
        
        ep_loss     = 0.
        ep_loss_data = 0.
        ep_loss_div = 0.
        ep_loss_mom = 0.
        ep_loss_temp = 0.
        
        ep_loss_u     = 0.
        ep_loss_v     = 0.
        ep_loss_w     = 0.
            
        # ep_grad      = 0.
        ep_grad_data = 0.
        ep_grad_div  = 0.
        ep_grad_mom  = 0.
        ep_grad_temp = 0.
        # ep_grad_srt  = 0.
        
        # min_loss    = 1e10
        
        self.ep_log         = []
        
        self.loss_log       = []
        self.loss_data_log  = []
        self.loss_div_log   = []
        self.loss_mom_log   = []
        self.loss_temp_log  = []
        self.loss_srt_log   = []
        
        self.rmse_u_log     = []
        self.rmse_v_log     = []
        self.rmse_w_log     = []
        
        self.tv_u_log     = []
        self.tv_v_log     = []
        self.tv_w_log     = []
        
        self.tv_ue_log     = []
        self.tv_ve_log     = []
        self.tv_we_log     = []
        
        self.w_div_log = []
        self.w_mom_log = []
        self.w_temp_log = []
        self.w_srt_log = []
        
        self._select_pde_function(self.NS_type)
        self._set_LHS_samples()
        # self.set_Chebyshev_samples()
        
        t0 = time.time()
        
        calc_gradients = False
        
        for ep in range(epochs):
            
            X_pde = self.gen_LHS_samples()
            # X_pde = self.gen_Chebyshev_samples()
            
            print(".", end='', flush=True)
            
            losses, grads = self.train_epoch(X_data,
                                             X_pde,
                                             w_data=w_data,
                                             w_pde=w_div,
                                             # w_mom=w_mom,
                                             # w_temp=w_temp,
                                             w_srt=w_srt,
                                             calc_gradients=calc_gradients,
                                             )
            
            if grads is not None:
                #Update gradient values
                # ep_grad         = grads[0]
                ep_grad_data    = grads[1]
                ep_grad_div     = grads[2]
                ep_grad_mom     = grads[2]
                ep_grad_temp    = grads[2]
                # ep_grad_srt     = grads[3]
                    
            ep_loss         = losses[0]
            ep_loss_data    = losses[1]
            ep_loss_div     = losses[2]
            ep_loss_mom     = losses[2]
            ep_loss_temp    = losses[2]
            ep_loss_srt     = losses[3]
            
            ep_loss_u = np.nan
            ep_loss_v = np.nan
            ep_loss_w = np.nan
            
            tv_u = np.nan
            tv_v = np.nan
            tv_w = np.nan
            
            tv_ue = np.nan
            tv_ve = np.nan
            tv_we = np.nan
            
            if ep % print_rate == 1:
                rmses = self.rmse(t_val, z_val, x_val, y_val,
                                  u_val, v_val, w_val)
                
                ep_loss_u = rmses[0].numpy()
                ep_loss_v = rmses[1].numpy()
                ep_loss_w = rmses[2].numpy()
                
                tv_u = rmses[3].numpy()
                tv_v = rmses[4].numpy()
                tv_w = rmses[5].numpy()
                
                tv_ue = rmses[6].numpy()
                tv_ve = rmses[7].numpy()
                tv_we = rmses[8].numpy()
                
                elps = time.time() - t0
            
                t0 = time.time()
                
                print("\nepoch: %d, elps: %ds" #, nu_scaling: %.2e, rho_scaling: %.2e" 
                    % (ep, 
                       elps,
                       # self.nu_scaling,
                       # self.rho_scaling,
                       )
                    )
                
                print("\t\t\ttotal \tdata \tPDE  \tSRT " )
                 
                 
                print("\tlosses : \t%.1e\t%.1e\t%.1e\t%.1e" 
                    % (
                       ep_loss,
                       ep_loss_data,
                       ep_loss_div,
                       # ep_loss_mom,
                       # ep_loss_temp,
                        ep_loss_srt
                       )
                    )
                
                print("\tweights: \t\t%.1e\t%.1e" #\t%.1e\t%.1e" 
                    % (
                       w_data,
                       w_div,
                       # w_mom,
                       # w_temp,
                       # w_srt
                       )
                    )
                
                print("\tgrads  : \t\t%.1e\t%.1e" #\t%.1e\t%.1e" 
                    % (
                       ep_grad_data,
                       ep_grad_div,
                       # ep_grad_mom,
                       # ep_grad_temp,
                       # ep_grad_srt
                       )
                    )
                
                print("\t\t\tu \t\tv \t\tw" )
                
                print("\tTV  : \t\t%.2e \t%.2e \t%.2e" 
                    % (tv_u,
                       tv_v,
                       tv_w,
                       )
                    )
                
                print("\tTVe : \t\t%.2e \t%.2e \t%.2e" 
                    % (tv_ue,
                       tv_ve,
                       tv_we,
                       )
                    )
                
                print("\trmse: \t\t%.2e \t%.2e \t%.2e" 
                    % (ep_loss_u,
                       ep_loss_v,
                       ep_loss_w,
                       )
                    )
            
            #Save logs
            self.ep_log.append(ep)
            
            self.loss_log.append(ep_loss.numpy())
            self.loss_data_log.append(ep_loss_data.numpy())
            
            self.loss_div_log.append(ep_loss_div.numpy())
            self.loss_mom_log.append(ep_loss_mom.numpy())
            self.loss_temp_log.append(ep_loss_temp.numpy())
            self.loss_srt_log.append(ep_loss_srt.numpy())
            
            self.w_div_log.append(w_div.numpy())
            self.w_mom_log.append(w_mom.numpy())
            self.w_temp_log.append(w_temp.numpy())
            self.w_srt_log.append(w_srt.numpy())
            
            self.rmse_u_log.append(ep_loss_u)
            self.rmse_v_log.append(ep_loss_v)
            self.rmse_w_log.append(ep_loss_w)
            
            self.tv_u_log.append(tv_u)
            self.tv_v_log.append(tv_v)
            self.tv_w_log.append(tv_w)
            
            self.tv_ue_log.append(tv_ue)
            self.tv_ve_log.append(tv_ve)
            self.tv_we_log.append(tv_we)
            
            # w_pde_update_rate_ = np.exp(20*(ep/epochs - 1))
            
            alpha = w_pde_update_rate
                
            if ep < 200 -1:
                continue
            
            #Adaptive PDE weights
            if ep_grad_div != 0:
                w_div  = (1.0-alpha)*w_div  + alpha*max(ep_grad_data/ep_grad_div, w_div)
            
            # if ep_grad_mom != 0:
            #     # w_mom  = w_div*1e4 
            #     w_mom  = (1.0-alpha)*w_mom  + alpha*max(ep_grad_data/ep_grad_mom, w_mom)
            #
            # if ep_grad_temp != 0:
            #     # w_temp = w_div*1e10
            #     w_temp = (1.0-alpha)*w_temp + alpha*max(ep_grad_data/ep_grad_temp, w_temp)
            
            # if ep_grad_srt != 0:
            #     w_srt = (1.0-w_pde_update_rate_)*w_srt   + 1e-2*w_pde_update_rate_*ep_grad_data/ep_grad_srt
                    
            if ep % saving_rate == 0:
                
                self.save(filename, ep)

            calc_gradients = False
            if ep % grad_upd_rate == 0:
                calc_gradients = True
            
            if ep_loss < tol:
                print(">>>>> program terminating with the loss converging to its tolerance.")
                break
        
        print("\n************************************************************")
        print("*****************     MAIN PROGRAM END     *****************")
        print("************************************************************")
        print(">>>>> end time:", datetime.datetime.now())
    
    def train_residuals(self,
              t, lon, lat, alt, d, k,
              t_val, lon_val, lat_val, alt_val,
              u_val, v_val, w_val,
              d_err = None,
              epochs = 10 ** 4,
              batch_size = None,
              tol = 1e-5,
              # print_rate    = 200,
              saving_rate   = 500,
              resampling_rate = 200,
              grad_upd_rate = 200,
              filename = None,
              w_pde_update_rate = 1e-4,
              ):
        
        print("\n>>>>> training setting;")
        print("         # of epoch     :", epochs)
        print("         batch size     :", batch_size)
        print("         convergence tol:", tol)
        
        t = t - time_base
        t_val = t_val - time_base 
        
        # optimization
        self.optimizer = self.opt_(self.lr, self.opt, epochs)
        
        x, y, z = lla2xyh(lat, lon, alt, self.lat_ref, self.lon_ref, self.alt_ref)
        x_val, y_val, z_val = lla2xyh(lat_val, lon_val, alt_val, self.lat_ref, self.lon_ref, self.alt_ref)
        
        # Training dataset
        t = tf.convert_to_tensor(t.reshape(-1,1), data_type)
        x = tf.convert_to_tensor(x.reshape(-1,1), data_type)
        y = tf.convert_to_tensor(y.reshape(-1,1), data_type)
        z = tf.convert_to_tensor(z.reshape(-1,1), data_type)
        
        d  = tf.convert_to_tensor(d.reshape(-1,1), data_type)
        k  = tf.convert_to_tensor(k.reshape(-1,3), data_type)
        
        if d_err is None:
            d_err = np.ones_like(d)
            
        d_err  = tf.convert_to_tensor(d_err.reshape(-1,1), data_type)
        
        #Validation dataset
        t_val = tf.convert_to_tensor(t_val.reshape(-1,1), data_type)
        x_val = tf.convert_to_tensor(x_val.reshape(-1,1), data_type)
        y_val = tf.convert_to_tensor(y_val.reshape(-1,1), data_type)
        z_val = tf.convert_to_tensor(z_val.reshape(-1,1), data_type)
        
        u_val  = tf.convert_to_tensor(u_val.reshape(-1,1), data_type)
        v_val  = tf.convert_to_tensor(v_val.reshape(-1,1), data_type)
        w_val  = tf.convert_to_tensor(w_val.reshape(-1,1), data_type)
        
        # bounds (for feature scaling)
        X_data  = tf.concat([t, z, x, y, d, k, d_err], axis=1)
        lb = tf.reduce_min (X_data, axis = 0)[:4]
        ub = tf.reduce_max (X_data, axis = 0)[:4]
        
        # lb = tf.constant([lb[0].numpy(),  80e3,-150e3,-150e3])
        # ub = tf.constant([ub[0].numpy(), 100e3, 150e3, 150e3])
        
        # rb = tf.multiply( (ub - lb), [0.,1.,1.,1.])
        
        self.lbi = lb #+ rb*0.1
        self.ubi = ub #- rb*0.1
        
        self.mn = tf.reduce_mean(X_data, axis = 0)[:4]
        
        print("         lower bounds    :", self.lbi.numpy())
        print("         upper bounds    :", self.ubi.numpy())
        
        w_data  = tf.convert_to_tensor(self.w_data, data_type)
        w_div   = tf.convert_to_tensor(self.w_div, data_type)
        w_mom   = tf.convert_to_tensor(self.w_mom, data_type)
        w_temp  = tf.convert_to_tensor(self.w_temp, data_type)
        w_srt   = tf.convert_to_tensor(self.w_srt, data_type)
        
        w_pde_update_rate = tf.convert_to_tensor(w_pde_update_rate, data_type)
        
        
        self.ep_log         = []
        
        self.loss_log       = []
        self.loss_data_log  = []
        self.loss_div_log   = []
        self.loss_mom_log   = []
        self.loss_temp_log  = []
        self.loss_srt_log   = []
        
        self.rmse_u_log     = []
        self.rmse_v_log     = []
        self.rmse_w_log     = []
        
        self.tv_u_log     = []
        self.tv_v_log     = []
        self.tv_w_log     = []
        
        self.tv_ue_log     = []
        self.tv_ve_log     = []
        self.tv_we_log     = []
        
        self.w_div_log = []
        self.w_mom_log = []
        self.w_temp_log = []
        self.w_srt_log = []
        
        self.set_LHS_samples()
        
        # w_pde_update_rate   = 1e-4
        w_div_final         = None#1e3
        # w_data = tf.constant(1e-1, dtype=data_type)
        
        self.train_batch(X_data,
                         epochs,
                         w_data,
                         w_div,
                         w_mom,
                         w_temp,
                         w_srt,
                         grad_update_steps=grad_upd_rate,
                         w_pde_update_rate=w_pde_update_rate,
                         filename=filename,
                         w_div_final=w_div_final,
                         t=t_val, x=x_val, y=y_val, z=z_val,
                         u=u_val, v=v_val, w=w_val
                         )
        # self.save(filename, 0)
        
        #######Train in batches of 1h
        trange = 3*60*60
        
        
        # w_pde_update_rate = 1e-2
        # w_data = tf.constant(1e-1, dtype=data_type)
        # w_div = tf.constant(1e7, dtype=data_type)
        
        # w_div_final         = 1e9
        # w_pde_update_rate   = 1e-2
        # epochs              = 201
        #
        # for i in range(20):
        #     ep = 1
        #     tmin = self.lbi[0].numpy() - trange
        #     while True:
        #
        #         tmin = tmin + trange
        #         tmax = tmin + trange
        #
        #         if tmin > self.ubi[0]:
        #             break
        #
        #         self.set_LHS_samples(tmin, tmax)
        #
        #         # ind = tf.where( (t>tmin) & (t<tmax))
        #         # ini = ind[0,0]
        #         # fin = ind[-1,0]
        #         #
        #         # X_subset = X_data[ini:fin+1,:]
        #
        #         self.train_batch(X_data,
        #                          epochs,
        #                          w_data,
        #                          w_div,
        #                          w_mom,
        #                          w_temp,
        #                          w_srt,
        #                          grad_update_steps=grad_upd_rate,
        #                          w_pde_update_rate=w_pde_update_rate,
        #                          w_div_final=w_div_final,
        #                          t=t_val, x=x_val, y=y_val, z=z_val,
        #                          u=u_val, v=v_val, w=w_val
        #                          )
        #
        #         self.save(filename, i*epochs + ep)
        #         ep = ep + 1
        
        # self.set_LHS_samples()
        #
        # self.train_batch(X_data,
        #                  5*epochs,
        #                  w_data,
        #                  w_div,
        #                  w_mom,
        #                  w_temp,
        #                  w_srt,
        #                  grad_update_steps=grad_upd_rate,
        #                  w_pde_update_rate=w_pde_update_rate,
        #                  t=t_val, x=x_val, y=y_val, z=z_val,
        #                  u=u_val, v=v_val, w=w_val
        #                  )
    
    def train_batch(self,
                    X_data,
                    epochs,
                    w_data=1.0,
                    w_div=1.0,
                    w_mom=1.0,
                    w_temp=1.0,
                    w_srt=1.0,
                    grad_update_steps=200,
                    w_pde_update_rate=1e-4,
                    filename=None,
                    w_div_final=None,
                    **kwargs
                    ):
        
        t0 = time.time()
        
        calc_gradients = False
        
        ep_loss     = 0.
        ep_loss_data = 0.
        ep_loss_div = 0.
        ep_loss_mom = 0.
        ep_loss_temp = 0.
        
        ep_loss_u     = 0.
        ep_loss_v     = 0.
        ep_loss_w     = 0.
            
        # ep_grad      = 0.
        ep_grad_data = 0.
        ep_grad_div  = 0.
        ep_grad_mom  = 0.
        ep_grad_temp = 0.
        
        for ep in range(epochs):
            
            X_pde = self.gen_LHS_samples()
            
            # if ep % grad_update_steps == 0:
            #     X_pde = self.adaptive_pde_sampling()
                
            # self.model.set_mask(ep, epochs)
                
                
                # for d in range(self.depth):
                #     self.alphas[d].assign(1.01*self.alphas[d])
                
            
            print(".", end='', flush=True)
            
            losses, grads = self.train_epoch(X_data,
                                             X_pde,
                                             w_data=w_data,
                                             w_div=w_div,
                                             w_mom=w_mom,
                                             w_temp=w_temp,
                                             w_srt=w_srt,
                                             calc_gradients=calc_gradients,
                                             )
            
            if grads is not None:
                #Update gradient values
                # ep_grad         = grads[0]
                ep_grad_data    = grads[1]
                ep_grad_div     = grads[2]
                ep_grad_mom     = grads[3]
                ep_grad_temp    = grads[4]
                # ep_grad_srt     = grads[5]
                    
            ep_loss         = losses[0]
            ep_loss_data    = losses[1]
            ep_loss_div     = losses[2]
            ep_loss_mom     = losses[3]
            ep_loss_temp    = losses[4]
            ep_loss_srt     = losses[5]
            
            ep_loss_u = np.nan
            ep_loss_v = np.nan
            ep_loss_w = np.nan
            
            tv_u = np.nan
            tv_v = np.nan
            tv_w = np.nan
            
            tv_ue = np.nan
            tv_ve = np.nan
            tv_we = np.nan
            
            if ep % grad_update_steps == 1:
                rmses = self.rmse(**kwargs)
                
                ep_loss_u = rmses[0].numpy()
                ep_loss_v = rmses[1].numpy()
                ep_loss_w = rmses[2].numpy()
                
                tv_u = rmses[3].numpy()
                tv_v = rmses[4].numpy()
                tv_w = rmses[5].numpy()
                
                tv_ue = rmses[6].numpy()
                tv_ve = rmses[7].numpy()
                tv_we = rmses[8].numpy()
                
                elps = time.time() - t0
            
                t0 = time.time()
                
                print("\nepoch: %d, elps: %ds" #, nu_scaling: %.2e, rho_scaling: %.2e" 
                    % (ep, 
                       elps,
                       # self.nu_scaling,
                       # self.rho_scaling,
                       )
                    )
                
                print("\t\t\ttotal \tdata \tdiv  \tmom  \tdiv_vort " )
                 
                 
                print("\tlosses : \t%.1e\t%.1e\t%.1e\t%.1e\t%.1e" 
                    % (
                       ep_loss,
                       ep_loss_data,
                       ep_loss_div,
                       ep_loss_mom,
                       ep_loss_temp,
                       # ep_loss_srt
                       )
                    )
                
                print("\tweights: \t\t%.1e\t%.1e\t%.1e\t%.1e" 
                    % (
                       w_data,
                       w_div,
                       w_mom,
                       w_temp,
                       # w_srt
                       )
                    )
                
                print("\tgrads  : \t\t%.1e\t%.1e\t%.1e\t%.1e" 
                    % (
                       ep_grad_data,
                       ep_grad_div,
                       ep_grad_mom,
                       ep_grad_temp,
                       # ep_grad_srt
                       )
                    )
                
                print("\t\t\tu \t\tv \t\tw" )
                
                print("\tTV  : \t\t%.2e \t%.2e \t%.2e" 
                    % (tv_u,
                       tv_v,
                       tv_w,
                       )
                    )
                
                print("\tTVe : \t\t%.2e \t%.2e \t%.2e" 
                    % (tv_ue,
                       tv_ve,
                       tv_we,
                       )
                    )
                
                print("\trmse: \t\t%.2e \t%.2e \t%.2e" 
                    % (ep_loss_u,
                       ep_loss_v,
                       ep_loss_w,
                       )
                    )
            
            #Save logs
            self.ep_log.append(ep)
            
            self.loss_log.append(ep_loss.numpy())
            self.loss_data_log.append(ep_loss_data.numpy())
            
            self.loss_div_log.append(ep_loss_div.numpy())
            self.loss_mom_log.append(ep_loss_mom.numpy())
            self.loss_temp_log.append(ep_loss_temp.numpy())
            self.loss_srt_log.append(ep_loss_srt.numpy())
            
            self.w_div_log.append(w_div.numpy())
            self.w_mom_log.append(w_mom.numpy())
            self.w_temp_log.append(w_temp.numpy())
            self.w_srt_log.append(w_srt.numpy())
            
            self.rmse_u_log.append(ep_loss_u)
            self.rmse_v_log.append(ep_loss_v)
            self.rmse_w_log.append(ep_loss_w)
            
            self.tv_u_log.append(tv_u)
            self.tv_v_log.append(tv_v)
            self.tv_w_log.append(tv_w)
            
            self.tv_ue_log.append(tv_ue)
            self.tv_ve_log.append(tv_ve)
            self.tv_we_log.append(tv_we)
            
            # w_pde_update_rate_ = np.exp(20*(ep/epochs - 1))
            
            w_pde_update_rate_ = w_pde_update_rate
                
            if ep < 200:
                continue
            
            #Adaptive PDE weights
            if w_div_final is not None:
                w_div  = (1.0-w_pde_update_rate_)*w_div  + w_pde_update_rate_*w_div_final
                w_mom  = w_div#*1e6 
                w_temp = w_div#*1e6
            else:
                if ep_grad_div != 0:
                    w_div  = (1.0-w_pde_update_rate_)*w_div  + w_pde_update_rate_*ep_grad_data/ep_grad_div
                
                if ep_grad_mom != 0:
                    w_mom  = (1.0-w_pde_update_rate_)*w_mom  + w_pde_update_rate_*ep_grad_data/ep_grad_mom
                
                if ep_grad_temp != 0:
                    w_temp = (1.0-w_pde_update_rate_)*w_temp + w_pde_update_rate_*ep_grad_data/ep_grad_temp
            
            # if ep_grad_srt != 0:
            #     w_srt = (1.0-w_pde_update_rate_)*w_srt   + 1e-2*w_pde_update_rate_*ep_grad_data/ep_grad_srt

            calc_gradients = False
            if ep % grad_update_steps == 0:
                calc_gradients = True
                self.save(filename, ep)
        
        print("\n************************************************************")
        print("*****************     MAIN PROGRAM END     *****************")
        print("************************************************************")
        print(">>>>> end time:", datetime.datetime.now())
        
    def invalid_mask(self, t, x, y, z):
        
        x0 = (self.ubi[2] +  self.lbi[2])/2.0
        y0 = (self.ubi[3] +  self.lbi[3])/2.0
        # z0 = (self.ubi[1] +  self.lbi[1])/2.0
        
        rx = (self.ubi[2] -  self.lbi[2])/2.0
        ry = (self.ubi[3] -  self.lbi[3])/2.0
        # rz = (self.ubi[1] -  self.lbi[1])/2.0
        
        cond = np.sqrt( ((x-x0)/rx)**2 + ((y-y0)/ry)**2)# + ((z-z0)/rz)**2 )
        mask = (cond > 1.0)
        
        mask |= (t<self.lbi[0]) | (t>self.ubi[0])
        # mask |= (x<self.lbi[2]) | (x>self.ubi[2])
        # mask |= (y<self.lbi[3]) | (y>self.ubi[3])
        mask |= (z<self.lbi[1]) | (z>self.ubi[1])
        
        return(mask)
        
    def infer(self, t, lon=None, lat=None, alt=None, filter_output=True,
              x=None, y=None, z=None, return_xyz=False):
        
        t = t - time_base
        
        if (x is None) or (y is None) or (z is None):
            x, y, z = lla2xyh(lat, lon, alt, self.lat_ref, self.lon_ref, self.alt_ref)
        
        mask = self.invalid_mask(t, x, y, z)
        
        t = tf.convert_to_tensor(t.reshape(-1,1), dtype = data_type)
        x = tf.convert_to_tensor(x.reshape(-1,1), dtype = data_type)
        y = tf.convert_to_tensor(y.reshape(-1,1), dtype = data_type)
        z = tf.convert_to_tensor(z.reshape(-1,1), dtype = data_type)
        
        outputs = self.forward_pass(self.model, tf.concat([t, z, x, y], axis=1), training=False)
        outputs = outputs.numpy()
        
        if filter_output:
            outputs[mask,:] = np.nan
        
        if return_xyz:
            return (outputs, x, y, z)
            
        # u = outputs[:,0]
        # v = outputs[:,1]
        # w = outputs[:,2]
        #
        # p = None
        # theta = None
        #
        # if outputs.shape[1] > 3: p = outputs[:,3]
        # if outputs.shape[1] > 4: theta = outputs[:,4]
        
        return (outputs)
    
    def infer_gradients(self, t, lon=None, lat=None, alt=None, filter_output=True,
                        x=None, y=None, z=None):
        
        t = t - time_base
        
        if (x is None) or (y is None) or (z is None):
            x, y, z = lla2xyh(lat, lon, alt, self.lat_ref, self.lon_ref, self.alt_ref)
        
        mask = self.invalid_mask(t, x, y, z)
        
        t = tf.convert_to_tensor(t.reshape(-1,1), dtype = data_type)
        x = tf.convert_to_tensor(x.reshape(-1,1), dtype = data_type)
        y = tf.convert_to_tensor(y.reshape(-1,1), dtype = data_type)
        z = tf.convert_to_tensor(z.reshape(-1,1), dtype = data_type)
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
            
            # tp.watch(t)
            tp.watch(x)
            tp.watch(y)
            tp.watch(z)
            
            outs = self.forward_pass(self.model, tf.concat([t, z, x, y], axis=1), training=False)
            
            u = outs[:,0:1]
            v = outs[:,1:2]
            w = outs[:,2:3]
        
        # u_t = tp.gradient(u, t)
        # v_t = tp.gradient(v, t)
        # w_t = tp.gradient(w, t)
        
        u_x = tp.gradient(u, x) #from m/s/m to m/s/km
        u_y = tp.gradient(u, y)
        u_z = tp.gradient(u, z)
        
        v_x = tp.gradient(v, x)
        v_y = tp.gradient(v, y)
        v_z = tp.gradient(v, z)
        
        w_x = tp.gradient(w, x)
        w_y = tp.gradient(w, y)
        w_z = tp.gradient(w, z)
        
        del tp
        
        Y = tf.concat([u, v, w,
                       u_x, v_x, w_x, 
                       u_y, v_y, w_y, 
                       u_z, v_z, w_z,
                       # u_t, v_t, w_t,
                       ],
                       axis=1)
        
        Y = Y.numpy()
        
        if filter_output:
            Y[mask,:] = np.nan
        
        return (Y)
    
    def infer_2nd_gradients(self, t, lon=None, lat=None, alt=None, return_numpy=True,
                        x=None, y=None, z=None):
        
        t = t - time_base
        
        if (x is None) or (y is None) or (z is None):
            x, y, z = lla2xyh(lat, lon, alt, self.lat_ref, self.lon_ref, self.alt_ref)
        
        mask = self.invalid_mask(t, x, y, z)
        
        t = tf.convert_to_tensor(t.reshape(-1,1), dtype = data_type)
        x = tf.convert_to_tensor(x.reshape(-1,1), dtype = data_type)
        y = tf.convert_to_tensor(y.reshape(-1,1), dtype = data_type)
        z = tf.convert_to_tensor(z.reshape(-1,1), dtype = data_type)
        
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp2:
            tp2.watch(x)
            tp2.watch(y)
            tp2.watch(z)
                
            with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
                
                tp.watch(x)
                tp.watch(y)
                tp.watch(z)
                
                outs = self.forward_pass(self.model, tf.concat([t, z, x, y], axis=1), training=False)
                
                u = outs[:,0:1]
                v = outs[:,1:2]
                w = outs[:,2:3]
                    
            u_x = tp.gradient(u, x) #from m/s/m to m/s/km
            # u_y = tp.gradient(u, y)
            # u_z = tp.gradient(u, z)
            
            # v_x = tp.gradient(v, x)
            v_y = tp.gradient(v, y)
            # v_z = tp.gradient(v, z)
            
            # w_x = tp.gradient(w, x)
            # w_y = tp.gradient(w, y)
            w_z = tp.gradient(w, z)
            
            del tp
        
        u_xx = tp2.gradient(u_x, x) #from m/s/m^2 to m/s/km^2
        v_yy = tp2.gradient(v_y, y)
        w_zz = tp2.gradient(w_z, z)
        
        del tp2
        
        Y = tf.concat([u, v, w,
                       u_x, v_y, w_z,
                       u_xx, v_yy, w_zz],
                       axis=1)
        
        if return_numpy:
            Y = Y.numpy()
            # Y[mask] = np.nan
        
        return (Y)
    
    @tf.function
    def rmse(self, t, z, x, y, u, v, w, sigma=tf.constant(1.0, dtype=data_type)):
        
        if tf.math.reduce_all(tf.math.is_nan(u)) == True:
            return (tf.constant(np.nan, dtype=data_type), tf.constant(np.nan, dtype=data_type), tf.constant(np.nan, dtype=data_type), tf.constant(np.nan, dtype=data_type), tf.constant(np.nan, dtype=data_type), tf.constant(np.nan, dtype=data_type), tf.constant(np.nan, dtype=data_type), tf.constant(np.nan, dtype=data_type), tf.constant(np.nan, dtype=data_type))
            
        outputs = self.forward_pass(self.model, tf.concat([t, z, x, y], axis=1), training=False)
        
        ue = outputs[:,0:1]
        ve = outputs[:,1:2]
        we = outputs[:,2:3]
        
        # u2 = 1e2#tf.reduce_mean(tf.square(u))
        # v2 = 1e2#tf.reduce_mean(tf.square(v))
        # w2 = 1e2#tf.reduce_mean(tf.square(w))
        
        rmse_u = tf.sqrt( tf.reduce_mean(tf.square((ue-u))) )
        rmse_v = tf.sqrt( tf.reduce_mean(tf.square((ve-v))) )
        rmse_w = tf.sqrt( tf.reduce_mean(tf.square((we-w))) )
        
        TV_u = tf.reduce_mean( tf.abs(u[1:,:] - u[:-1,:]) )
        TV_v = tf.reduce_mean( tf.abs(v[1:,:] - v[:-1,:]) )
        TV_w = tf.reduce_mean( tf.abs(w[1:,:] - w[:-1,:]) )
        
        TV_ue = tf.reduce_mean( tf.abs(ue[1:,:] - ue[:-1,:]) )
        TV_ve = tf.reduce_mean( tf.abs(ve[1:,:] - ve[:-1,:]) )
        TV_we = tf.reduce_mean( tf.abs(we[1:,:] - we[:-1,:]) )
        
        # op, op_a = tf.contrib.metrics.steraming_pearson_correlation(u, ue)
        
        return (rmse_u,rmse_v,rmse_w, TV_u, TV_v, TV_w, TV_ue, TV_ve, TV_we)
        
    def save(self, filename, log_index=None):
        
        #Save model
        # filename_log =  filename
        
        file_weights = get_log_file(filename, log_index)
            
        self.model.save_weights(file_weights)#, save_format="tf")
        
        #Save parms
        with h5py.File(filename,'w') as fp:
            
            fp['shape_in'] = self.shape_in
            fp['shape_out'] = self.shape_out
            
            fp['width'] = self.width
            fp['depth'] = self.depth
            fp['nnodes'] = self.nnodes
            fp['nblocks'] = self.nblocks
            
            fp['nn_type']  = self.nn_type
            
            fp['ns_pde'] = self.N_pde
            fp['r_seed'] = self.r_seed
            
            fp['act'] = self.act
            fp['opt'] = self.opt
            fp['f_scl_in'] = self.f_scl_in    # "minmax"
            
            fp['laaf'] = self.laaf
            
            fp['lr'] = self.lr
            fp['ini_w_data'] = self.w_data
            
            fp['NS_type']   = self.NS_type
            
            fp['ini_w_Ph1'] = self.w_div
            fp['ini_w_Ph2'] = self.w_mom
            fp['ini_w_Ph3'] = self.w_temp
            fp['ini_w_Ph4'] = self.w_srt
            
            
            # fp['f_scl_out'] = self.f_scl_out  # 1
            
            fp['lb'] = self.lbi
            fp['ub'] = self.ubi
            fp['mn'] = self.mn
            
            fp['lon_ref'] = self.lon_ref
            fp['lat_ref'] = self.lat_ref
            fp['alt_ref'] = self.alt_ref
            
            fp['ep_log'] = self.ep_log
            
            fp['loss_log'] = self.loss_log
            fp['loss_data_log'] = self.loss_data_log
            
            fp['loss_Ph1_log'] = self.loss_div_log
            fp['loss_Ph2_log'] = self.loss_mom_log
            fp['loss_Ph3_log'] = self.loss_temp_log
            fp['loss_Ph4_log'] = self.loss_srt_log
            
            fp['rmse_u_log'] = self.rmse_u_log
            fp['rmse_v_log'] = self.rmse_v_log
            fp['rmse_w_log'] = self.rmse_w_log
            
            fp['tv_u_log'] = self.tv_u_log
            fp['tv_v_log'] = self.tv_v_log
            fp['tv_w_log'] = self.tv_w_log
            
            fp['tv_ue_log'] = self.tv_ue_log
            fp['tv_ve_log'] = self.tv_ve_log
            fp['tv_we_log'] = self.tv_we_log
            
            fp['w_Ph1_log'] = self.w_div_log
            fp['w_Ph2_log'] = self.w_mom_log
            fp['w_Ph3_log'] = self.w_temp_log
            fp['w_Ph4_log'] = self.w_srt_log
            
            fp['dropout'] = self.dropout
            
            fp['residual_layer'] = self.residual_layer
            
            
            # fp['nu_scaling'] = self.nu_scaling
            # fp['rho_scaling'] = self.rho_scaling
            
            if self.ext_forcing is not None:
                fp['ext_forcing']  = self.ext_forcing
    
            
            # for k in range(self.nblocks):
            #
            #     g = fp.create_group('block_%02d' %k)
            #
            #     g_weights = g.create_group('weights')
            #     g_biases = g.create_group('biases')
            #     g_alphas = g.create_group('alphas')
            #
            #     block_nn   = self.blocks[k]
            #
            #     for i, w in enumerate(block_nn['weights']):
            #         g_weights['w%03d' %i]  = w
            #
            #     for i, b in enumerate(block_nn['biases']):
            #         g_biases['b%03d' %i]  = b
            #
            #     for i, a in enumerate(block_nn['alphas']):
            #         g_alphas['a%03d' %i]  = a
            
    
    # def restore(self, filename, log_index=None):
    #
    #     file_weights = get_log_file(filename, log_index)
    #
    #     keys_float  = ['lr',
    #                    # 'ini_w_data', 'ini_w_Ph1', 'ini_w_Ph2', 'ini_w_Ph3', 'ini_w_Ph4',
    #                    # 'f_scl_out',
    #                    'lb', 'ub', 'mn',
    #                    'lon_ref', 'lat_ref', 'alt_ref',
    #                    ]
    #
    #     keys_int    = ['shape_in', 'shape_out',
    #                    'width', 'depth',
    #                    'nnodes',
    #                    'ns_pde', 'r_seed',
    #                    'laaf',
    #                    'dropout',
    #                     'residual_layer',
    #
    #                    ]
    #
    #     keys_str    = ['act', 'opt', 'f_scl_in', 'nn_type']
    #
    #     kwargs = {}
    #
    #     print("Opening NN file: %s ..." %filename)
    #
    #     with h5py.File(filename,'r') as fp:
    #
    #         for key in keys_float:
    #             kwargs[key] = fp[key][()]
    #
    #         for key in keys_int:
    #
    #             if key not in fp.keys():
    #                     continue
    #
    #             kwargs[key] = int(fp[key][()])
    #
    #         for key in keys_str:
    #             try:
    #                 kwargs[key] = fp[key][()].decode('utf-8')
    #             except:
    #                 continue
    #
    #         ext_forcing = None
    #         if 'ext_forcing' in fp.keys():
    #             ext_forcing = fp['ext_forcing'][()]
    #
    #         self.loss_log = fp['loss_log'][()]
    #         self.loss_data_log = fp['loss_data_log'][()]
    #
    #         self.loss_div_log = fp['loss_Ph1_log'][()]
    #         self.loss_mom_log = fp['loss_Ph2_log'][()]
    #         self.loss_temp_log = fp['loss_Ph3_log'][()]
    #         self.loss_srt_log = fp['loss_Ph4_log'][()]
    #
    #         self.w_div_log = fp['w_Ph1_log'][()]
    #         self.w_mom_log = fp['w_Ph2_log'][()]
    #         self.w_temp_log = fp['w_Ph3_log'][()]
    #         self.w_srt_log = fp['w_Ph4_log'][()]
    #
    #         try:
    #             self.rmse_u_log = fp['rmse_u_log'][()]
    #             self.rmse_v_log = fp['rmse_v_log'][()]
    #             self.rmse_w_log = fp['rmse_w_log'][()]
    #
    #             self.tv_u_log = fp['tv_u_log'][()]
    #             self.tv_v_log = fp['tv_v_log'][()]
    #             self.tv_w_log = fp['tv_w_log'][()]
    #
    #             self.tv_ue_log = fp['tv_ue_log'][()]
    #             self.tv_ve_log = fp['tv_ve_log'][()]
    #             self.tv_we_log = fp['tv_we_log'][()]
    #
    #         except:
    #             nan_array = np.array(self.w_div_log) + np.nan
    #             self.rmse_u_log = nan_array
    #             self.rmse_v_log = nan_array
    #             self.rmse_w_log = nan_array
    #
    #             self.tv_u_log = nan_array
    #             self.tv_v_log = nan_array
    #             self.tv_w_log = nan_array
    #
    #             self.tv_ue_log = nan_array
    #             self.tv_ve_log = nan_array
    #             self.tv_we_log = nan_array
    #
    #         self.ep_log = fp['ep_log'][()]
    #
    #         # nblocks = kwargs['nblocks']
    #         #
    #         # blocks = []
    #         # for i in range(nblocks):
    #         #     g = fp['block_%02d' %i]
    #         #
    #         #     g_weights = g['weights']
    #         #     g_biases = g['biases']
    #         #     g_alphas = g['alphas']
    #         #
    #         #     weights = []
    #         #     for key in g_weights.keys():
    #         #         weights.append( g_weights[key][()] )
    #         #
    #         #     biases = []
    #         #     for key in g_biases.keys():
    #         #         biases.append( g_biases[key][()] )
    #         #
    #         #     alphas = []
    #         #     for key in g_alphas.keys():
    #         #         alphas.append( g_alphas[key][()] )
    #         #
    #
    #             # block = {}
    #             # block['weights'] = weights
    #             # block['biases'] = biases
    #             # block['alphas'] = alphas              
    #             #
    #             # blocks.append(block)
    #
    #     # nn_type = 'uni'
    #     # if weights[1].shape[0] != weights[1].shape[1]:
    #     #     nn_type = 'non_uni'
    #     #
    #     # kwargs['nn_type'] = nn_type
    #
    #     self.__init__(**kwargs)
    #
    #     print("\nLoading weight file: %s ..." %file_weights)
    #     self.model.load_weights(file_weights)#, by_name=True, skip_mismatch=True)
    #
    #     # for k in range(self.nblocks):
    #     #
    #     #     block_file = blocks[k]
    #     #     block_nn   = self.blocks[k]
    #     #
    #     #     for i, w in enumerate(block_file['weights']):
    #     #         block_nn['weights'][i].assign(w)
    #     #
    #     #     for i, b in enumerate(block_file['biases']):
    #     #         block_nn['biases'][i].assign(b)
    #     #
    #     #     for i, a in enumerate(block_file['alphas']):
    #     #         try:
    #     #             block_nn['alphas'][i].assign(a)
    #     #         except AttributeError:
    #     #             continue
    #
    #     if self.ext_forcing is not None:
    #         self.ext_forcing.assign(ext_forcing)
    #
    #     #ext_force 
    #     self.lbi = tf.convert_to_tensor(kwargs['lb'], dtype = data_type)
    #     self.ubi = tf.convert_to_tensor(kwargs['ub'], dtype = data_type)
    #     self.mn = tf.convert_to_tensor(kwargs['mn'], dtype = data_type)
    #
    #     print("         lower bounds    :", self.lbi.numpy())
    #     print("         upper bounds    :", self.ubi.numpy())
    #
    #     return
    
    @tf.function
    def discontinuity(self, X):
        
        t, z, x, y, _, _, _, _ = tf.split(X, num_or_size_splits=8, axis=-1)
            
        with tf.GradientTape(persistent = True, watch_accessed_variables=False) as tp:
            
            tp.watch(t)
            tp.watch(x)
            tp.watch(y)
            tp.watch(z)
            
            outs = self.forward_pass(self.model, tf.concat([t, z, x, y], axis=1))
            
            u = outs[:,0:1]
            v = outs[:,1:2]
            w = outs[:,2:3]
        
        u_t = tp.gradient(u, t)
        v_t = tp.gradient(v, t)
        w_t = tp.gradient(w, t)
        
        u_x = tp.gradient(u, x)
        u_y = tp.gradient(u, y)
        u_z = tp.gradient(u, z)
        
        v_x = tp.gradient(v, x)
        v_y = tp.gradient(v, y)
        v_z = tp.gradient(v, z)
        
        w_x = tp.gradient(w, x)
        w_y = tp.gradient(w, y)
        w_z = tp.gradient(w, z)
        
        
        del tp
        
        d_x     = tf.abs(u_t) + tf.abs(u_x) + tf.abs(u_y) + tf.abs(u_z)*1e-2
        d_y     = tf.abs(v_t) + tf.abs(v_x) + tf.abs(v_y) + tf.abs(v_z)*1e-2
        d_z     = tf.abs(w_t) + tf.abs(w_x) + tf.abs(w_y) + tf.abs(w_z)*1e-2
            
        d_total = d_x + d_y + d_z
        
        return(d_total)
    
    def _set_LHS_samples(self, tmin=None, tmax=None, mult=1.0):
        
        """
        Raissi et al 2019. JCP
        """
        
        if tmin is None: tmin = self.lbi[0].numpy()
        if tmax is None: tmax = self.ubi[0].numpy()
        
        xmin = self.lbi[2].numpy()
        xmax = self.ubi[2].numpy()
        
        ymin = self.lbi[3].numpy()
        ymax = self.ubi[3].numpy()
        
        zmin = self.lbi[1].numpy()
        zmax = self.ubi[1].numpy()
        
        dx = 50e3*mult
        dy = 50e3*mult
        dz = 5e3*mult
        dt = 60*60*mult
        
        #Set number of intervals
        Nx  = int( (xmax-xmin)/dx )
        Ny  = int( (ymax-ymin)/dy )
        Nz  = int( (zmax-zmin)/dz )
        Nt  = int( (tmax-tmin)/dt ) #A sample every 15min
        
        x0 = (np.arange(Nx, dtype=np.float32) - (Nx-1.0)/2.0)*dx + (xmax+xmin)/2.0
        y0 = (np.arange(Ny, dtype=np.float32) - (Ny-1.0)/2.0)*dy + (ymax+ymin)/2.0
        z0 = (np.arange(Nz, dtype=np.float32) - (Nz-1.0)/2.0)*dz + (zmax+zmin)/2.0
        t0 = (np.arange(Nt, dtype=np.float32) - (Nt-1.0)/2.0)*dt + (tmax+tmin)/2.0
        
        # T0, Z0, X0, Y0 = np.meshgrid(t0, z0, x0, y0, indexing='ij')
        
        # mask = self._invalid_volumen(T0, X0, Y0, Z0)
        # valid = ~mask
        
        self.lhs_dx = dx
        self.lhs_dy = dy
        self.lhs_dz = dz
        self.lhs_dt = dt
        
        self.lhs_x0 = x0 #X0[valid]
        self.lhs_y0 = y0 #Y0[valid]
        self.lhs_z0 = z0 #Z0[valid]
        self.lhs_t0 = t0 #T0[valid]
        
        self.lhs_nx = Nx
        self.lhs_ny = Ny
        self.lhs_nz = Nz
        self.lhs_nt = Nt
        
        self.pde_xc = (xmax+xmin)/2.0
        self.pde_yc = (ymax+ymin)/2.0
        self.pde_zc = (zmax+zmin)/2.0
        self.pde_tc = (tmax+tmin)/2.0
        
        self.pde_xw = (xmax-xmin)/2.0
        self.pde_yw = (ymax-ymin)/2.0
        self.pde_zw = (zmax-zmin)/2.0
        self.pde_tw = (tmax-tmin)/2.0
        
        self.lhs_N = len(self.lhs_x0)
        
        # print("\nNumber of collocation points (t,x,y,z): [%d, %d, %d, %d] = %d" %(Nt, Nx, Ny, Nz, Nt*Nx*Ny*Nz) )
        print("Time range: %e s - %e s = %e min" %(tmin, tmax, (tmax-tmin)/60.0) )
        
    def gen_LHS_samples(self):
        
        """
        Raissi et al 2019. JCP
        """
        
        shape = (self.lhs_nt, self.lhs_nz, self.lhs_nx, self.lhs_ny)
        
        ###Draw uniformly sampled collocation points
        # delta   = np.random.uniform(-0.5, 0.5, size=self.N_pde)
        # x       = self.lhs_x0 + delta*self.lhs_dx
        #
        # delta   = np.random.uniform(-0.5, 0.5, size=self.N_pde)
        # y       = self.lhs_y0 + delta*self.lhs_dy
        #
        # delta   = np.random.uniform(-0.5, 0.5, size=self.N_pde)
        # z       = self.lhs_z0 + delta*self.lhs_dz
        #
        # delta   = np.random.uniform(-0.5, 0.5, size=self.N_pde)
        # t       = self.lhs_t0 + delta*self.lhs_dt
        
        delta   = np.random.uniform(-1.0, 1.0, size=self.N_pde)
        x = self.pde_xw*delta + self.pde_xc
        
        delta   = np.random.uniform(-1.0, 1.0, size=self.N_pde)
        y = self.pde_yw*delta + self.pde_yc
        
        delta   = np.random.uniform(-1.0, 1.0, size=self.N_pde)
        z = self.pde_zw*delta + self.pde_zc
        
        delta   = np.random.uniform(-1.0, 1.0, size=self.N_pde)
        t = self.pde_tw*delta + self.pde_tc
        
        # ind4d = np.unravel_index(idx, shape)
        
        # idx = np.random.choice(self.lhs_N, size=self.N_pde, replace=False)
        # t = self.lhs_t0[idx]
        # z = self.lhs_z0[idx]
        # x = self.lhs_x0[idx]
        # y = self.lhs_y0[idx]
        
        t = t.reshape(-1,1)
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)
        
        if self.msis is None:
            # zeros = tf.zeros_like(z)
            zeros = tf.constant(0.0, shape=(self.N_pde,1))
            ones = tf.constant(1.0, shape=(self.N_pde,1))
            
            N       = ones
            nu      = 100*ones
            rho     = ones
            rho_z   = zeros
            rho_ratio   = zeros
        else:
            N       = self.msis.get_N(z)
            nu      = self.msis.get_nu(z)
            rho     = self.msis.get_rho(z)
            rho_z   = self.msis.get_rho_z(z)
            rho_ratio = rho_z/rho
        
            N       = tf.convert_to_tensor(N.reshape(-1,1), dtype=data_type)
            nu      = tf.convert_to_tensor(nu.reshape(-1,1), dtype=data_type)
            rho     = tf.convert_to_tensor(rho.reshape(-1,1), dtype=data_type)
            rho_z   = tf.convert_to_tensor(rho_z.reshape(-1,1), dtype=data_type)
            rho_ratio = tf.convert_to_tensor(rho_ratio.reshape(-1,1), dtype=data_type)
        
        X_pde = tf.concat((t, z,
                           x, y,
                           nu, rho, rho_ratio, N,
                           ),
                           axis=1)
        
        return(X_pde)

    def set_Chebyshev_samples(self, tmin=None, tmax=None, mult=1.0):
        
        """
        Raissi et al 2019. JCP
        """
        
        if tmin is None: tmin = self.lbi[0].numpy()
        if tmax is None: tmax = self.ubi[0].numpy()
        
        xmin = self.lbi[2].numpy()
        xmax = self.ubi[2].numpy()
        
        ymin = self.lbi[3].numpy()
        ymax = self.ubi[3].numpy()
        
        zmin = self.lbi[1].numpy()
        zmax = self.ubi[1].numpy()
        
        dx = 5e3*mult
        dy = 5e3*mult
        dz = 5e2*mult
        dt = 5*60*mult
        
        #Set number of intervals
        Nx  = int( (xmax-xmin)/dx )
        Ny  = int( (ymax-ymin)/dy )
        Nz  = int( (zmax-zmin)/dz )
        Nt  = int( (tmax-tmin)/dt ) #A sample every 15min
        
        k = np.arange(Nx)
        x0 = (xmax-xmin)/2.0*np.cos( (2*k+1)/(2*Nx)*np.pi ) + (xmax+xmin)/2.0
        k = np.arange(Ny)
        y0 = (ymax-ymin)/2.0*np.cos( (2*k+1)/(2*Ny)*np.pi ) + (ymax+ymin)/2.0
        k = np.arange(Nz)
        z0 = (zmax-zmin)/2.0*np.cos( (2*k+1)/(2*Nz)*np.pi ) + (zmax+zmin)/2.0
        k = np.arange(Nx)
        t0 = (tmax-tmin)/2.0*np.cos( (2*k+1)/(2*Nt)*np.pi ) + (tmax+tmin)/2.0
        
        # T0, Z0, X0, Y0 = np.meshgrid(t0, z0, x0, y0, indexing='ij')
        
        # mask = self._invalid_volumen(T0, X0, Y0, Z0)
        # valid = ~mask
        
        self.lhs_dx = dx
        self.lhs_dy = dy
        self.lhs_dz = dz
        self.lhs_dt = dt
        
        self.lhs_x0 = x0 #X0[valid]
        self.lhs_y0 = y0 #Y0[valid]
        self.lhs_z0 = z0 #Z0[valid]
        self.lhs_t0 = t0 #T0[valid]
        
        self.lhs_nx = Nx
        self.lhs_ny = Ny
        self.lhs_nz = Nz
        self.lhs_nt = Nt
        
        self.pde_xc = (xmax+xmin)/2.0
        self.pde_yc = (ymax+ymin)/2.0
        self.pde_zc = (zmax+zmin)/2.0
        self.pde_tc = (tmax+tmin)/2.0
        
        self.pde_xw = (xmax-xmin)/2.0
        self.pde_yw = (ymax-ymin)/2.0
        self.pde_zw = (zmax-zmin)/2.0
        self.pde_tw = (tmax-tmin)/2.0
        
        self.lhs_N = len(self.lhs_x0)
        
        # print("\nNumber of collocation points (t,x,y,z): [%d, %d, %d, %d] = %d" %(Nt, Nx, Ny, Nz, Nt*Nx*Ny*Nz) )
        print("Time range: %e s - %e s = %e min" %(tmin, tmax, (tmax-tmin)/60.0) )
        
    def gen_Chebyshev_samples(self):
        
        """
        Raissi et al 2019. JCP
        """
        
        idx = np.random.choice(self.lhs_nx, size=self.N_pde, replace=True)
        idy = np.random.choice(self.lhs_ny, size=self.N_pde, replace=True)
        idz = np.random.choice(self.lhs_nz, size=self.N_pde, replace=True)
        idt = np.random.choice(self.lhs_nt, size=self.N_pde, replace=True)
        
        x = self.lhs_x0[idx]
        y = self.lhs_y0[idy]
        z = self.lhs_z0[idz]
        t = self.lhs_t0[idt]
        
        t = t.reshape(-1,1)
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)
        
        if self.msis is None:
            # zeros = tf.zeros_like(z)
            zeros = tf.constant(0.0, shape=(self.N_pde,1))
            ones = tf.constant(10.0, shape=(self.N_pde,1))
            
            N       = ones
            nu      = ones
            rho     = zeros
            rho_z   = zeros
            rho_ratio   = zeros
        else:
            N       = self.msis.get_N(z)
            nu      = self.msis.get_nu(z)
            rho     = self.msis.get_rho(z)
            rho_z   = self.msis.get_rho_z(z)
            rho_ratio = rho_z/rho
        
            N       = tf.convert_to_tensor(N.reshape(-1,1), dtype=data_type)
            nu      = tf.convert_to_tensor(nu.reshape(-1,1), dtype=data_type)
            rho     = tf.convert_to_tensor(rho.reshape(-1,1), dtype=data_type)
            rho_z   = tf.convert_to_tensor(rho_z.reshape(-1,1), dtype=data_type)
            rho_ratio = tf.convert_to_tensor(rho_ratio.reshape(-1,1), dtype=data_type)
        
        X_pde = tf.concat((t, z,
                           x, y,
                           nu, rho, rho_ratio, N,
                           ),
                           axis=1)
        
        return(X_pde)
    
    def adaptive_pde_sampling(self):
        
        #Replace 10% of the points with sampling points with high losses
        
        print('r', end='')
        
        # new_X_pde = self.gen_random_samples(self.N_pde)
        new_X_pde = self.gen_LHS_samples()
        
        if self.X_pde is None:
            self.X_pde = new_X_pde
            return(new_X_pde)
        
        weights = [self.w_div, self.w_mom, self.w_temp, self.w_temp, self.w_temp, self.w_srt]
        
        losses = self.pde(new_X_pde)
        loss_next = 0
        for i in range(6):
            loss_next += tf.square(losses[i])*weights[i]
            
        losses = self.pde(self.X_pde)
        loss_current = 0
        for i in range(6):
            loss_current += tf.square(losses[i])*weights[i]
        
        #Keep the sample points with higher losses
        X_pde = tf.where(loss_next > loss_current, new_X_pde, self.X_pde)
    
        # self.plot_pde_samples(self.t_pde, self.x_pde, self.y_pde, self.z_pde)
        
        self.X_pde = X_pde
        
        return(X_pde)
    
    def plot_pde_samples(self, t, x, y, z):
        
        # print("         lower bounds:", tf.reduce_min(self.t_pde).numpy(), tf.reduce_min(self.z_pde).numpy())
        # print("         upper bounds:", tf.reduce_max(self.t_pde).numpy(), tf.reduce_max(self.z_pde).numpy())
        
        self.counter += 1
        figname = 'fig_pde_sampling_%03d.png' %self.counter
        
        plt.subplot(121)
        plt.plot(t.numpy(), z.numpy(), '.', alpha=0.1)
        plt.xlim(self.lbi[0].numpy(), self.ubi[0].numpy())
        plt.ylim(self.lbi[1].numpy(), self.ubi[1].numpy())
        plt.grid()
        
        plt.subplot(122)
        plt.plot(x.numpy(), y.numpy(), '.', alpha=0.1)
        plt.xlim(self.lbi[2].numpy(), self.ubi[2].numpy())
        plt.ylim(self.lbi[3].numpy(), self.ubi[3].numpy())
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()
        
        
    def plot_loss_history(self, figname='./loss_hist.png'):
        
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(121)
        
        epochs = np.arange( len(self.ep_log ) )
        
        total = np.array(self.loss_log)
        data = np.array(self.loss_data_log)
        
        scale_tot = 1e0
        scale_div = 1e2
        scale_mom = 1e2
        scale_vor = 1e2
        
        lamb_div = np.array(self.w_div_log)/(scale_div)
        lamb_mom = np.array(self.w_mom_log)/(scale_mom)
        lamb_vor = np.array(self.w_temp_log)/(scale_vor)
        
        scale_div_ = scale_div
        scale_mom_ = scale_mom
        scale_vor_ = scale_vor
        
        #Comment these lines since v300
        # scale_div_ = 1/lamb_div
        # scale_mom_ = 1/lamb_mom
        # scale_vor_ = 1/lamb_vor
        
        total = np.array(total, dtype=np.float64)*scale_tot
        data = np.array(data, dtype=np.float64)*scale_tot
        div = np.array(self.loss_div_log, dtype=np.float64)*scale_div_
        mom = np.array(self.loss_mom_log, dtype=np.float64)*scale_mom_
        vor = np.array(self.loss_temp_log, dtype=np.float64)*scale_vor_
        
        tv_u_log = np.array(self.tv_u_log)
        tv_v_log = np.array(self.tv_v_log)
        tv_w_log = np.array(self.tv_w_log)
        
        tv_ue_log = np.array(self.tv_ue_log)
        tv_ve_log = np.array(self.tv_ve_log)
        tv_we_log = np.array(self.tv_we_log)
        
        rmse_ue_log = np.array(self.rmse_u_log)
        rmse_ve_log = np.array(self.rmse_v_log)
        rmse_we_log = np.array(self.rmse_w_log)
        
        ax.semilogy(epochs, total, 'k-', label=r'$\mathcal{L}$: Total loss x %1g' %scale_tot, alpha=1.0)
        
        ax.semilogy(epochs, data, '--', label=r'$\mathcal{L}_d$ x %1g' %scale_tot, alpha=0.5, color='r')
        
        ax.semilogy(epochs, div, '--', label=r'$\mathcal{R}_{pde}$ x %1g' %scale_div, alpha=0.5, color='b')
        # ax.semilogy(epochs, mom, '--', label=r'$\mathcal{R}_{mom}$ x %1g' %scale_mom, alpha=0.5, color='g')
        # ax.semilogy(epochs, vor, '--', label=r'$\mathcal{R}_{vor}$ x %1g' %scale_vor, alpha=0.5, color='y')
        
        mask = np.isfinite(self.rmse_u_log)
        if np.any( mask ):
            
            ax.semilogy(epochs[mask], tv_u_log[mask],'r-.', label=r'$TV_{u}$', alpha=0.5)
            ax.semilogy(epochs[mask], tv_v_log[mask],'g-.', label=r'$TV_{v}$', alpha=0.5)
            ax.semilogy(epochs[mask], tv_w_log[mask],'b-.', label=r'$TV_{w}$', alpha=0.5)
            
            ax.semilogy(epochs[mask], tv_ue_log[mask],'r>', label=r'$TV_{ue}$', alpha=0.5)
            ax.semilogy(epochs[mask], tv_ve_log[mask],'g>', label=r'$TV_{ve}$', alpha=0.5)
            ax.semilogy(epochs[mask], tv_we_log[mask],'b>', label=r'$TV_{we}$', alpha=0.5)
            
            ax.semilogy(epochs[mask], rmse_ue_log[mask],'rs', label=r'$RMSE_{u}$', alpha=0.5)
            ax.semilogy(epochs[mask], rmse_ve_log[mask],'gs', label=r'$RMSE_{v}$', alpha=0.5)
            ax.semilogy(epochs[mask], rmse_we_log[mask],'bs', label=r'$RMSE_{w}$', alpha=0.5)
            
            # ax.semilogy(epochs, self.hist_P_true,'.-', label='True P loss', alpha=0.5)
        
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('RMSE')
        ax.set_title('(a) Training loss')
        
        ax.set_ylim(1e-2,1e3)
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle=(0, (1, 10)))
        ax.legend()
        
        # ax_w = ax.twinx()
        # ax_w.semilogy(epochs[::200], self.w_div_log[::200], 'g.', alpha=0.5, label='w_div')
        # ax_w.semilogy(epochs[::200], self.w_mom_log[::200], 'g.', alpha=0.5, label='w_mom')
        # ax_w.semilogy(epochs[::200], self.w_temp_log[::200] , 'g.', alpha=0.5, label='w_temp')
        #
        # ax_w.set_ylabel('weights', color='m')
        # ax_w.legend()
        
        ax = fig.add_subplot(122)
        
        ax.semilogy(epochs, lamb_div, 's--', label=r'$\lambda_1$/%1g' %scale_div, alpha=0.5, color='b')
        # ax.semilogy(epochs, lamb_mom, 'h--', label=r'$\lambda_2$/%1g' %scale_mom, alpha=0.5, color='g')
        # ax.semilogy(epochs, lamb_vor, '>--', label=r'$\lambda_3$/%1g' %scale_vor, alpha=0.5, color='y')
        
        ax.set_xlabel('Number of iterations')
        # ax.set_ylabel('Weighting coefficients')
        ax.set_title('(b) Weighting coefficients')
        
        ax.set_ylim(1e-3,1e3)
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle=(0, (1, 10)))
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()
        
        return
    
    def plot_solution(self, 
                      t, x, y, z,
                      u, v, w, d,
                      u_nn, v_nn, w_nn, d_nn,
                      P=None, P_nn=None,
                      k_x=None, k_y=None, k_z=None,
                      figname_winds='./winds.png',
                      figname_errs='./errors.png',
                      figname_errs_k='./errors_k.png',
                      figname_pressure='./pressure.png',
                      alpha=0.2, bins=40,
                      norm=False):
        
        if np.all(np.isnan(u)):
            return
        
        bins_d = np.linspace(np.nanmin(d), np.nanmax(d), bins)
        bins_u = np.linspace(np.nanmin(u), np.nanmax(u), bins)
        bins_v = np.linspace(np.nanmin(v), np.nanmax(v), bins)
        bins_w = np.linspace(np.nanmin(w), np.nanmax(w), bins)
        
        bins_k = np.arange(0,1.2,0.1)
        
        bins_dd = np.linspace(0, 5, bins)
        bins_du = np.linspace(0, 15, bins)
        bins_dw = np.linspace(0, 5, bins)
        
        #######################################################
        
        t_num = epoch2num(t)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
    
        xs = [u, v, w]#, d]
        ys = [u_nn, v_nn, w_nn, d_nn]
        
        xlabels = [r'$u_{true}$', r'$v_{true}$', r'$w_{true}$', r'$Doppler_{true}$', r'$Pressure_{true}$']
        ylabels = [r'$u_{PIML}$',     r'$v_{PIML}$',    r'$w_{PIML}$',    r'$Doppler_{PIML}$',  r'$Pressure_{PIML}$']
        bins_list = [bins_u, bins_v, bins_w, bins_d]
        
        if P_nn is not None:
            bins_P = np.linspace(np.nanmin(P), np.nanmax(P), bins)
            xs.append(P)
            ys.append(P_nn)
            bins_list.append(bins_P)
        
        nrows = 1
        ncols = len(xs)
        
        fig = plt.figure(figsize=(ncols*4,4.5))
        
        i = 0
        for j in range(ncols):
            
            parm_x = xs[j]
            parm_y = ys[j]
            
            xlabel = xlabels[j]
            ylabel = ylabels[j]
            
            bins_ = bins_list[j]
            
            axXY = plt.subplot2grid( (nrows*10-1,ncols*10-1), (i*10+1,j*10),        colspan=8, rowspan=8 )
            axX  = plt.subplot2grid( (nrows*10-1,ncols*10-1), (i*10,j*10),          colspan=8 )
            axY  = plt.subplot2grid( (nrows*10-1,ncols*10-1), (i*10+1,(j+1)*10-2),  rowspan=8 )
            
            ax_2dhist([axXY,axX,axY],
                      parm_x, parm_y,
                      bins=bins_,
                      labelX=xlabel,
                      labelY=ylabel,
                      norm=False)
            
            # axXY.label_outer()
        
        fig.subplots_adjust(0.05, 0.1, 0.98, 0.98,
                            wspace=0.1, hspace=0.1)
        
        plt.savefig(figname_winds)
        plt.close()
        
        bins = bins//2
        #########################################################
        #Errors
        delta_u = np.abs(u - u_nn)
        delta_v = np.abs(v - v_nn)
        delta_w = np.abs(w - w_nn)
        delta_dop = np.abs(d - d_nn)
    
        xs = [t_num, x, y, z]
        ys = [delta_u, delta_v, delta_w, delta_dop]
        
        bins_x = [bins, bins, bins, bins]
        bins_y = [bins_du, bins_du, bins_dw, bins_dd] 
        
        xlabels = ['UTC', 'Longitude', 'Latitude', 'Altitude']
        ylabels = [r'$|\Delta u|$', r'$|\Delta v|$', r'$|\Delta w|$', r'$|\Delta Dop|$']
        
        nrows = len(ys)
        ncols = len(xs)
        
        fig = plt.figure(figsize=(ncols*4,nrows*4))
        
        for i in range(nrows):
            for j in range(ncols):
                
                parm_x = xs[j]
                parm_y = ys[i]
                
                xlabel = xlabels[j]
                ylabel = ylabels[i]
                
                bins_ = [bins_x[j], bins_y[i]]
                
                # ax = plt.subplot(nrows, ncols, i*nrows+j+1)
                #
                # ax_2dhist_simple(parm_x, parm_y, ax, cmap=cmap,
                #                  bins=bins_,
                #                  )
                #
                # ax.set(xlabel=xlabel, ylabel=ylabel)
                
                axX = None
                axY = None
                
                axXY = plt.subplot2grid( (nrows*8+1,ncols*8+1), (i*8+1,j*8), colspan=8, rowspan=8 )
                if i == 0:
                    axX  = plt.subplot2grid( (nrows*8+1,ncols*8+1), (i*8,j*8), colspan=8 )
                if j == (ncols-1):
                    axY  = plt.subplot2grid( (nrows*8+1,ncols*8+1), (i*8+1,(j+1)*8), rowspan=8 )
                
                ax_2dhist([axXY,axX,axY],
                          parm_x, parm_y,
                          bins=bins_,
                          labelX=xlabel,
                          labelY=ylabel,
                          norm=False)
                
                if j==0:
                    axXY.xaxis.set_major_locator(locator)
                    axXY.xaxis.set_major_formatter(formatter)
                
                axXY.label_outer()
        
        fig.subplots_adjust(0.05, 0.05, 0.99, 1.0,
                            wspace=0.1, hspace=0.1)
        
        plt.savefig(figname_errs)
        plt.close()
        
        ####################################################
        
        if (k_x is None) or (k_y is None) or (k_z is None):
            return
        
        xs = [np.abs(k_x), np.abs(k_y), np.abs(k_z)]
        ys = [delta_u, delta_v, delta_w]
        
        xlabels = [r'$|k_x|$', r'$|k_y|$', r'$|k_z|$']
        ylabels = [r'$|\Delta u|$', r'$|\Delta v|$', r'$|\Delta w$|']
        
        nrows = len(ys)
        ncols = len(xs)
        
        fig = plt.figure(figsize=(5*ncols,5*nrows))
        
        for i in range(nrows):
            for j in range(ncols):
                
                parm_x = xs[j]
                parm_y = ys[i]
                
                xlabel = xlabels[j]
                ylabel = ylabels[i]
                
                bins_ = [bins_k, bins_du]
                if i == 2:
                    bins_ = [bins_k, bins_dw]
                    
                axXY = plt.subplot2grid( (nrows*8+1,ncols*8+1), (i*8+1,j*8), colspan=8, rowspan=8 )
                
                axX = None
                axY = None
                if i == 0:
                    axX  = plt.subplot2grid( (nrows*8+1,ncols*8+1), (i*8,j*8), colspan=8 )
                if j == (ncols-1):
                    axY  = plt.subplot2grid( (nrows*8+1,ncols*8+1), (i*8+1,(j+1)*8), rowspan=8 )
    
                
                ax_2dhist([axXY,axX,axY],
                          parm_x, parm_y,
                          bins=bins_,
                          labelX=xlabel,
                          labelY=ylabel,
                          norm=False)
                
                axXY.label_outer()
        
        fig.subplots_adjust(0.05, 0.05, 0.99, 1.0,
                            wspace=0.1, hspace=0.1)
        
        plt.savefig(figname_errs_k)
        plt.close()
        
        
        return
    
    def plot_statistics(self, 
                        u, v, w,
                        u_nn, v_nn, w_nn,
                        figname='./statistics.png',
                        bins=None,
                        **kwargs):
        
        nrows = 1
        if ~np.all(np.isnan(u)): nrows = 2
        
        if nrows == 1:
            umax = 150#np.nanmax( np.abs(u_nn) )
            vmax = 150#np.nanmax( np.abs(v_nn) )
            wmax = 10#np.nanmax( np.abs(w_nn) )
            
        else:
            umax = np.nanmax( np.abs(u) )
            vmax = np.nanmax( np.abs(v) )
            wmax = np.nanmax( np.abs(w) )
        
        if bins is None: bins = 40

        bins_u = np.linspace(-umax, umax, bins)
        bins_v = np.linspace(-vmax, vmax, bins)
        bins_w = np.linspace(-wmax, wmax, bins)
            
        fig = plt.figure(figsize=(9,3*nrows))
        
        ax = fig.add_subplot(nrows,3,1)
        plt.hist(u_nn.flatten(), bins_u, density=True, facecolor='r', alpha=0.75)
        ax.set_xlabel(r'$u_{nn}$')
        ax.set_ylabel('Density')
        ax.set_xlim(-umax,umax)
        ax.set_ylim(0,0.035)
        ax.grid(True)
        
        ax = fig.add_subplot(nrows,3,2)
        plt.hist(v_nn.flatten(), bins_v, density=True, facecolor='r', alpha=0.75)
        ax.set_xlabel(r'$v_{nn}$')
        ax.set_ylabel('Density')
        ax.set_xlim(-vmax,vmax)
        ax.set_ylim(0,0.035)
        ax.grid(True)
        
        ax = fig.add_subplot(nrows,3,3)
        plt.hist(w_nn.flatten(), bins_w, density=True, facecolor='r', alpha=0.75)
        ax.set_xlabel(r'$w_{nn}$')
        ax.set_ylabel('Density')
        ax.set_xlim(-wmax,wmax)
        ax.set_ylim(0,0.5)
        ax.grid(True)
        
        if nrows > 1:
            ax = fig.add_subplot(nrows,3,4)
            plt.hist(u.flatten(), bins_u, density=True, facecolor='g', alpha=0.75)
            ax.set_xlabel('u')
            ax.set_ylabel('Density')
            ax.set_xlim(-umax,umax)
            ax.set_ylim(0,0.035)
            ax.grid(True)
        
            ax = fig.add_subplot(nrows,3,5)
            plt.hist(v.flatten(), bins_v, density=True, facecolor='g', alpha=0.75)
            ax.set_xlabel('v')
            ax.set_ylabel('Density')
            ax.set_xlim(-vmax,vmax)
            ax.set_ylim(0,0.035)
            ax.grid(True)
        
            ax = fig.add_subplot(nrows,3,6)
            plt.hist(w.flatten(), bins_w, density=True, facecolor='g', alpha=0.75)
            ax.set_xlabel('w')
            ax.set_ylabel('Density')
            ax.set_xlim(-wmax,wmax)
            ax.set_ylim(0,0.5)
            ax.grid(True)
            
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()
    
    def plot_PDE_sampling(self, X, figname='./pd_sampling.png', alpha=0.1, **kwargs):
        
        # X_pde = tf.concat((t, x, y, z, nu, rho, rho_ratio, N, ), axis=1)
        
        # t, x, y, z, _, _, _, _ = tf.split(X, num_or_size_splits=8, axis=1)
        
        p = X[:,:4].numpy()
        
        t = p[:,0] + time_base
        x = p[:,2]*1e-3
        y = p[:,3]*1e-3
        z = p[:,1]*1e-3
        
        t_num = epoch2num(t)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.ConciseDateFormatter(locator)
    
        fig = plt.figure(figsize=(9,9))
        
        ax0 = fig.add_subplot(3,3,(1,2))
        ax1 = fig.add_subplot(3,3,(4,5), sharex=ax0)
        ax2 = fig.add_subplot(3,3,(7,8), sharex=ax0)
        
        ax3 = fig.add_subplot(3,3,3)
        ax4 = fig.add_subplot(3,3,6)
        ax5 = fig.add_subplot(3,3,9)
        
        ax0.scatter(t_num, x, alpha=alpha)
        ax1.scatter(t_num, y, alpha=alpha)
        ax2.scatter(t_num, z, alpha=alpha)
        
        ax0.set_ylabel('East-West (km)')
        ax1.set_ylabel('North-South (km)')
        ax2.set_ylabel('Altitude (km)')
        
        ax0.grid(True)
        ax1.grid(True)
        ax2.grid(True)
        
        ax0.xaxis.set_major_locator(locator)
        ax0.xaxis.set_major_formatter(formatter)
    
        ax3.scatter(x, z, alpha=alpha)
        ax4.scatter(y, z, alpha=alpha)
        ax5.scatter(x, y, alpha=alpha)
        
        ax3.set_xlabel('East-West (km)')
        ax4.set_xlabel('North-South (km)')
        ax5.set_xlabel('East-West (km)')
        
        ax3.set_ylabel('Altitude (km)')
        ax4.set_ylabel('Altitude (km)')
        ax5.set_ylabel('North-South (km)')
        
        ax3.grid(True)
        ax4.grid(True)
        ax5.grid(True)
        
        plt.tight_layout()
        plt.savefig(figname)
    
    def get_lb(self):
        
        lb = self.lbi + tf.constant([time_base, 0, 0, 0])
        
        return(lb.numpy())
    
    def get_ub(self):
        
        lb = self.ubi + tf.constant([time_base, 0, 0, 0])
        
        return(lb.numpy())
    
    lb = property(get_lb, None)
    ub = property(get_ub, None)
        
def eq_continuity(u_x, v_y, w_z, w=tf.constant(0., dtype=data_type), rho_ratio=tf.constant(0., dtype=data_type)):
    """
    Mass continuity equation in 3D for an incomprensible flow
    
    Inputs:
        u_i    :    partial derivative of u respect to i (m/s/m)
        
        rho_ratio    :    ratio between derivative of rho respect to z and rho. Units m^-1
        
                        = rho_z / rho
    """
    #Scale (1e-2): u_x = w_z = 10/10e3 m/s/ m
    y = 1e3*(rho_ratio*w + u_x + v_y + w_z)
    # y = u_x + v_y + w_z
    
    return(y)

# def eq_continuity_full(u_x, v_y, w_z, u, v, w, rho, rho_t, rho_x, rho_y, rho_z):
#     """
#     Mass continuity equation in 3D for an incomprensible flow
#
#     Inputs:
#         u_i    :    partial derivative of u respect to i (m/s/m)
#
#         rho_ratio    :    ratio between derivative of rho respect to z and rho. Units m^-1
#
#                         = rho_z / rho
#     """
#     #Scale (1e-2): u_x = w_z = 10*1e-3 m/s/km
#     y = rho_t + (u*rho_x + v*rho_y + w*rho_z) + ( rho+tf.constant(1.0, dtype=data_type) )*(u_x + v_y + w_z)
#     # y = u_x + v_y + w_z
#
#     return(y)
#
def eq_horizontal_momentum(u, v, w,
                           u_t, u_x, u_y, u_z,
                           u_xx, u_yy, u_zz,
                           p_x,
                           F    = tf.constant(0.0, dtype=data_type),
                           nu   = tf.constant(0.0, dtype=data_type),
                           rho  = tf.constant(1.0, dtype=data_type),
                           ):
    """
    Momentum equation in 3D for an incomprensible flow

    Inputs:
        u    :    velocity field in the x direction (m/s)
        v    :    velocity field in the y direction (m/s)
        w    :    velocity field in the z direction (m/s)

        u_i    :    partial derivative of u respect to i (m/s/s or m/s/km)
        u_ii    :   2nd partial derivative of u respect to i (m/s/km/km)

        nu     :    kinematic viscocity
        p_x    :    presure/rho_0 change in the x direction
        f_x    :    external force in the x direction
    """
    #Scale (1e0): u = 100 m/s, u_x = 10*1e-3 m/s/km, u_z = 100*1e-3 m/s/km
    y = u_t + u*u_x + v*u_y + w*u_z + p_x/rho - F - nu*(u_xx+u_yy+u_zz) 

    return(y)

def eq_vertical_momentum(u, v, w,
                         w_t, w_x, w_y, w_z,
                         w_xx, w_yy, w_zz,
                         p_z,
                         F      = tf.constant(0.0, dtype=data_type),
                         nu     = tf.constant(0.0, dtype=data_type),
                         rho    = tf.constant(1.0, dtype=data_type),
                         N      = tf.constant(0.0, dtype=data_type),
                         theta  = tf.constant(0.0, dtype=data_type),
                         ):
    """
    Momentum equation in 3D for an incomprensible flow

    Inputs:
        u    :    velocity field in the x direction (m/s)
        v    :    velocity field in the y direction (m/s)
        w    :    velocity field in the z direction (m/s)

        w_i    :    partial derivative of u respect to i (m/s/s or m/s/m)

        nu     :    kinematic viscocity
        p_z    :    presure/rho_0 change in the z direction
        f_z    :    external force in the z direction

        N    :    Brunt Vaisala frequency = SQRT{-(g/rho_0)*rho_z}
        theta    :    scaled density perturbation g*rho'/rho_0
    """
    #Scale (1e0): w = 10 m/s, w_z = 10*1e-3 m/s/km, theta = 1e0
    y = w_t + u*w_x + v*w_y + w*w_z + p_z/rho + N*theta - F - nu*(w_xx+w_yy+w_zz)

    return(y)

def eq_temperature(u, v, w,
                   theta_t, theta_x, theta_y, theta_z,
                   theta_xx, theta_yy, theta_zz,
                   N    = tf.constant(0.0, dtype=data_type),
                   k    = tf.constant(0.0, dtype=data_type),
                   ):
    """
    Momentum equation in 3D for an incomprensible flow
    
    Inputs:
        u    :    velocity field in the x direction (m/s)
        v    :    velocity field in the y direction (m/s)
        w    :    velocity field in the z direction (m/s)
        
        theta    :    scaled density perturbation g*rho'/(N*rho_0), linearly proportional to temperature
        theta_i   :    partial derivative of potential temperature respect to i (m/s/s or m/s/m)
        
        k     :    thermal diffusivity <> kinematic viscocity
        N    :    Brunt Vaisala frequency = SQRT{-(g/rho_0)*rho_z}
    """
    #Scale (1e1): w = 10 m/s, u = 100 m/s, theta_x = 1e-1
    y = theta_t + u*theta_x + v*theta_y + w*theta_z - N*w - k*(theta_xx+theta_yy+theta_zz)

    return(y)

def eq_vorticity_LF(omega_t,
                    omega_xx, omega_yy, omega_zz,
                    nxwxu_k_j, nxwxu_j_k,
                    theta_j = tf.constant(1.0, dtype=data_type),
                    nu  = tf.constant(0.0, dtype=data_type),
                    N   = tf.constant(0.0, dtype=data_type),
                   ):
    """
    Vorticity equation using the Laplacian formulation 
    
    Inputs:
        omega_i    :    partial derivative of the vorticity (omega) respect to i
        nxwxu_k_j  :    derivative if the k component of (Nabla x omega x u) respect to j
        nxwxu_j_k  :    derivative if the j component of (Nabla x omega x u) respect to k
        
        theta_j    :    partial derivative of Temperature (theta) respect to j
    """
    #Scale (1e0): u = 100 m/s, u_x = 10*1e-3 m/s/km, u_z = 100*1e-3 m/s/km
    y = 1e5*(omega_t + nxwxu_k_j - nxwxu_j_k - nu*(omega_xx + omega_yy + omega_zz) + N*theta_j)
    
    return(y)

def eq_vorticity_RF(omega_t,
                    nxnxw_k_j, nxnxw_j_k,
                    nxwxu_k_j, nxwxu_j_k,
                    theta_j = tf.constant(1.0, dtype=data_type),
                    nu  = tf.constant(0.0, dtype=data_type),
                    N   = tf.constant(0.0, dtype=data_type),
                   ):
    """
    Vorticity equation using the Rotational formulation. Assuming Div omega = 0
    
    Inputs:
        omega_i    :    partial derivative of the vorticity (omega) respect to i
        k_j        :    derivative if the k component of (Nabla x omega x u) respect to j
        j_k        :    derivative if the j component of (Nabla x omega x u) respect to k
        
        r_j        :    partial derivative of the r component of (Nabla x Nabla x omega) respect to j
        theta_j    :    partial derivative of Temperature (theta) respect to j
        
    """
    #Scale (1e0): u = 100 m/s, u_x = 10*1e-3 m/s/km, u_z = 100*1e-3 m/s/km
    y = 1e5*(omega_t + nxwxu_k_j - nxwxu_j_k + nu*(nxnxw_k_j - nxnxw_j_k) + N*theta_j)
    
    return(y)

def eq_vorticity_RF_F(omega_t, F_k_j, F_j_k,
                    theta_j = tf.constant(0.0, dtype=data_type),
                    N   = tf.constant(0.0, dtype=data_type),
                   ):
    """
    Vorticity equation using the Rotational formulation. Assuming Div omega = 0
    
    Inputs:
        omega_i    :    partial derivative of the vorticity (omega) respect to i
        k_j        :    derivative if the k component of (Nabla x omega x u) respect to j
        j_k        :    derivative if the j component of (Nabla x omega x u) respect to k
        
        r_j        :    partial derivative of the r component of (Nabla x Nabla x omega) respect to j
        theta_j    :    partial derivative of Temperature (theta) respect to j
        
    """
    #Scale (1e0): u = 100 m/s, u_x = 10*1e-3 m/s/km, u_z = 100*1e-3 m/s/km
    y = 1e5*(omega_t + F_k_j - F_j_k + N*theta_j)
    
    return(y)

def eq_poisson(ui_xx, ui_yy, ui_zz,
               omegaz_y, omegay_z,
               ):
    """
    Momentum equation in 3D for an incomprensible flow
    
    Inputs:
        u    :    velocity field in the x direction (m/s)
        v    :    velocity field in the y direction (m/s)
        w    :    velocity field in the z direction (m/s)
        
        u_i    :    partial derivative of u respect to i (m/s/s or m/s/km)
        u_ii    :   2nd partial derivative of u respect to i (m/s/km/km)
        
        nu     :    kinematic viscocity
        p_x    :    presure/rho_0 change in the x direction
        f_x    :    external force in the x direction
    """
    #Scale (1e0): u = 100 m/s, u_x = 10*1e-3 m/s/km, u_z = 100*1e-3 m/s/km
    y = 1e6*(ui_xx + ui_yy + ui_zz + omegaz_y - omegay_z)
    
    return(y)

def get_log_file(filename, index=None):
    
    path, file_base = os.path.split(filename)
    
    rpath = os.path.join(path, 'log')
    
    if not os.path.exists(rpath):
        os.mkdir(rpath)
    
    if index is None:
        prefix = '.'
    elif isinstance(index, int):
        prefix = '.%05d.' %(index)
    else:
        prefix = '.%s.' %(index)
    
    filename_log = os.path.join(rpath, '%s%s%s.h5' %(file_base, prefix, 'weights') )
    
    return(filename_log)

def restore(filename, log_index=None, include_res_layer=None, activation=None, skip_mismatch=False,
            NS_type=None):
            
    keys_float  = ['lr',
                   # 'ini_w_data', 'ini_w_Ph1', 'ini_w_Ph2', 'ini_w_Ph3', 'ini_w_Ph4',
                   # 'f_scl_out',
                   'lb', 'ub', 'mn',
                   'lon_ref', 'lat_ref', 'alt_ref',
                   ]
    
    keys_int    = ['shape_in', 'shape_out',
                   'width', 'depth',
                   'nnodes',
                   'nblocks',
                   'ns_pde', 'r_seed',
                   'laaf',
                   'dropout',
                   'residual_layer',
                   # 'nblocks',
                   ]
    
    keys_str    = ['act', 'opt', 'f_scl_in', 'nn_type', 'NS_type']
        
    kwargs = {'nblocks':3}
    
    print("Opening NN file: %s ..." %filename)
    
    with h5py.File(filename,'r') as fp:
        
        for key in keys_float:
            kwargs[key] = fp[key][()]
            
        for key in keys_int:
            
            # if key == 'dropout':
            if key not in fp.keys():
                # kwargs[key] = None
                continue
            
            kwargs[key] = int(fp[key][()])
            
        for key in keys_str:
            try:
                kwargs[key] = fp[key][()].decode('utf-8')
            except:
                continue
    
    if include_res_layer is not None: kwargs['residual_layer'] = include_res_layer
    if activation is not None: kwargs['act'] = activation
    if NS_type is not None: kwargs['NS_type'] = NS_type
    
    nn = App(**kwargs)
    
    file_weights = get_log_file(filename, log_index)
    print("Opening NN file: %s ..." %file_weights)
    nn.model.load_weights(file_weights)#, skip_mismatch=skip_mismatch)
    
    # if include_res_layer:
    #     mean_model = nn.model.layers[1]
    #     mean_model.trainable = False
    #     nn.model.summary()
    
    nn.lbi = tf.convert_to_tensor(kwargs['lb'])
    nn.ubi = tf.convert_to_tensor(kwargs['ub'])
    nn.mn  = tf.convert_to_tensor(kwargs['mn'])
    
    print("         lower bounds    :", nn.lbi.numpy())
    print("         upper bounds    :", nn.ubi.numpy())
    
    with h5py.File(filename,'r') as fp:
        
        nn.loss_log = fp['loss_log'][()]
        nn.loss_data_log = fp['loss_data_log'][()]
        
        nn.loss_div_log = fp['loss_Ph1_log'][()]
        nn.loss_mom_log = fp['loss_Ph2_log'][()]
        nn.loss_temp_log = fp['loss_Ph3_log'][()]
        nn.loss_srt_log = fp['loss_Ph4_log'][()]
        
        nn.w_div_log = fp['w_Ph1_log'][()]
        nn.w_mom_log = fp['w_Ph2_log'][()]
        nn.w_temp_log = fp['w_Ph3_log'][()]
        nn.w_srt_log = fp['w_Ph4_log'][()]
        
        try:
            nn.rmse_u_log = fp['rmse_u_log'][()]
            nn.rmse_v_log = fp['rmse_v_log'][()]
            nn.rmse_w_log = fp['rmse_w_log'][()]
            
            nn.tv_u_log = fp['tv_u_log'][()]
            nn.tv_v_log = fp['tv_v_log'][()]
            nn.tv_w_log = fp['tv_w_log'][()]
            
            nn.tv_ue_log = fp['tv_ue_log'][()]
            nn.tv_ve_log = fp['tv_ve_log'][()]
            nn.tv_we_log = fp['tv_we_log'][()]
            
        except:
            nan_array = np.array(nn.w_div_log) + np.nan
            nn.rmse_u_log = nan_array
            nn.rmse_v_log = nan_array
            nn.rmse_w_log = nan_array
        
            nn.tv_u_log = nan_array
            nn.tv_v_log = nan_array
            nn.tv_w_log = nan_array
            
            nn.tv_ue_log = nan_array
            nn.tv_ve_log = nan_array
            nn.tv_we_log = nan_array
            
        nn.ep_log = fp['ep_log'][()]
        
    
    return(nn)