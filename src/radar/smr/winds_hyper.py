'''
Created on 4 Sep 2022

@author: mcordero
'''
# A module getting windfield cuts using Brian's wind field  inversion on MMARIA systems

import os
from datetime import datetime

# Import TensorFlow and NumPy
import numpy as np

from pinn import hyper as pinn

from utils.PhysicalParameters import MSIS
from radar.smr.smr_file import filter_data, plot_spatial_sampling, plot_Doppler_sampling

def train_hyper(df,
                 tini=0,
                 dt=24, #hours
                 dlon=None,
                 dlat=None,
                 dh=None,
                 rpath='./',
                 num_outputs=3, #3 = only divergence, 4 = div + momentum, 5 = div + momentum + ext. forcing
                 num_hidden_layers=6,    #Total hidden layers = 2*n + 1 
                 num_neurons_per_layer=64,
                 n_nodes=16,
                 n_blocks=2,
                 activation='tanh',
                 nepochs=10000,
                 w_data=1.0,
                 w_pde=1e3,
                 w_srt=1e-1,
                 N_pde = 10000, #px+py+pz
                 learning_rate=1e-4,
                 noise_sigma = 0,
                 filename_model=None,
                 laaf=0,
                 transfer_learning=0,
                 filename_model_tl=None,
                 short_naming=True,
                 init_sigma=1.0,
                 NS_type='VP',
                 w_init='Lecun',
                 lon_center=None,
                  lat_center=None,
                  alt_center=None,
                  batch_size=None,
                  only_SMR=True,
                  dropout=True,
                  w_pde_update_rate=1e-4,
                  nn_type = 'deeponet',
                 ):
    
    # if dt < 0: tini = -dt
    # else: tini = 0
    # dt = np.abs(dt)
    
    # config_gpu(gpu_flg = 1)
    
    seed = 191
    # Set data type
    np.random.seed(seed)
    
    ini_date = datetime.utcfromtimestamp(df['times'].min())    
    plot_spatial_sampling(df, path=rpath, suffix='prefilter_%s' %ini_date.strftime('%Y%m%d'),
                        # vmin=0, vmax=500,
                        )
    
    df_filtered = filter_data(df,
                     tini=tini, dt=dt,
                     dlon=dlon, dlat=dlat, dh=dh,
                     lon_center=lon_center,
                      lat_center=lat_center,
                      alt_center=alt_center,
                      only_SMR=only_SMR)
    
    # bgn = -10
    # df.loc[:,'u'] = df['u'].values + bgn
    # df.loc[:,'v'] = df['v'].values + bgn
    # df.loc[:,'w'] = df['w'].values + bgn
    # df.loc[:,'dops'] = df['dops'].values - bgn/(2*np.pi)*(df['braggs_x'].values + df['braggs_y'].values + df['braggs_z'].values)
    
    df_training = df_filtered
    df_test     = df_filtered#.sample(frac=0.01, random_state=0) #
    
    # df_training = df_filtered.sample(frac=0.95, random_state=0)
    # df_test     = df_filtered.drop(df_training.index)
    
    df_training.sort_index(inplace=True)
    
    ini_date = datetime.utcfromtimestamp(df_training['times'].min())
    plot_spatial_sampling(df_training, path=rpath,
                        suffix='postfilter_%s' %ini_date.strftime('%Y%m%d'),
                        # vmin=0, vmax=100,
                        )
    
    plot_Doppler_sampling(df, df_filtered, rpath,
                          suffix='postfilter_%s' %ini_date.strftime('%Y%m%d')
                          )
    

    # plot_2point_sampling(df_training, path=rpath,
    #                     suffix='delta_sampling_%s' %ini_date.strftime('%Y%m%d-%H%M%S'),)
    
    # plot_delta_sampling(df, rpath)

    # return 

    msis = None
    # try:
    #     msis = MSIS(ini_date,
    #                  glat=df['lats'].median(),
    #                  glon=df['lons'].median(),
    #                  time_range=dt,
    #                  plot_values=True)
    # except:
    #     msis = None
    
    if short_naming:
        suffix = '%s' %ini_date.strftime('%Y%m%d-%H%M%S')
    else:
        suffix = "%s_w%02dn%03.2f[%s]a%sl%02dn%03dd%03db%02dw%2.1elr%2.1em%2.1e_laaf%2.1e_%s_ur=%2.1e" %(
                                                            ini_date.strftime('%Y%m%d'),
                                                            dt,
                                                            noise_sigma,
                                                            NS_type,
                                                            activation[:3],
                                                            num_hidden_layers,
                                                            num_neurons_per_layer,
                                                            n_nodes,
                                                            n_blocks,
                                                            w_pde,
                                                            learning_rate,
                                                            N_pde,
                                                            w_srt,
                                                            # dropout,
                                                            w_init[:2],
                                                            # init_sigma,
                                                            w_pde_update_rate,
                                                            )
    
    ###########################
    #Traiing dataset

    t = df_training['times'].values                          #[N]
    
    x = df_training['lons'].values                             #[N]
    y = df_training['lats'].values                             #[N]
    z = df_training['heights'].values                             #[N]
    
    kx = df_training['braggs_x'].values                     #[N]
    ky = df_training['braggs_y'].values
    kz = df_training['braggs_z'].values
    
    k = np.stack([kx, ky, kz], axis=1)                      #[N,3]
    
    u = df_training['u'].values                              #[N]
    v = df_training['v'].values                              #[N]
    w = df_training['w'].values                              #[N]
    
    # T = df_training['T'].values                            #Kelvin
    # rho = df_training['rho'].values                            #Kg/m3
    # P = df_training['P'].values                              #Pascal <> N/m2
    # rho_msis     = msis.get_rho(z*1e3)
    # N   = msis.get_N(z*1e3)
    
    dops        = 2*np.pi*df_training['dops'].values
    dop_errs    = 2*np.pi*df_training['dop_errs'].values
    
    dop_std     = np.std(dops)
    noise_std   = 3#np.std(dop_errs)
    
    d = dops
    
    if noise_sigma < 0:
        noise_std  = 1.0
    elif noise_sigma > 0:
        print("Adding synthetic noise ...")
        noise = noise_sigma*dop_std*np.random.normal(0, 1, size=dops.shape[0])
        d   = dops + noise    #[N]
        noise_std   = np.std(noise)
    
    d_err = np.ones_like(dops)*noise_std
    
    print('*'*40)
    print("Doppler std=", dop_std )
    print("Noise std=", noise_std)
    print('*'*40)
    
    ###########################
    # df_test = df_training
    
    #Test dataset
    t_test = df_test['times'].values                          #[N]
    
    x_test = df_test['lons'].values                             #[N]
    y_test = df_test['lats'].values                             #[N]
    z_test = df_test['heights'].values                             #[N]
    
    u_test = df_test['u'].values                              #[N]
    v_test = df_test['v'].values                              #[N]
    w_test = df_test['w'].values                              #[N]
    
    # P_test   = df_test['P'].values
    # rho_test = df_test['rho'].values
    
    kx_test = df_test['braggs_x'].values                     #[N]
    ky_test = df_test['braggs_y'].values
    kz_test = df_test['braggs_z'].values
    
    d_test = 2*np.pi*df_test['dops'].values
    
    ###########################
    if filename_model is None:
        filename_model = 'model_%s.h5' %suffix #ini_date.strftime('%Y%m%d')
        
    # filename_mean = os.path.join(rpath, filename_model)
    
    suffix_mean = suffix #'%s_w%02d_n%03.1f' %(ini_date.strftime('%Y%m%d-%H%M%S'), dt, noise_sigma) # #
    filename_mean = os.path.join(rpath, 'mean_model_%s.h5' %suffix_mean)
    
    if os.path.exists(filename_mean):
    
        # print('Loading %s' %filename_mean)
        nn = pinn.restore(filename_mean)
    
    else:
        # Initialize Neural Network model
        nn = pinn.App(
                        shape_in  = 4,
                        shape_out = num_outputs,
                        width   = num_neurons_per_layer,
                        depth   = num_hidden_layers,
                        nnodes  = n_nodes,
                        nblocks = n_blocks,
                        act     = activation,
                        w_init  = w_init,
                        msis    = msis,
                        lon_ref  = (x.min() + x.max())/2,
                        lat_ref  = (y.min() + y.max())/2,
                        alt_ref  = (z.min() + z.max())/2,
                        NS_type  = NS_type,
                        nn_type = nn_type,
                    )
        
        # with tf.device("/device:GPU:0"):
        nn.train(t, x, y, z, d, k,
                 t_test, x_test, y_test, z_test, u_test, v_test, w_test,
                 # t, x, y, z, u, v, w,
                 d_err      = d_err,
                 epochs     = nepochs,
                 filename   = filename_mean,
                 w_pde_update_rate = w_pde_update_rate,
                 lr      = learning_rate,
                w_data  = w_data,
                w_div   = w_pde,
                w_mom   = w_pde,
                w_temp  = w_pde,
                w_srt   = w_srt,
                ns_pde  = N_pde,
                # NS_type  = NS_type,
                dropout  = dropout,
                laaf=laaf,
                 )
    
        nn.save(filename_mean)
        # nn.restore(filename_mean)
    
    figname01 = os.path.join(rpath, 'loss_hist_%s.png' %suffix_mean)
    
    nn.plot_loss_history(figname=figname01)
    
    ############################################################
    #Traning points
    figname02 = os.path.join(rpath, 'training_winds_%s.png' %suffix_mean)
    figname03 = os.path.join(rpath, 'training_errors_%s.png' %suffix_mean)
    figname04 = os.path.join(rpath, 'training_errors_k_%s.png' %suffix_mean)
    figname05 = os.path.join(rpath, 'training_statistics_%s.png' %suffix_mean)
    figname_P = os.path.join(rpath, 'training_pressure_%s.png' %suffix_mean)
    
    outputs = nn.infer(t, x, y, z)
    
    u_nn = outputs[:,0]
    v_nn = outputs[:,1]
    w_nn = outputs[:,2]
    
    d_nn = -(u_nn*kx + v_nn*ky + w_nn*kz)
    
    nn.plot_statistics(
                    u, v, w,
                    u_nn, v_nn, w_nn,
                    figname=figname05)
    
    nn.plot_solution(t, x, y, z,
                     u, v, w, d,
                     u_nn, v_nn, w_nn, d_nn,
                      k_x=kx, k_y=ky, k_z=kz,
                     figname_winds=figname02,
                     figname_errs=figname03,
                     figname_errs_k=figname04,
                     figname_pressure=figname_P)
    
    ############################################################
    #Validation points
    
    figname02 = os.path.join(rpath, 'testing_winds_%s.png' %suffix)
    figname03 = os.path.join(rpath, 'testing_errors_%s.png' %suffix)
    figname04 = os.path.join(rpath, 'testing_errors_k_%s.png' %suffix)
    figname05 = os.path.join(rpath, 'testing_statistics_%s.png' %suffix)
    figname_P = os.path.join(rpath, 'testing_pressure_%s.png' %suffix)
    
    outputs = nn.infer(t_test, x_test, y_test, z_test)
    
    u_nn = outputs[:,0]
    v_nn = outputs[:,1]
    w_nn = outputs[:,2]
    
    d_nn = -(u_nn*kx_test + v_nn*ky_test + w_nn*kz_test)
    
    nn.plot_statistics(
                    u_test, v_test, w_test,
                    u_nn, v_nn, w_nn,
                    figname=figname05)
    
    nn.plot_solution(t_test, x_test, y_test, z_test,
                     u_test, v_test, w_test, d_test,
                     u_nn, v_nn, w_nn, d_nn,
                    k_x=kx_test, k_y=ky_test, k_z=kz_test,
                     figname_winds=figname02,
                     figname_errs=figname03,
                     figname_errs_k=figname04,
                     figname_pressure=figname_P)
    
    return( filename_mean )