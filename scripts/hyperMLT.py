import os
import time

import numpy as np

from datetime import datetime
from radar.smr.smr_file import SMRReader

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
    noise_std   = 1.0#np.std(dop_errs)
    
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
                        laaf=laaf,
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

if __name__ == '__main__':
    
    #delay in mins
    delay = 0#230/60. + 300
    only_SMR    = True
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    #VORTEX cmapaign
    parser.add_argument('-e', '--exp', dest='exp', default='nm', help='Experiment configuration')
    
    #-d /Users/radar/Data/IAP/SIMONe/Norway/VorTex --lon-center=16.4 --lat-center=69.3 --alt-center=89 --lon-range=7.5 --lat-range=2.5 --alt-range=14
    
    parser.add_argument('-d', '--dpath', dest='dpath', default=None, help='Data path')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Data path')
    
    parser.add_argument('-n', '--neurons-per_layer',  dest='neurons_per_layer', default=128, help='# kernel', type=int)
    parser.add_argument('-l', '--hidden-layers',      dest='hidden_layers', default=5, help='# kernel layers', type=int)
    parser.add_argument('-c', '--nodes',              dest='n_nodes', default=0, help='# nodes', type=int)
    parser.add_argument('--nblocks',                  dest='n_blocks', default=0, help='', type=int)
    
    parser.add_argument('--npde',                     dest='N_pde', default=5000, help='', type=int)
    parser.add_argument('--ns',                       dest='nepochs', default=5000, help='', type=int)
    
    parser.add_argument('--learning-rate',      dest='learning_rate', default=1e-3, help='', type=float)
    parser.add_argument('--pde-weight-upd-rate', dest='w_pde_update_rate', default=1e-5, help='', type=float)
    
    parser.add_argument('--pde-weight',         dest='w_pde', default=1e0, help='PDE weight', type=float)
    parser.add_argument('--data-weight',        dest='w_data', default=1e0, help='data fidelity weight', type=float)
    parser.add_argument('--srt-weight',        dest='w_srt', default=1e0, help='Slope recovery time loss weight', type=float)
    
    parser.add_argument('--pde',        dest='NS_type', default="VV", help='Navier-Stokes formulation, either VP (velocity-pressure) or VV (velocity-vorticity)')
    parser.add_argument('--noutputs',   dest='noutputs', default=3, help='', type=int)
    
    parser.add_argument('--noise', dest='noise_sigma', default=0.0, help='', type=float)
    
    parser.add_argument('--architecture', dest='nn_type', default='respinn', help='')
    parser.add_argument('--version',     dest='nn_version', default=3.52, type=float)
    parser.add_argument('--activation',  dest='nn_activation', default='sine')
    parser.add_argument('--laaf',        dest='nn_laaf', default=1, type=int)
    parser.add_argument('--dropout',     dest='nn_dropout', default=0, type=int)
    
    parser.add_argument('-s', '--nn-init-std',    dest='nn_init_sigma', default=None, type=float)
    parser.add_argument('-i', '--nn-w-init',    dest='nn_w_init', default='GlorotNormal', type=str)
    
    parser.add_argument('--time-window', dest='dtime', default=24, help='hours', type=int)
    parser.add_argument('--initime',    dest='tini', default=0, help='hours', type=float)
    
    parser.add_argument('--lon-range', dest='dlon', default=None, help='degrees', type=float)
    parser.add_argument('--lat-range', dest='dlat', default=None, help='degrees', type=float)
    parser.add_argument('--alt-range', dest='dh', default=None, help='km', type=float)
    parser.add_argument('--lon-center', dest='lon_center', default=None, help='degrees', type=float)
    parser.add_argument('--lat-center', dest='lat_center', default=None, help='degrees', type=float)
    parser.add_argument('--alt-center', dest='alt_center', default=None, help='km', type=float)
    
    ###SIMONe 2018
    # parser.add_argument('--lon-range', dest='dlon', default=6, help='degrees', type=float)
    # parser.add_argument('--lat-range', dest='dlat', default=3, help='degrees', type=float)
    # parser.add_argument('--alt-range', dest='dh', default=24, help='km', type=float)
    # parser.add_argument('--lon-center', dest='lon_center', default=12.5, help='degrees', type=float)
    # parser.add_argument('--lat-center', dest='lat_center', default=54, help='degrees', type=float)
    # parser.add_argument('--alt-center', dest='alt_center', default=91, help='km', type=float)
    
    ##VORTEX event
    # parser.add_argument('--time-window', dest='dtime', default=24, help='hours', type=float)
    # parser.add_argument('--initime',    dest='tini', default=0, help='hours', type=float)
    
    # parser.add_argument('--lon-range', dest='dlon', default=8, help='degrees', type=float)
    # parser.add_argument('--lat-range', dest='dlat', default=2.5, help='degrees', type=float)
    # parser.add_argument('--alt-range', dest='dh', default=None, help='km', type=float)
    # parser.add_argument('--lon-center', dest='lon_center', default=16, help='degrees', type=float)
    # parser.add_argument('--lat-center', dest='lat_center', default=69.25, help='degrees', type=float)
    # parser.add_argument('--alt-center', dest='alt_center', default=90, help='km', type=float)
    
    ###Extreme event
    # parser.add_argument('--initime',    dest='tini', default=3, help='hours', type=float)
    # parser.add_argument('--time-window', dest='dtime', default=4, help='hours', type=float)
    # parser.add_argument('--lon-range', dest='dlon', default=6, help='degrees', type=float)
    # parser.add_argument('--lat-range', dest='dlat', default=2, help='degrees', type=float)
    # parser.add_argument('--alt-range', dest='dh', default=None, help='km', type=float)
    # parser.add_argument('--lon-center', dest='lon_center', default=16.25, help='degrees', type=float)
    # parser.add_argument('--lat-center', dest='lat_center', default=69.25, help='degrees', type=float)
    # parser.add_argument('--alt-center', dest='alt_center', default=90, help='km', type=float)
    
    ###Extreme event 2
    # parser.add_argument('--initime',    dest='tini', default=4, help='hours', type=float)
    # parser.add_argument('--time-window', dest='dtime', default=3, help='hours', type=float)
    # parser.add_argument('--lon-range', dest='dlon', default=4, help='degrees', type=float)
    # parser.add_argument('--lat-range', dest='dlat', default=1, help='degrees', type=float)
    # parser.add_argument('--alt-range', dest='dh', default=16, help='km', type=float)
    # parser.add_argument('--lon-center', dest='lon_center', default=16.5, help='degrees', type=float)
    # parser.add_argument('--lat-center', dest='lat_center', default=70, help='degrees', type=float)
    # parser.add_argument('--alt-center', dest='alt_center', default=91, help='km', type=float)
    
    parser.add_argument('--output-file', dest='filename_model', default=None, help='')
    parser.add_argument('--output-file-short-naming', dest='short_naming', default=0, type=int)
    parser.add_argument('--realtime', dest='realtime', default=0, help='', type=int)
    
    parser.add_argument('--transfer-learning', dest='transfer_learning', default=0, help='', type=int)
    parser.add_argument('--basefile', dest='filename_model_tl', default=None, help='')
    
    args            = parser.parse_args()
    
    tini            = args.tini
    dt              = args.dtime
    dlon            = args.dlon
    dlat            = args.dlat
    dh              = args.dh
    
    lon_center      = args.lon_center
    lat_center      = args.lat_center
    alt_center      = args.alt_center
    
    num_outputs     = args.noutputs
    
    path            = args.dpath
    rpath           = args.rpath
    
    w_pde           = args.w_pde
    w_data          = args.w_data
    w_srt           = args.w_srt
    
    w_pde_update_rate = args.w_pde_update_rate
    
    N_pde           = args.N_pde
    NS_type         = args.NS_type
    
    learning_rate   = args.learning_rate
    noise_sigma     = args.noise_sigma
    nepochs         = args.nepochs
    
    n_nodes   = args.n_nodes
    n_blocks   = args.n_blocks
    
    hidden_layers   = args.hidden_layers
    neurons_per_layer   = args.neurons_per_layer
    
    nn_type         = str.upper(args.nn_type)
    nn_version      = args.nn_version
    nn_activation   = args.nn_activation
    nn_laaf         = args.nn_laaf
    nn_dropout      = args.nn_dropout
    
    nn_init_sigma   = args.nn_init_sigma
    nn_w_init       = args.nn_w_init
    
    filename_model  = args.filename_model
    realtime        = args.realtime
    
    filename_model_tl = args.filename_model_tl
    transfer_learning = args.transfer_learning
    
    short_naming    = args.short_naming
    
    exp             = args.exp
    
    batch_size      = None
    
    if not only_SMR:
        nn_version += 10
    
    if exp is not None:
        if exp.upper()  == 'SIMONE2018':
            
            tini            = 0
            dt              = 24
            # dlon            = 6
            # dlat            = 3
            # dh              = 24
            
            lon_center      = 12.5
            lat_center      = 54
            alt_center      = 91
            path            = "/Users/radar/Data/IAP/SIMONe/Germany/Simone2018"
        
        elif exp.upper()  == 'EXTREMEW':
            
            tini            = 0
            dt              = 24
            # dlon            = 6
            # dlat            = 3
            # dh              = 24
            
            # lon_center      = 12.5
            # lat_center      = 54
            # alt_center      = 91
            path            = "/Users/radar/Data/IAP/SIMONe/Germany/ExtremeW"
        
        
        elif exp.upper()  == 'VORTEX':
            
            tini            = 0
            dt              = 24
            # dlon            = 8
            # dlat            = 2.5
            
            # lon_center      = 16.25
            # lat_center      = 69.25
            # alt_center      = 90
            path            = "/Users/radar/Data/IAP/SIMONe/Norway/VorTex"
        
        elif exp.upper()  == 'WAVECONVECTION':
            
            tini            = 0
            dt              = 24
            dlon            = 8
            dlat            = 2.5
            
            lon_center      = 16.25
            lat_center      = 69.25
            # alt_center      = 90
            path            = "/Users/radar/Data/IAP/SIMONe/Norway/WaveConvection"
        
        elif exp.upper()  == 'EXT24':
            
            tini            = 3
            dt              = 3
            # dlon            = 7
            # dlat            = 2.4
            # dh              = 16
            
            # lon_center      = 16.25
            # lat_center      = 69.25
            # alt_center      = 89
            path            = "/Users/radar/Data/IAP/SIMONe/Norway/ExtremeEvent"
            
        
        elif exp.upper()  == 'EXT1':
            
            tini            = 3
            dt              = 3
            dlon            = 7
            dlat            = 2.4
            # dh              = 16
            
            lon_center      = 16.25
            lat_center      = 69.25
            alt_center      = 89
            path            = "/Users/radar/Data/IAP/SIMONe/Norway/ExtremeEvent"
            
        elif exp.upper()  == 'EXT2':
            
            tini            = 3
            dt              = 4
            # dlon            = 4
            # dlat            = 1
            # dh              = 16
            
            # lon_center      = 16.5
            # lat_center      = 70
            # alt_center      = 89
            path            = "/Users/radar/Data/IAP/SIMONe/Norway/ExtremeEvent"
        
        elif exp.upper()  == 'NM':
            
            tini            = 0
            dt              = 24
            
            # alt_center      = 90
            path            = "/Users/radar/Data/IAP/SIMONe/NewMexico/Eclipse"
            
        elif exp.upper()  == 'NM2':
            
            tini            = 0
            dt              = 24
            
            # alt_center      = 90
            path            = "/Users/radar/Data/IAP/SIMONe/NewMexico/EclipseApr"
            
               
        elif exp.upper()  == 'TONGA1':
            
            tini            = 0
            dt              = 24
            path            = "/Users/radar/Data/IAP/SIMONe/Condor/Tonga"
        
        elif exp.upper()  == 'TONGA2':
            
            tini            = 0
            dt              = 24
            path            = "/Users/radar/Data/IAP/SIMONe/JRO/Tonga"
        
        elif exp.upper()  == 'TONGA3':
            
            tini            = 0
            dt              = 24
            path            = "/Users/radar/Data/IAP/SIMONe/Piura/Tonga"
            
        elif exp.upper()  == 'DNS':
            
            path            = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/DNSx10_+12+53+91"
            noise_sigma     = -1
            tini            = 0
            dt              = 4
            
        elif exp.upper()  == 'ICON2015':
            
            tini            = 0
            dt              = 4
            path            = "/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815/ICON_+00+70+90"
            noise_sigma     = -1.0
            
        elif exp.upper()  == 'ICON2016':
            
            path            = "/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160816/ICON_-08+73+90"
            # noise_sigma     = 6.0
        
        else:
            raise ValueError('Experiment option not implemented ...')
    
    if rpath is None:
        rpath = os.path.dirname( os.path.realpath(path) )
        rpath = os.path.join(rpath, 'winds')
        
        if not os.path.exists(rpath): os.mkdir(rpath)
        
    
    rpath = os.path.join(rpath, 'nn%s_%3.2f' %(nn_type, nn_version) )
    
    if not os.path.exists(rpath): os.mkdir(rpath)
    
    #Read meteor data in LLA coordinates
    meteor_obj = SMRReader(path, realtime=realtime)
    meteor_obj.set_spatial_center(lon_center=lon_center,
                                  lat_center=lat_center,
                                  alt_center=alt_center)
    
    print('Waiting %d min ...' %delay)
    time.sleep(delay*60)
    
    # info = meteor_obj.read_next_file()
    
    # for i in range(8):
    
    while True:
    
        info = meteor_obj.read_next_file(enu_coordinates=True)
        
        if info != 1: break
        
        train_hyper(meteor_obj.df,
                             tini=tini,
                            dt=dt,
                            dlon=dlon,
                            dlat=dlat,
                            dh=dh,
                            rpath=rpath,
                            num_outputs=num_outputs,
                            w_pde=w_pde,
                            w_data=w_data,
                            w_srt=w_srt,
                            laaf=nn_laaf,
                            learning_rate=learning_rate,
                            noise_sigma=noise_sigma,
                            nepochs=nepochs, 
                            N_pde=N_pde,
                            num_neurons_per_layer=neurons_per_layer,
                            num_hidden_layers=hidden_layers,
                            n_nodes=n_nodes,
                            n_blocks=n_blocks,
                            filename_model=filename_model,
                            transfer_learning=transfer_learning,
                            filename_model_tl=filename_model_tl,
                            short_naming=short_naming,
                            activation=nn_activation,
                            init_sigma=nn_init_sigma,
                            NS_type=NS_type,
                            w_init=nn_w_init,
                            lon_center=lon_center,
                            lat_center=lat_center,
                            alt_center=alt_center,
                            batch_size=batch_size,
                            dropout=nn_dropout,
                            only_SMR=only_SMR,
                            w_pde_update_rate=w_pde_update_rate,
                            nn_type=nn_type,
                            )
        
        # break
        # nn_init_sigma *= 2
        