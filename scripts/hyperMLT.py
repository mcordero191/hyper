import os, glob
import time, datetime

import numpy as np

import sys
sys.path.insert(0,'../src')

from pinn import hyper as pinn

from radar.smr.smr_file import SMRReader
from utils.io import read_vortex_files
from utils.PhysicalParameters import MSIS

def get_filename_suffix(short_naming,
                        ini_date, dt, noise_sigma, NS_type,
                        activation, num_hidden_layers,
                        num_neurons_per_layer, n_nodes,
                        n_blocks, w_pde, w_srt, learning_rate,
                        N_pde, laaf, w_init, w_pde_update_rate,
                        dropout=0,
                        sampling_method="",
                        init_sigma = None,
                        ensemble = 0,
                        ):
    
    if short_naming:
        suffix = '%s_%03d' %(ini_date.strftime('%Y%m%d_%H%M'), ensemble)
    else:
        suffix = "%s_w%02dn%3.2f%sl%02d%03dw%2.1elr%2.1eur%2.1e%3.2f_%03d" %(
                                                            ini_date.strftime('%Y%m%d_%H%M'),
                                                            dt,
                                                            noise_sigma,
                                                            NS_type,
                                                            # activation[:3],
                                                            num_hidden_layers,
                                                            num_neurons_per_layer,
                                                            # n_nodes,
                                                            # n_blocks,
                                                            w_pde,
                                                            # w_srt,
                                                            learning_rate,
                                                            # N_pde,
                                                            # laaf,
                                                            # dropout,
                                                            # w_init[:2],
                                                            # init_sigma,
                                                            w_pde_update_rate,
                                                            # sampling_method[:3],
                                                            init_sigma,
                                                            ensemble,
                                                            )
        
    return suffix
    
def train_hyper(df,
                df_testing=None,
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
                sevenfold=False,
                dropout=0,
                w_pde_update_rate=1e-4,
                nn_type = 'deeponet',
                sampling_method = "lhs",
                ensemble = 0,
                overwrite = 0,
                verbose = False,
                ):
    
    # config_gpu(gpu_flg = 1)
    # seed = 191
    # np.random.seed(seed)
    
    data_date =  datetime.datetime.utcfromtimestamp(df['times'].min()) 
    
    df_training = df#.sample(frac=0.95, random_state=191)
    df_training.sort_index(inplace=True)
    
    if df_testing is None:
        # df_testing  = df.drop(df_training.index) 
        df_testing  = df.sample(frac=0.01, random_state=0)
    
    suffix = get_filename_suffix(short_naming,
                                data_date, dt, noise_sigma, NS_type,
                                activation, num_hidden_layers,
                                num_neurons_per_layer, n_nodes,
                                n_blocks, w_pde, w_srt, learning_rate,
                                N_pde, laaf, w_init, w_pde_update_rate,
                                dropout  = dropout,
                                sampling_method = sampling_method,
                                init_sigma = init_sigma,
                                ensemble = ensemble
                                )
    
    ###########################
    if filename_model is None:
        filename_model = os.path.join(rpath, 'h%s.h5' %suffix) 

    if os.path.isfile(filename_model) & (overwrite == 0) :
        print('File exist ',filename_model)
        print('Avoiding training, if you want to redo the training use -- overwrite 1')
        return(filename_model)
    
    msis = None
    # try:
    #     msis = MSIS(data_date,
    #                  glat=df_training['lats'].mean(),
    #                  glon=df_training['lons'].mean(),
    #                  time_range=dt,
    #                  plot_values=True)
    # except:
    #     msis = None
    
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
                NS_type = NS_type,
                nn_type = nn_type,
                laaf=laaf,
                dropout  = dropout,
                init_sigma = init_sigma,
            )
    
    # with tf.device("/device:GPU:0"):
    nn.train(df_training,
             df_testing,
             # t, x, y, z, u, v, w,
             epochs     = nepochs,
             filename   = filename_model,
             w_pde_update_rate = w_pde_update_rate,
             lr      = learning_rate,
             w_data  = w_data,
             w_div   = w_pde,
             w_mom   = w_pde,
             w_temp  = w_pde,
             w_srt   = w_srt,
             ns_pde  = N_pde,
             sampling_method = sampling_method,
             # saving_rate = 10000,
             # NS_type  = NS_type,
             )

    nn.save(filename_model)
    
    if not verbose:
        return( filename_model )
    
    
    # filename_model = os.path.join(rpath, 'h%s.keras' %suffix) 
    # nn.model.save(filename_model)
    
    figname01 = os.path.join(rpath, 'loss_%s.png' %suffix)
    
    nn.plot_loss_history(figname=figname01)
    
    ############################################################
    #Validation points
    
    figname02 = os.path.join(rpath, 'testing_winds_%s.png' %suffix)
    figname03 = os.path.join(rpath, 'testing_errors_%s.png' %suffix)
    figname04 = os.path.join(rpath, 'testing_errors_k_%s.png' %suffix)
    figname05 = os.path.join(rpath, 'testing_statistics_%s.png' %suffix)
    figname_P = os.path.join(rpath, 'testing_pressure_%s.png' %suffix)
    
    outputs = nn.infer_from_df(df_testing)
    
    u_nn = outputs[:,0]
    v_nn = outputs[:,1]
    w_nn = outputs[:,2]
    
    df_testing["u_hyper"] = u_nn
    df_testing["v_hyper"] = v_nn
    df_testing["w_hyper"] = w_nn
    
    # d_nn = -(u_nn*kx_test + v_nn*ky_test + w_nn*kz_test)
    
    nn.plot_statistics(df_testing,
                       figname=figname05)
    
    nn.plot_solution(df_testing,
                    figname_winds=figname02,
                     # figname_errs=figname03,
                     # figname_errs_k=figname04,
                     # figname_pressure=figname_P,
                     )
    
    return( filename_model )

if __name__ == '__main__':
    
    from multiprocessing import Process
    
    #delay in mins
    delay = 0#120+120+240
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    parser.add_argument('-e', '--exp', dest='exp', default='vortex', help='Experiment configuration')
    
    parser.add_argument('-d', '--dpath', dest='dpath', default=None, help='Data path')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Resource path')
    
    parser.add_argument('-n', '--neurons-per_layer',  dest='neurons_per_layer', default=128, help='# kernel', type=int)
    parser.add_argument('-l', '--hidden-layers',      dest='hidden_layers', default=5, help='# kernel layers', type=int)
    parser.add_argument('-c', '--nodes',              dest='n_nodes', default=0, help='# nodes', type=int)
    parser.add_argument('--nblocks',                  dest='n_blocks', default=0, help='', type=int)
    
    parser.add_argument('--npde',                     dest='N_pde', default=5000, help='', type=int)
    parser.add_argument('--ns',                       dest='nepochs', default=5000, help='', type=int)
    
    parser.add_argument('--learning-rate',      dest='learning_rate', default=1e-3, help='', type=float)
    parser.add_argument('--pde-weight-upd-rate', dest='w_pde_update_rate', default=1e-5, help='', type=float)
    
    parser.add_argument('--data-weight',        dest='w_data', default=1e0, help='data fidelity weight', type=float)
    parser.add_argument('--pde-weight',         dest='w_pde', default=1e-5, help='PDE weight', type=float)
    parser.add_argument('--srt-weight',        dest='w_srt', default=1e-1, help='Slope recovery time loss weight', type=float)
    
    parser.add_argument('--laaf',        dest='nn_laaf', default=0, type=int)
    parser.add_argument('--dropout',     dest='nn_dropout', default=0, type=int)
    
    parser.add_argument('--pde',        dest='NS_type', default="VV", help='Navier-Stokes formulation, either VP (velocity-pressure) or VV (velocity-vorticity)')
    parser.add_argument('--noutputs',   dest='noutputs', default=3, help='', type=int)
    
    parser.add_argument('--nensembles',   dest='nensembles', default=10, help='Generates a number of ensembles to compute the statistical uncertainty of the model', type=int)
    
    parser.add_argument('--clustering-filter',   dest='ena_clustering', default=1, help='Apply clustering filter to the meteor data', type=int)
    
    parser.add_argument('--noise', dest='noise_sigma', default=0.0, help='', type=float)
    
    parser.add_argument('--architecture', dest='nn_type', default='respinn', help='select the network architecture: gpinn, respinn, ...')
    parser.add_argument('--version',     dest='nn_version', default=3.00, type=float)
    parser.add_argument('--activation',  dest='nn_activation', default='sine')
    
    parser.add_argument('--sampling_method',  dest='sampling_method', default='random')
    
    parser.add_argument('-s', '--nn-init-std',    dest='nn_init_sigma', default=1.0, type=float)
    parser.add_argument('-i', '--nn-w-init',    dest='nn_w_init', default='GlorotNormal', type=str)
    
    parser.add_argument('--time-window', dest='dtime', default=24, help='hours', type=int)
    parser.add_argument('--initime',    dest='tini', default=0, help='hours', type=float)
    
    parser.add_argument('--verbose', dest='verbose', default=0, help='', type=int)
    parser.add_argument('--overwrite', dest='overwrite', default=0, help='', type=int)
    
    parser.add_argument('--lon-range', dest='dlon', default=None, help='degrees', type=float)
    parser.add_argument('--lat-range', dest='dlat', default=None, help='degrees', type=float)
    parser.add_argument('--alt-range', dest='dh', default=None, help='km', type=float)
    
    parser.add_argument('--lon-center', dest='lon_center', default=None, help='degrees', type=float)
    parser.add_argument('--lat-center', dest='lat_center', default=None, help='degrees', type=float)
    parser.add_argument('--alt-center', dest='alt_center', default=None, help='km', type=float)
    
    parser.add_argument('--output-file', dest='filename_model', default=None, help='')
    parser.add_argument('--output-file-short-naming', dest='short_naming', default=1, type=int)
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
    
    ena_clustering  = args.ena_clustering
    
    num_outputs     = args.noutputs
    num_ensembles   = args.nensembles
    
    path            = args.dpath
    resource_path   = args.rpath
    
    w_pde           = args.w_pde
    w_data          = args.w_data
    w_srt           = args.w_srt
    
    w_pde_update_rate = args.w_pde_update_rate
    
    N_pde           = args.N_pde
    NS_type         = args.NS_type
    
    learning_rate   = args.learning_rate
    noise_sigma     = args.noise_sigma
    nepochs         = args.nepochs
    
    n_nodes         = args.n_nodes
    n_blocks        = args.n_blocks
    
    hidden_layers   = args.hidden_layers
    neurons_per_layer   = args.neurons_per_layer
    
    nn_type         = str.upper(args.nn_type)
    nn_version      = args.nn_version
    nn_activation   = args.nn_activation
    nn_laaf         = args.nn_laaf
    nn_dropout      = args.nn_dropout
    
    nn_init_sigma   = args.nn_init_sigma
    nn_w_init       = args.nn_w_init
    
    verbose         = args.verbose
    overwrite       = args.overwrite
    
    filename_model  = args.filename_model
    realtime        = args.realtime
    
    filename_model_tl = args.filename_model_tl
    transfer_learning = args.transfer_learning
    
    short_naming    = args.short_naming
    
    exp             = args.exp
    
    sampling_method = args.sampling_method
    
    sevenfold       = False
    batch_size      = None
    ##No synthetic noise
    noise_sigma     = 0.0
    df_testing      = None
    paths           = None
    
    single_day = False 
    skip_training = False
    
    if sevenfold:
        nn_version += 0.7
    
    home_directory = "/Users/mcordero"
    
    if exp is not None:
        if exp.upper()  == 'OPERATIONAL':
            
            tini            = 0
            dt              = 3
            
            path            = "%s/Data/IAP/SIMONe/" %home_directory
            noise_sigma     = 0.0
            short_naming    = True
            
            paths           = glob.glob(path+"*/DataE")
            
        elif exp.upper()  == 'DEFAULT':
            
            tini            = 0
            # dt              = 24
            
            path            = "../data/" 
            noise_sigma     = 0.0
        
        elif exp.upper()  == 'FILTER':
            
            tini            = 0
            dt              = 24
            
            short_naming    = True
            
            # path            = "%s/remote/radar/for_miguel/"
            # paths           = glob.glob( os.path.join(path, "*", "Data") )
            
            paths = []
            
            # path            = "%s/remote/METnwPER1"
            # tmp             = glob.glob( os.path.join(path, "*", "*") )
            # paths.extend(tmp)
            
            # resource_path   = "%s/Data/IAP/SIMONe/Peru"
            
            path            = "%s/remote/METnwPER2"  %home_directory
            tmp             = glob.glob( os.path.join(path, "*", "*") )
            paths.extend(tmp)
            
            path            = "%s/remote/METnwCONDOR"  %home_directory
            tmp             = glob.glob( os.path.join(path, "*", "*") )
            paths.extend(tmp)
            
            resource_path   = "%s/remote/scratch/Miguel/for_Fede"  %home_directory
            
            single_day      = True
            skip_training   = True
            
        elif exp.upper()  == 'SIMONE2018':
            
            tini            = 0
            # dt              = 24
            # dlon            = 6
            # dlat            = 3
            # dh              = 24
            
            lon_center      = 12.5
            lat_center      = 54
            alt_center      = 91
            path            = "%s/Data/IAP/SIMONe/Germany/Simone2018"  %home_directory
        
        elif exp.upper()  == 'EXTREMEW':
            
            tini            = 0
            # dt              = 24
            # dlon            = 6
            # dlat            = 3
            # dh              = 24
            
            # lon_center      = 12.5
            # lat_center      = 54
            # alt_center      = 91
            path            = "%s/Data/IAP/SIMONe/Germany/ExtremeW"  %home_directory
        
        
        elif exp.upper()  == 'VORTEX':
            
            tini            = 20
            dt              = 3
            
            dlon            = 400e3
            dlat            = 400e3
            dh              = 25e3
            
            lon_center      = 16.25
            lat_center      = 69.25
            alt_center      = 89
            path            = "%s/Data/IAP/SIMONe/Norway/VorTex"  %home_directory
            
            df_testing = read_vortex_files(path)
            
            single_day      = True
        
        elif exp.upper()  == 'VORTEXHD':
            
            tini            = 19
            # dt              = 4
            
            dlon            = 400e3
            dlat            = 400e3
            dh              = 25e3
            
            lon_center      = 16.25
            lat_center      = 69.25
            alt_center      = 89
            path            = "%s/Data/IAP/SIMONe/Norway/VorTex"  %home_directory
            
            df_testing = read_vortex_files(path)
            
        elif exp.upper()  == 'WAVECONVECTION':
            
            tini            = 0
            # dt              = 24
            # dlon            = 8
            # dlat            = 2.5
            
            lon_center      = 16.25
            lat_center      = 69.25
            alt_center      = 90
            path            = "%s/Data/IAP/SIMONe/Norway/WaveConvection"  %home_directory
        
        elif exp.upper()  == 'EXT24':
            
            tini            = 3
            # dt              = 3
            # dlon            = 7
            # dlat            = 2.4
            # dh              = 16
            
            lon_center      = 16.25
            lat_center      = 69.25
            alt_center      = 89
            path            = "%s/Data/IAP/SIMONe/Norway/ExtremeEvent"  %home_directory
            
        
        elif exp.upper()  == 'EXT1':
            
            tini            = 2.5
            dt              = 4
            dlon            = 400e3
            dlat            = 400e3
            dh              = 16e3
            
            lon_center      = 16.25
            lat_center      = 69.25
            alt_center      = 89
            path            = "%s/Data/IAP/SIMONe/Norway/ExtremeEvent"  %home_directory
            
        elif exp.upper()  == 'EXT2023':
            
            tini            = 10
            dt              = 3
            # dlon            = 600e3
            # dlat            = 350e3
            # dh              = 16e3
            
            lon_center      = 16.3
            lat_center      = 69.75
            alt_center      = 89
            path            = "%s/Data/IAP/SIMONe/Norway/Ext2023_OFF"  %home_directory
            
            single_day      = True
            
        elif exp.upper()  == 'NM':
            
            tini            = 0
            # dt              = 24
            
            # alt_center      = 90
            path            = "%s/Data/IAP/SIMONe/NewMexico/Eclipse"  %home_directory
            
        elif exp.upper()  == 'NM2':
            
            tini            = 0
            # dt              = 24
            
            # alt_center      = 90
            path            = "%s/Data/IAP/SIMONe/NewMexico/EclipseApr"  %home_directory
            
               
        elif exp.upper()  == 'TONGA1':
            
            tini            = 0
            # dt              = 24
            path            = "%s/Data/IAP/SIMONe/Condor/Tonga"  %home_directory
        
        elif exp.upper()  == 'TONGA2':
            
            tini            = 0
            # dt              = 24
            path            = "%s/Data/IAP/SIMONe/JRO/Tonga"  %home_directory
        
        elif exp.upper()  == 'TONGA3':
            
            tini            = 0
            # dt              = 24
            path            = "%s/Data/IAP/SIMONe/Piura/Tonga"  %home_directory
            
        elif exp.upper()  == 'DNS':
            
            path            = "%s/Data/IAP/SIMONe/Virtual/DNS_Simone2018/DNSx10_+12+53+91"  %home_directory
            noise_sigma     = 0
            tini            = 0
            dt              = 4
            
        elif exp.upper()  == 'ICON2015':
            
            tini            = 0
            dt              = 3
            
            path            = "%s/Data/IAP/SIMONe/Virtual/ICON_20160815/ICON_+00+70+90"  %home_directory
            noise_sigma     = 0.2
            
        elif exp.upper()  == 'ICON2016':
            
            path            = "%s/Data/IAP/SIMONe/Virtual/ICON_20160816/ICON_-08+73+90"  %home_directory
            # noise_sigma     = 6.0
        
        else:
            raise ValueError('Experiment option not implemented ...')
    
    if paths is None:
        paths = [path]
    
    for path in paths:
        
        path = os.path.realpath(path)
        
        if resource_path is None:
            # basepath, exp_name = os.path.split(path)
            rpath = os.path.join(path, 'winds')
        else:
            rpath = resource_path
            
        if not os.path.exists(rpath): os.mkdir(rpath)
        
        #Read meteor data in LLA coordinates
        meteor_data = SMRReader(path, realtime=realtime)
        
        print('Setting new meteor middle point (LLA): ', lon_center, lat_center, alt_center)
        meteor_data.set_spatial_center(lon_center=lon_center,
                                      lat_center=lat_center,
                                      alt_center=alt_center)
        
        #Get the updated center (in case lon, lat and alt_center were None
        lon_center, lat_center, alt_center = meteor_data.get_spatial_center()
        
        while True:
            
            info = meteor_data.read_next_file(enu_coordinates=True, single_day=single_day)
            
            if info != 1: break
            
            exp_date =  datetime.datetime.utcfromtimestamp(meteor_data.df['times'].min())
            
            exp_path = os.path.join(rpath, exp_date.strftime("c%Y%m%d"))
            
            if not ena_clustering:
                exp_path = os.path.join(rpath, exp_date.strftime("nc%Y%m%d"))
                
            if not os.path.exists(exp_path): os.mkdir(exp_path)
            
            #Plot original sampling
            meteor_data.plot_sampling(path=exp_path, suffix='prefilter')
            meteor_data.add_synthetic_noise(noise_sigma)
            
            nblocks = 24//dt
            
            for i in range(nblocks):
            
                ti = tini + i*dt
                
                meteor_data.filter(tini=ti, dt=dt,
                               dlon=dlon, dlat=dlat, dh=dh,
                               sevenfold=sevenfold,
                               ena_clustering=ena_clustering,
                               # path=exp_path,
                              )
                
                # meteor_data.save(exp_path)
            
                #Plot filtered data
                meteor_data.plot_sampling(path=exp_path, suffix='postfilter')
            
                if skip_training:
                    continue
                
                # meteor_data.plot_hist(path=exp_path, suffix='postfilter')
                
                args = [meteor_data.df]
                
                kwargs = {
                            "df_testing":df_testing,
                            "tini":ti,
                            "dt":dt,
                            "dlon":dlon,
                            "dlat":dlat,
                            "dh":dh,
                            "rpath":exp_path,
                            "num_outputs":num_outputs,
                            "w_pde":w_pde,
                            "w_data":w_data,
                            "w_srt":w_srt,
                            "laaf":nn_laaf,
                            "learning_rate":learning_rate,
                            "noise_sigma":noise_sigma,
                            "nepochs":nepochs, 
                            "N_pde":N_pde,
                            "num_neurons_per_layer":neurons_per_layer,
                            "num_hidden_layers":hidden_layers,
                            "n_nodes":n_nodes,
                            "n_blocks":n_blocks,
                            "filename_model":filename_model,
                            "transfer_learning":transfer_learning,
                            "filename_model_tl":filename_model_tl,
                            "short_naming":short_naming,
                            "activation":nn_activation,
                            "init_sigma":nn_init_sigma,
                            "NS_type":NS_type,
                            "w_init":nn_w_init,
                            "lon_center":lon_center,
                            "lat_center":lat_center,
                            "alt_center":alt_center,
                            "batch_size":batch_size,
                            "dropout":nn_dropout,
                            "sevenfold":sevenfold,
                            "w_pde_update_rate":w_pde_update_rate,
                            "nn_type":nn_type,
                            "sampling_method":sampling_method,
                            "overwrite":overwrite,
                            "verbose":verbose,
                    }
                
                for j in range(num_ensembles):
                    kwargs["ensemble"] = j
                    
                    #Start a child process to make sure Tensorflow frees memory after training
                    p = Process(target=train_hyper, args=args, kwargs=kwargs)
                    p.start()
                    p.join()
        