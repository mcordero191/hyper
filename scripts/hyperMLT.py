import os
import time, datetime
import numpy as np

from radar.smr.smr_file import SMRReader
from radar.smr.winds_hyper import train_hyper

if __name__ == '__main__':
    
    #delay in mins
    delay = 0#230/60. + 300
    only_SMR    = True
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    #VORTEX cmapaign
    parser.add_argument('-e', '--exp', dest='exp', default='dns', help='Experiment configuration')
    
    #-d /Users/radar/Data/IAP/SIMONe/Norway/VorTex --lon-center=16.4 --lat-center=69.3 --alt-center=89 --lon-range=7.5 --lat-range=2.5 --alt-range=14
    
    parser.add_argument('-d', '--dpath', dest='dpath', default=None, help='Data path')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Data path')
    
    parser.add_argument('-n', '--neurons-per_layer',  dest='neurons_per_layer', default=64, help='# kernel', type=int)
    parser.add_argument('-l', '--hidden-layers',      dest='hidden_layers', default=10, help='# kernel layers', type=int)
    parser.add_argument('-c', '--nodes',              dest='n_nodes', default=128, help='# nodes', type=int)
    parser.add_argument('--nblocks',                  dest='n_blocks', default=2, help='', type=int)
    
    parser.add_argument('--npde',                     dest='N_pde', default=5000, help='', type=int)
    parser.add_argument('--ns',                       dest='nepochs', default=5000, help='', type=int)
    
    parser.add_argument('--learning-rate',      dest='learning_rate', default=1e-3, help='', type=float)
    parser.add_argument('--pde-weight-upd-rate', dest='w_pde_update_rate', default=1e-7, help='', type=float)
    
    parser.add_argument('--pde-weight',         dest='w_pde', default=1e-5, help='PDE weight', type=float)
    parser.add_argument('--data-weight',        dest='w_data', default=1e0, help='data fidelity weight', type=float)
    parser.add_argument('--srt-weight',        dest='w_srt', default=1e0, help='Slope recovery time loss weight', type=float)
    
    parser.add_argument('--pde',        dest='NS_type', default="VV", help='Navier-Stokes formulation, either VP (velocity-pressure) or VV (velocity-vorticity)')
    parser.add_argument('--noutputs',   dest='noutputs', default=3, help='', type=int)
    
    parser.add_argument('--noise', dest='noise_sigma', default=0.0, help='', type=float)
    
    parser.add_argument('--architecture', dest='nn_type', default='respinn', help='')
    parser.add_argument('--version',     dest='nn_version', default=3.18, type=float)
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
            path            = "/Users/radar/Data/IAP/SIMONe/NewMexico/EclipseOct"
            
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
            noise_sigma     = -1.0
            tini            = 0
            dt              = 4
            
        elif exp.upper()  == 'ICON2015':
            
            tini            = 0
            dt              = 3
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
        