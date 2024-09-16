import os
import time, datetime
import numpy as np

from radar.smr.smr_file import SMRReader
from pinn import hyper as pinn

def plot_loss_terms(dpath, rpath, filename_model):
    
    
    ###########################
    if filename_model is None:
        filename_model = 'model_%s.h5' %suffix
        
    suffix = filename_model.removeprefix('model_')
    
    filename = os.path.join(dpath, filename_model)
    
    # Initialize Neural Network model
    nn = pinn.restore(filename)
    
    figname01 = os.path.join(rpath, 'loss_hist_%s.png' %suffix)    
    
    nn.plot_loss_history(figname=figname01)
    
    
    
    ############################################################
    #Traning points
    # figname02 = os.path.join(rpath, 'winds_%s_train.png' %suffix)
    # figname03 = os.path.join(rpath, 'errors_%s_train.png' %suffix)
    # figname04 = os.path.join(rpath, 'errors_k_%s_train.png' %suffix)
    # figname05 = os.path.join(rpath, 'statistics_%s_train.png' %suffix)
    # figname_P = os.path.join(rpath, 'pressure_%s_train.png' %suffix)
    #
    # outputs = nn.infer(t, x, y, z)
    #
    # u_nn = outputs[:,0]
    # v_nn = outputs[:,1]
    # w_nn = outputs[:,2]
    #
    # d_nn = -(u_nn*kx + v_nn*ky + w_nn*kz)
    #
    # nn.plot_statistics(
    #                 u, v, w,
    #                 u_nn, v_nn, w_nn,
    #                 figname=figname05)
    #
    # nn.plot_solution(t, x, y, z,
    #                  u, v, w, d,
    #                  u_nn, v_nn, w_nn, d_nn,
    #                   k_x=kx, k_y=ky, k_z=kz,
    #                  figname_winds=figname02,
    #                  figname_errs=figname03,
    #                  figname_errs_k=figname04,
    #                  figname_pressure=figname_P)
    
    ############################################################
    #Validation points
    
    # figname02 = os.path.join(rpath, 'winds_%s_test.png' %suffix)
    # figname03 = os.path.join(rpath, 'errors_%s_test.png' %suffix)
    # figname04 = os.path.join(rpath, 'errors_k_%s_test.png' %suffix)
    # figname05 = os.path.join(rpath, 'statistics_%s_test.png' %suffix)
    # figname_P = os.path.join(rpath, 'pressure_%s_test.png' %suffix)
    #
    # outputs = nn.infer(t_test, x_test, y_test, z_test)
    #
    # u_nn = outputs[:,0]
    # v_nn = outputs[:,1]
    # w_nn = outputs[:,2]
    #
    # d_nn = -(u_nn*kx_test + v_nn*ky_test + w_nn*kz_test)
    #
    # nn.plot_statistics(
    #                 u_test, v_test, w_test,
    #                 u_nn, v_nn, w_nn,
    #                 figname=figname05)
    #
    # nn.plot_solution(t_test, x_test, y_test, z_test,
    #                  u_test, v_test, w_test, d_test,
    #                  u_nn, v_nn, w_nn, d_nn,
    #                 k_x=kx_test, k_y=ky_test, k_z=kz_test,
    #                  figname_winds=figname02,
    #                  figname_errs=figname03,
    #                  figname_errs_k=figname04,
    #                  figname_pressure=figname_P)
    
    return( filename_model )

if __name__ == '__main__':
    
    #delay in mins
    delay = 0#2*60
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    #VORTEX cmapaign
    parser.add_argument('-e', '--exp', dest='exp', default='EXT24', help='Experiment configuration')
    
    #-d /Users/mcordero/Data/IAP/SIMONe/Norway/VorTex --lon-center=16.4 --lat-center=69.3 --alt-center=89 --lon-range=7.5 --lat-range=2.5 --alt-range=14
    
    parser.add_argument('-d', '--dpath', dest='dpath', default=None, help='Data path')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Data path')
    
    args            = parser.parse_args()
    
    dpath            = args.dpath
    rpath           = args.rpath
    
    dpath = '/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/sDMD_L2_sim_nograd_PINN_2.31'
    # dpath = '/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815/SPINN_L2_PINN_1.43'
    
    dpath = '/Users/radar/Data/IAP/SIMONe/Norway/winds/nnRESPINN_10.07'
    dpath = '/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815/winds/nnRESPINN_10.07'
    
    if rpath is None:
        rpath = os.path.join(dpath, 'loss_terms')
    
    if not os.path.exists(rpath): os.mkdir(rpath)
    
    files = os.listdir(dpath)
    
    for ifile in files:
        
        if os.path.splitext(ifile)[1] != '.h5':
            continue
        
        if not os.path.isfile( os.path.join(dpath,ifile)):
            continue
        
        if 'weights' in ifile:
            continue
        
        if 'mean_wind' in ifile:
            continue
        
        plot_loss_terms(dpath, rpath, ifile)
        # except:
        #     pass
        