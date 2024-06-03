'''
Created on 30 Aug 2022

@author: mcordero
'''
import os, time, glob
import datetime
import numpy as np

import h5py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter

from atmospheric_models.DNS import DNSReader
from georeference.geo_coordinates import lla2enu, lla2xyh

from pinn import hyper as pinn
from radar.specular_meteor_radars.SMR import SMRReader, filter_data

from utils.io import save_h5file
from utils.plotting import plot_field, plot_rmses
from utils.operations import profile_cut, calc_rmse


def derivative(u):
    '''
    Calculate the derivative of "u" in the x, y and z direction
    
    Returns:
        u_x, u_y, u_z with the same dimension as u.
        Invalid derivative points are nans.
    '''
    
    u_x = np.empty_like(u) + np.nan
    u_y = np.empty_like(u) + np.nan
    u_z = np.empty_like(u) + np.nan
    
    u_x[1:, :, :] = u[1:,:,:] - u[:-1,:,:]
    u_y[ :,1:, :] = u[:,1:,:] - u[:,:-1,:]
    u_z[ :, :,1:] = u[:,:,1:] - u[:,:,:-1]
    
    return(u_x, u_y, u_z)


def plot_field_plane(lons, lats, alts,
               fields,
               fields_est=None,
               prefix='',
               path='',
               cmap='RdBu_r',
               titles=None,
               vmins=None,
               vmaxs=None,
               df_sampling=None,
               alpha=1,
               ):
    
    nfields = len(fields)
    nz, ny, nx = fields[0].shape
    
    if titles is None: titles = ['']*nfields
    if vmins is None: vmins = [None]*nfields
    if vmaxs is None: vmaxs = [None]*nfields
    
    if fields_est is None:
        nrows = 1
    else:
        nrows = 3
        
    LON, LAT = np.meshgrid(lons, lats)
    
    for zi in range(nz):
        
        filename = os.path.join(path, "wind_field_%s_%2.1f.png" %(prefix, alts[zi]) )
        
        fig = plt.figure(figsize=(6*nfields,5*nrows))
        
        df_sampling_z = df_sampling[ np.abs(df_sampling['heights'] - alts[zi]) < 1.2 ]
        
        samp_lons = df_sampling_z['lons'].values
        samp_lats = df_sampling_z['lats'].values
        
        for iplot in range(nfields):
            ax = fig.add_subplot(nrows, nfields, iplot+1)
        
            f = fields[iplot]
            vmin = vmins[iplot]
            vmax = vmaxs[iplot]
            
            if vmin is None: vmin = np.nanmin(f[zi])
            if vmax is None: vmax = np.nanmax(f[zi])
            
            im = ax.pcolormesh(lons, lats, f[zi],
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('%s %2.1f km' %(titles[iplot], alts[zi]) )
            
            ax.plot(samp_lons, samp_lats, 'kx')
            
            # ax.set_xlim(-2,2)
            # ax.set_ylim(0,2)
            
            plt.colorbar(im, ax=ax)
            
            if nrows>1:
                ax = fig.add_subplot(nrows, nfields, nfields+iplot+1)
        
                f = fields_est[iplot]
                
                if f is None:
                    continue
                
                im = ax.pcolormesh(lons, lats, f[zi],
                                  cmap=cmap,
                                  vmin=vmin,
                                  vmax=vmax)
                
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('%s_est %2.1f km' %(titles[iplot], alts[zi]) )
                
                ax.plot(samp_lons, samp_lats, 'kx')
                
                plt.colorbar(im, ax=ax)
            
            if nrows>2:
                ax = fig.add_subplot(nrows, nfields, 2*nfields+iplot+1)
        
                f = fields[iplot] - fields_est[iplot]
                vmin = vmins[iplot]
                vmax = vmaxs[iplot]
                
                im = ax.pcolormesh(lons, lats, f[zi],
                                  cmap='RdBu',
                                  vmin=-10,
                                  vmax=10)
                
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('%s error %2.1f km' %(titles[iplot], alts[zi]) )
                
                ax.plot(samp_lons, samp_lats, 'kx')
                
                plt.colorbar(im, ax=ax)
                
            
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
def calc_divergence(u, v, w, time,
                    lons, lats, alts,
                    path=''):
    
    u_z, u_y, u_x = derivative(u)
    v_z, v_y, v_x = derivative(v)
    w_z, w_y, w_x = derivative(w)
    
    u_x /= dx
    v_y /= dy
    w_z /= dz
    
    hor_div = u_x + v_y
    
    prefix = '20181102_%05d' %(time)
    
    plot_field(lons, lats, alts, u, path=path, title='U', prefix=prefix)
    plot_field(lons, lats, alts, v, path=path, title='V', prefix=prefix)
    plot_field(lons, lats, alts, w, path=path, title='W', prefix=prefix,
               vmin=-5, vmax=5)
    
    plot_field(lons, lats, alts, u_x, path=path, title='u_x', prefix=prefix,
               vmin=-5, vmax=5)
    plot_field(lons, lats, alts, v_y, path=path, title='v_y', prefix=prefix,
               vmin=-5, vmax=5)
    plot_field(lons, lats, alts, w_z, path=path, title='w_z', prefix=prefix,
               vmin=-5, vmax=5)
    
    plot_field(lons, lats, alts, hor_div, path=path, title='horizontal div', prefix=prefix,
               vmin=-5, vmax=5)

def plot_comparison(
                      u, v, w,
                      u_p, v_p, w_p,
                      k_x=None, k_y=None, k_z=None,
                      figpath='./',
                      epoch=0,
                      alpha=0.2):
        
    figname='./test.png'
    figname_winds='./winds.png'
    figname_errs='./errors.png'
    figname_errs_k='./errors_k.png'

    fig = plt.figure(figsize=(9,3))
    
    ax = fig.add_subplot(131)
    ax.plot(u.flatten(), u_p.flatten(), 'o', alpha=alpha)
    ax.plot([-60,60],[-60,60], 'r--')
    ax.set_xlabel('$u$')
    ax.set_ylabel('$u_p$')
    ax.grid(True)
    ax.set_xlim(-60,60)
    ax.set_ylim(-60,60)
    
    ax = fig.add_subplot(132)
    ax.plot(v.flatten(), v_p.flatten(), 'o', alpha=alpha)
    ax.plot([-60,60],[-60,60], 'r--')
    ax.set_xlabel('$v$')
    ax.set_ylabel('$v_p$')
    ax.grid(True)
    ax.set_xlim(-60,60)
    ax.set_ylim(-60,60)
    
    ax = fig.add_subplot(133)
    ax.plot(w.flatten(), w_p.flatten(), 'o', alpha=alpha)
    ax.plot([-10,10],[-10,10], 'r--')
    ax.set_xlabel('$w$')
    ax.set_ylabel('$w_p$')
    ax.grid(True)
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

        
def calculate_spectra(y, t, axis=0):
    '''
    Inputs:
        y    :    array-like. (nz, ny, nx)
        t    :    array-like. (nz, ny, nx)
    '''
    
    y = np.where(np.isfinite(y), y, 0)
    
    n = t.shape[axis]
    indices = np.arange(n)
    
    dt = np.max( np.take(t, indices[1:], axis=axis) - np.take(t, indices[:-1], axis=axis) )
    
    freqs = np.fft.rfftfreq(n, dt)
    
    e = np.einsum('f,zyx->zyxf', -2j*np.pi*freqs, t)
    
    if axis==0:
        F = np.einsum('zyx,zyxf->yxf', y, np.exp(e) )
    elif axis==1:
        F = np.einsum('zyx,zyxf->zxf', y, np.exp(e) )
    elif axis==2:
        F = np.einsum('zyx,zyxf->zyf', y, np.exp(e) )
    else:
        raise ValueError
    
    # F = np.mean( F, axis=axis ) #Get rid of the extra dimension
    
    Fm = np.mean( 1.0/n*np.abs(F)**2, axis=(0,1) )
    
    return(Fm, freqs)
   
def plot_spectra(F1s, F2s, k, filename='./test.png'):
    
    l = k**(-5/3.)#*1e-7
    
    F1s = np.array(F1s)
    F2s = np.array(F2s)
    
    
    F1 = np.nanmean(F1s, axis=0)
    F2 = np.nanmean(F2s, axis=0)
    
    # F1 /= np.nanmax(F1)
    # F2 /= np.nanmax(F2)
    
    l /= np.nanmax(l)*1e3
    
    # plt.subplot(211)
    # plt.plot(k, F1, 'ko-', label='Truth')
    # plt.plot(k, l, 'r--')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.legend()
    # plt.xlim(5e-6, 5e-4)
    # plt.grid()
    
    plt.subplot(111)
    plt.plot(k, F1s.T, 'k-', alpha=0.1)
    plt.plot(k, F2s.T, 'b-', alpha=0.1)
    
    plt.plot(k, F1, 'ko', label='Truth')
    plt.plot(k, F2, 'bo', label='Estimated')
    plt.plot(k, l, 'r--')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlim(5e-6, 5e-4)
    plt.grid()
    
    plt.savefig(filename)
    plt.close()


    
# def plot_rmses(times, y, rmses, labels,
#                      figname=None,
#                      cmap='inferno',
#                      vmaxs=None):
#
#     '''
#     Inputs:
#         times    :    1D ARRAY
#         y    :    1D ARRAY
#         rmses   :    dimension [ntimes, nfields, nalt]
#
#         labels    :    dimension []
#                         :
#
#     '''
#
#     rmses =  np.array(rmses)
#
#     nplots = rmses.shape[1]
#
#     x = mdates.epoch2num(times)
#     locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
#     formatter = mdates.ConciseDateFormatter(locator)
#
#     fig, axs = plt.subplots(nplots, 1, sharex=True, sharey=True)
#
#     for i in range(nplots):
#
#         ax = axs[i]
#         rmse = rmses[:,i,:]
#         label = labels[i]
#
#         if vmaxs is not None: vmax = vmaxs[i]
#         else: vmax = None
#
#         im = ax.pcolormesh(x, y, rmse.T,
#                            cmap=cmap,
#                            vmin=0, vmax=vmax)
#
#         plt.colorbar(im, ax=ax, label=label)
#
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(formatter)
#
#     plt.tight_layout()
#
#     if figname is not None:
#         plt.savefig(figname)
#     else:
#         plt.show()
#     plt.close()
     
def main(model_name, decS=1, decZ=10,
         plot_corr=False,
         plot_rti=False,
         zdecimation=1,
         ):
    
    figpath = os.path.join(path_PINN, 'final_%s' %(os.path.splitext(model_name)[0]) )
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    figpath = os.path.join(figpath, '%s' %(log_index) )
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    figwinds = os.path.join(figpath, 'winds')
    
    if not os.path.exists(figwinds):
        os.mkdir(figwinds)
        
    binpath = os.path.join(figpath, 'outs')
    
    if not os.path.exists(binpath):
        os.mkdir(binpath)
        
    filename = os.path.join(path_PINN, model_name)
    
    # alt_range = 8
    # lon_range = 4
    # lat_range = 2.5
    
    nn = pinn.restore(filename, log_index=log_index)
    
    lat_center = nn.lat_ref
    lon_center = nn.lon_ref
    alt_center = nn.alt_ref
    
    # print("NN center (LLA): %2.1f, %2.1f, %2.1f" %(lon_center, lat_center, alt_center) )
    
    tmin, xmin, ymin, zmin = nn.lb
    tmax, xmax, ymax, zmax = nn.ub
    
    meteor_sys = SMRReader(path_meteor_data)
    meteor_sys.set_spatial_center(lat_center, lon_center, alt_center)
    
    info = meteor_sys.read_next_file(enu_coordinates=True)
    df_meteor = meteor_sys.df
    
    df_meteor = filter_data(df_meteor, only_SMR=only_SMR)
    
    # mask  = np.abs(df_meteor['lons'] - lon_center) < lon_range/2.
    # mask &= np.abs(df_meteor['lats'] - lat_center) < lat_range/2.
    # mask &= np.abs(df_meteor['heights'] - alt_center) < alt_range/2.
    #
    # df_meteor = df_meteor[mask]
    
    #Read model data in LLA coordinates
    model_sys = DNSReader(path_DNS_model, decS=decS, decZ=decZ, dimZ=100)
    
    # lat_center, lon_center, alt_center = meteor_sys.get_spatial_center()
    # model_sys.set_meteor_spatial_center(lat_center, lon_center, alt_center)
    
    # alt_center = 91.3
    
    ini_time = meteor_sys.get_initial_time()
    model_sys.set_meteor_initial_time(ini_time)
    
    dx = 0#36419.681261896214
    dy = 0#-5903.283308092517
    dz = 0#0.3
    
    lons = model_sys.x#*1e3 + dx
    lats = model_sys.y#*1e3 + dy
    alts = (model_sys.z + alt_center)#*1e3
    
    X = model_sys.X*1e3 + dx
    Y = model_sys.Y*1e3 + dy
    Z = (model_sys.Z + alt_center)*1e3
    
    #Dimensions [alt, lat, lon] <> [z,y,x]
    dz, _, _ = derivative(Z)
    _, dy, _ = derivative(Y)
    _, _, dx = derivative(X)
    
    vmins = [-10,-10,-5, None]
    vmaxs = [ 10, 10, 5, None]
    
    vmins_der = [-1e-3,-1e-3,-1e-3, None]
    vmaxs_der = [ 1e-3, 1e-3, 1e-3, None]
    
    vmins_der2 = [-2e-7,-2e-7,-2e-6, None]
    vmaxs_der2 = [ 2e-7, 2e-7, 2e-6, None]
    
    Fx = 0 
    Fxe = 0
    
    N = 0
    
    times = []
    rmse_alt_log = []
    rmse_lat_log = []
    rmse_lon_log = []
    
    ux_log = []
    uy_log = []
    uz_log = []
    
    uex_log = []
    uey_log = []
    uez_log = []
    
    drhoe = None
    
    ue = np.empty_like(X, dtype=np.float32)
    ve = np.empty_like(X, dtype=np.float32)
    we = np.empty_like(X, dtype=np.float32)
    
    ue_x = np.empty_like(X, dtype=np.float32)
    ve_x = np.empty_like(X, dtype=np.float32)
    we_x = np.empty_like(X, dtype=np.float32)
    
    ue_y = np.empty_like(X, dtype=np.float32)
    ve_y = np.empty_like(X, dtype=np.float32)
    we_y = np.empty_like(X, dtype=np.float32)
    
    ue_z = np.empty_like(X, dtype=np.float32)
    ve_z = np.empty_like(X, dtype=np.float32)
    we_z = np.empty_like(X, dtype=np.float32)
    
    nz = ue.shape[0]
    shape = ue[0].shape
    
    
    j = 0
    while True:
        
        skip_block = False
        
        # if j%2 != 0 :
        #     skip_block = True
        
        j += 1
        if j >= 10:
            break
        
        df = model_sys.read_next_block(skip_block=skip_block)
        if df is None: break  
        
        if len(df.keys()) == 0:
            continue
        
        time = int(df['time'])
        
        if time < tmin:
            print('x', end=' ')
            continue
        if time > tmax:
            print('x', end=' ')
            continue
        
        times.append(time)
        dt = datetime.datetime.utcfromtimestamp(time)
        
        u = df['u'] #-30
        v = df['v'] #-35
        w = df['w']
        # P = df['P']
        # rho = df['rho']
        
        # rho0 = np.median(rho)#, axis=(1,2))
        # drho = (rho-rho0)/rho0
    
        if gradients:
            w_z, _, _ = derivative(w)
            _, v_y, v_x = derivative(v)
            _, u_y, u_x = derivative(u)
            
            # w_zz, _, _ = derivative(w_z)
            # _, v_yy, _ = derivative(v_y)
            # _, _, u_xx = derivative(u_x)
            
            u_x = u_x/dx
            u_y = u_y/dy
            
            v_x = v_x/dx
            v_y = v_y/dy
            
            w_z = w_z/dz
            
            # u_xx = u_xx/dx**2
            # v_yy = v_yy/dy**2
            # w_zz = w_zz/dz**2
        
        t = np.zeros_like(u) + time
        
        func = nn.infer
        if gradients: func = nn.infer_gradients
        
        for i in range(nz):
            outs = func(t[i].flatten(), x = X[i].flatten(), y=Y[i].flatten(), z=Z[i].flatten())
            
            uet  = outs[:,0]#-30
            vet  = outs[:,1]#-35
            wet  = outs[:,2]
        
            uet = np.reshape(uet, shape)
            vet = np.reshape(vet, shape)
            wet = np.reshape(wet, shape)
            
            ue[i] = uet
            ve[i] = vet
            we[i] = wet
        
            if gradients:
                # raise NotImplementedError
            
                ue_xt  = outs[:,3]
                ue_yt  = outs[:,4]
                
                ve_xt  = outs[:,6]
                ve_yt  = outs[:,7]
                
                we_zt  = outs[:,11]
                
                ue_xt = np.reshape(ue_xt, shape)
                ue_yt = np.reshape(ue_yt, shape)
                
                ve_xt = np.reshape(ve_xt, shape)
                ve_yt = np.reshape(ve_yt, shape)
                
                we_zt = np.reshape(we_zt, shape)
                
                ue_x[i] = ue_xt
                ue_y[i] = ue_yt
                
                ve_x[i] = ve_xt
                ve_y[i] = ve_yt
                
                we_z[i] = we_zt
            
                # ue_xx  = outs[:,6]
                # ve_yy  = outs[:,7]
                # we_zz  = outs[:,8]
                #
                # ue_xx = np.reshape(ue_xx,shape)
                # ve_yy = np.reshape(ve_yy,shape)
                # we_zz = np.reshape(we_zz,shape)
            
        #mask data
        # mask = np.isnan(ue)
        #
        # u[mask] = np.nan
        # v[mask] = np.nan
        # w[mask] = np.nan
        
        df_t = df_meteor[ np.abs(df_meteor['times']-time) < 5*60]
        
        if plot_corr:
            rmse_lon, rmse_lat, rmse_alt  = calc_rmse(fields = [u, v, w],
                                                      fields_est = [ue, ve, we],
                                                      vmaxs = vmaxs,
                                                      type  = rti_type,
                                                      )
            rmse_alt_log.append(rmse_alt)
            rmse_lat_log.append(rmse_lat)
            rmse_lon_log.append(rmse_lon)
        
        if plot_rti:
            #Mean values
            func = np.nanmean
            func = np.nanstd
            func = profile_cut
            
            uz0 = func(u, axis=(1,2))
            uy0 = func(u, axis=(0,2))
            ux0 = func(u, axis=(0,1))
            
            vz0 = func(v, axis=(1,2))
            vy0 = func(v, axis=(0,2))
            vx0 = func(v, axis=(0,1))
            
            wz0 = func(w, axis=(1,2))
            wy0 = func(w, axis=(0,2))
            wx0 = func(w, axis=(0,1))
            
            uz_log.append([uz0, vz0, wz0])
            uy_log.append([uy0, vy0, wy0])
            ux_log.append([ux0, vx0, wx0])
            
            #Mean values EST
            uz0 = func(ue, axis=(1,2))
            uy0 = func(ue, axis=(0,2))
            ux0 = func(ue, axis=(0,1))
            
            vz0 = func(ve, axis=(1,2))
            vy0 = func(ve, axis=(0,2))
            vx0 = func(ve, axis=(0,1))
            
            wz0 = func(we, axis=(1,2))
            wy0 = func(we, axis=(0,2))
            wx0 = func(we, axis=(0,1))
            
            uez_log.append([uz0, vz0, wz0])
            uey_log.append([uy0, vy0, wy0])
            uex_log.append([ux0, vx0, wx0])
        
        if plot_fields:
            plot_field(lons, lats, alts,
                       fields = [u, v, w],
                       fields_est = [ue, ve, we],
                       titles=['u','v','w', '\rho'],
                       vmins = vmins,
                       vmaxs = vmaxs,
                       prefix='%s' %dt.strftime('%Y.%m.%d_%H:%M:%S'),
                       path=figwinds,
                       df_sampling=df_t,
                        zdecimation=zdecimation,
                       )
        
        if gradients:
            
            a0 = u_x + v_y + w_z
            a1 = v_x - u_y
            
            b0 = ue_x + ve_y + we_z
            b1 = ve_x - ue_y
            
            plot_field(lons, lats, alts,
                       fields = [u_x, v_y, a0],
                       fields_est = [ue_x, ve_y, b0],
                       titles=['ux','vy','Div'],
                       vmins = vmins_der,
                       vmaxs = vmaxs_der,
                       prefix='grad_1st_%s' %dt.strftime('%Y%m%d_%H%M%S'),
                       path=figwinds,
                       df_sampling=df_t,
                       zdecimation=zdecimation,
                       )
            
            # plot_field(lons, lats, alts,
            #            fields = [u_xx, v_yy, w_zz],
            #            fields_est = [ue_xx, ve_yy, we_zz],
            #            titles=['u_xx','v_yy','w_zz', 'rho'],
            #            vmins = vmins_der2,
            #            vmaxs = vmaxs_der2,
            #            prefix='grad_2nd_%s' %dt.strftime('%Y%m%d_%H%M%S'),
            #            path=figwinds,
            #            df_sampling=df_t,
            #            zdecimation=zdecimation,
            #            )
        
        if save_cuts:
            # filename = os.path.join(figpath, 'winds_%s.h5' %dt.strftime('%Y%m%d_%H%M%S'))
            #
            # save_h5file(alts, lats, lons,
            #             u=u, v=v, w=w,
            #             ue=ue, ve=ve, we=we,
            #             filename=filename)
            
            # filename = os.path.join(figpath, 'winds_%s.h5' %dt.strftime('%Y%m%d_%H%M%S'))
            
            model_sys.write_data(ue, ve, we, path=binpath)
            
        # F, k = calculate_spectra(u, x=Z, axis=0)
        # Fx = Fx + F
        #
        # F, k = calculate_spectra(ue, x=Z, axis=0)    
        # Fxe = Fxe + F
        #
        # N+=1
        #
        # spec_figname = os.path.join(figpath, 'spectra_%02d.png' %N)
        #
        # plot_spectra(Fx/N, Fxe/N, k, filename=spec_figname)
    
    if plot_corr:
        figname1 = os.path.join(figpath, 'u0_lon.png')
        figname2 = os.path.join(figpath, 'u0_lat.png')
        figname3 = os.path.join(figpath, 'u0_alt.png')
        
        plot_rmses(times, lons,
                 ux_log,
                 labels=['u','v','w'],
                 figname=figname1,
                 vmins=vmins,
                 vmaxs=vmaxs)
        
        plot_rmses(times, lats,
                 uy_log,
                 labels=['u','v','w'],
                 figname=figname2,
                 vmins=vmins,
                 vmaxs=vmaxs)
        
        plot_rmses(times, alts,
                 uz_log,
                 labels=['u','v','w'],
                 figname=figname3,
                 vmins=vmins,
                 vmaxs=vmaxs)
        
        figname1 = os.path.join(figpath, 'ue0_lon.png')
        figname2 = os.path.join(figpath, 'ue0_lat.png')
        figname3 = os.path.join(figpath, 'ue0_alt.png')
        
        plot_rmses(times, lons,
                 uex_log,
                 labels=['u','v','w'],
                 figname=figname1,
                 vmins=vmins,
                 vmaxs=vmaxs)
        
        plot_rmses(times, lats,
                 uey_log,
                 labels=['u','v','w'],
                 figname=figname2,
                 vmins=vmins,
                 vmaxs=vmaxs)
        
        plot_rmses(times, alts,
                 uez_log,
                 labels=['u','v','w'],
                 figname=figname3,
                 vmins=vmins,
                 vmaxs=vmaxs)
    
    #################
    
    if plot_rti:
        vmins = [0]*3
        vmaxs = [1]*3
        cmap = "inferno"
        
        figname1 = os.path.join(figpath, '%s_per_lon.png' %rti_type)
        figname2 = os.path.join(figpath, '%s_per_lat.png' %rti_type)
        figname3 = os.path.join(figpath, '%s_per_alt.png' %rti_type)
        
        plot_rmses(times, lons,
                 rmse_lon_log,
                 labels=['u','v','w'],
                 figname=figname1,
                 vmins=vmins,
                 vmaxs=vmaxs,
                 cmap=cmap)
        
        plot_rmses(times, lats,
                 rmse_lat_log,
                 labels=['u','v','w'],
                 figname=figname2,
                 vmins=vmins,
                 vmaxs=vmaxs,
                 cmap=cmap)
        
        plot_rmses(times, alts,
                 rmse_alt_log,
                 labels=['u','v','w'],
                 figname=figname3,
                 vmins=vmins,
                 vmaxs=vmaxs,
                 cmap=cmap)
       
if __name__ == '__main__':
    
    path_DNS_model     = '/Users/radar/Data/IAP/Models/DNS/StratifiedHD/outs'
    path_meteor_data = '/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/DNSx10_+12+53+91'
    
    path_DNS_model     = '/Users/radar/Data/IAP/Models/DNS/NonStratified'
    path_meteor_data = '/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/DNSx10_+12+53+91'
    
    #Paper sampling x1
    model_name      = 'model_20181102-000000_w03_n1.0_NS[VV]_o3_atanh_l06_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS1.0e+04_0_uni_None.h5'
    subfolder       = 'fcDOL1v2PINN_221.01_paper'
    
    #Paper sampling x10
    # model_name      = 'model_20181102-000000_w03_n1.0_NS[VV]_o3_atanh_l06_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_None.h5'
    # subfolder       = 'fcDOL1v2PINN_221.02_paperx10'
    
    model_name = 'model_20181102-000000_w01_n0.0_NS[VP_div]_o3_asine_l04_d128_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_1_He_None.h5'
    subfolder = 'DMDL2_ur=1e-3_PINN_4.00'
    
    model_name = 'mean_model_20181102-000000_w04_n0.3_NS[VV]_o3_asine_l02_n064_d005_b02_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_0_Glorot_None_ur=1.0e-05.h5'
    subfolder = 'nnDEEPMULTINET_12.20'
    
    model_name = 'mean_model_20181102-000000_w04_n0.3_NS[VV]_o3_asine_l02_n064_d005_b02_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_0_Glorot_None_ur=1.0e-05.h5'
    subfolder = 'nnDEEPMULTINET_12.02'
    
    model_name = 'mean_model_20181102-000000_w04_n0.3_NS[VV]_o3_asine_l03_n064_d032_b02_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_0_HeNorm_None_ur=1.0e-05.h5'
    subfolder = 'nnIPINN_15.03'
    
    model_name = None #'mean_model_20181102-000000_w04_n0.3_NS[VV]_o3_asine_l03_n064_d032_b02_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_0_HeNorm_None_ur=1.0e-05.h5'
    subfolder = 'nnRESPINN_13.02'
    
    log_index       = None
    
    units           = 'm'
    gradients       = True
    plot_fields     = True
    only_SMR        = False
    rti_type        = 'corr'
    save_cuts       = True
        
    path_data = os.path.split( os.path.realpath(path_meteor_data) )[0]
    path_PINN = os.path.join(path_data, "winds", subfolder)
    
    if model_name is None:
        models = glob.glob1(path_PINN, '*model*[!s].h5')
        models = sorted(models)
    else:
        models = [  model_name ]
    
    for model_name in models[:]:
        # try:
        main(model_name, decS=1, decZ=1,
             plot_rti=False,
             zdecimation=20,
             )
        # except:
        #     continue
        