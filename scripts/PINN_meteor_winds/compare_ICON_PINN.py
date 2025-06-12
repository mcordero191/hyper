'''
Created on 30 Aug 2022

@author: mcordero
'''
import os, time
import datetime
import numpy as np

import h5py

import scipy
DTYPE='float32'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from skimage.filters import gaussian
# from scipy.ndimage import gaussian_filter

from atmospheric_models.ICON import ICONReader
from georeference.geo_coordinates import lla2enu, lla2xyh

from pinn import hyper as pinn
from radar.smr.smr_file import SMRReader

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

def plot_field_ICON(lons, lats, alts,
               fields,
               fields_est=None,
               prefix='',
               path='',
               cmap='seismic',
               titles=None,
               vmins=None,
               vmaxs=None,
               df_sampling=None,
               alpha=1,
               zdecimation=4,
               apply_nan_mask=True,
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
    
    for zi in range(0,nz,zdecimation):
        
        ve = fields_est[0][zi]
        
        if np.all(np.isnan(ve)):
            continue
        
        filename = os.path.join(path, "wind_field_%2.1f_%s.png" %(alts[zi], prefix) )
        
        fig = plt.figure(figsize=(5*nfields,5*nrows))
        plt.suptitle('%s, h=%2.1f km' %(prefix, alts[zi]) )
        
        df_sampling_z = df_sampling[ np.abs(df_sampling['heights'] - alts[zi]) <= 0.6 ]
        
        samp_lons = df_sampling_z['lons'].values
        samp_lats = df_sampling_z['lats'].values
        
        for iplot in range(nfields):
            
            mask = None
            if nrows>1:
                ax = fig.add_subplot(nrows, nfields, nfields+iplot+1) #, projection='3d')
                
                if fields_est[iplot] is None: continue
                
                ve = fields_est[iplot][zi]
                
                vmin = vmins[iplot]
                vmax = vmaxs[iplot]
                
                if vmin is None: vmin = np.nanmin(ve)
                if vmax is None: vmax = np.nanmax(ve)
                
                im = ax.pcolormesh(lons, lats, ve,
                                  cmap=cmap,
                                  vmin=vmin,
                                  vmax=vmax)
                
                ax.plot(samp_lons, samp_lats, 'mx')
                
                # im = ax.plot_surface(LON, LAT, ve,
                #                  cmap=cmap,
                #                  vmin=vmin,
                #                  vmax=vmax,
                #                  alpha=alpha)
                #
                # ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x')
                # ax.set_zlim(vmin,vmax)
                
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('%s_est' %(titles[iplot]) )
                
                plt.colorbar(im, ax=ax)
                
                if apply_nan_mask:
                    mask = np.isnan(ve)
            
            ax = fig.add_subplot(nrows, nfields, iplot+1) #, projection='3d')
        
            v = fields[iplot][zi]
            
            v = gaussian(v, 1.5)
            
            if mask is not None:
                v[mask] = np.nan
            
            vmin = vmins[iplot]
            vmax = vmaxs[iplot]
            
            
            
            if vmin is None: vmin = np.nanmin(v)
            if vmax is None: vmax = np.nanmax(v)
            
            im = ax.pcolormesh(lons, lats, v,
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax)
            ax.plot(samp_lons, samp_lats, 'mx')
            
            # 
            # im = ax.plot_surface(LON, LAT, v,
            #                      cmap=cmap,
            #                      vmin=vmin,
            #                      vmax=vmax,
            #                      alpha=alpha)
            #
            # ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x')
            # ax.set_zlim(vmin, vmax)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('%s' %(titles[iplot]) )
            
            # ax.set_xlim(-2,2)
            # ax.set_ylim(0,2)
            
            plt.colorbar(im, ax=ax)
            
            if nrows>2:
                ax = fig.add_subplot(nrows, nfields, 2*nfields+iplot+1) #,projection='3d')
        
                idx = np.where(np.isfinite(ve))
                
                f1 = v[idx] - np.mean(v[idx])
                f2 = ve[idx] - np.mean(ve[idx])
                
                corr = np.dot(f1, f2)/np.sqrt(np.nansum(f1**2)*np.nansum(f2**2))
                
                f = (v - ve)**2/(v**2)
                mse = np.sqrt( np.nanmean(f) )
                
                im = ax.pcolormesh(lons, lats, f,
                                  cmap='binary',
                                  vmin=0,
                                  vmax=1)
                
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('RMSE=%3.2f, Corr=%3.2f' %(mse, corr) )
                
                ax.plot(samp_lons, samp_lats, 'mx')
                
                plt.colorbar(im, ax=ax)
                
            
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

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
    
    plot_field(lons, lats, alts, u, path=path, title='u', prefix=prefix)
    plot_field(lons, lats, alts, v, path=path, title='v', prefix=prefix)
    plot_field(lons, lats, alts, w, path=path, title='w', prefix=prefix,
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

        
def calculate_spectra(s, x, axis=0):
    '''
    Inputs:
        s    :    Signal. array-like. (nz, ny, nx)
        x    :    Space grid. array-like. (nz, ny, nx)
    '''
    
    s = np.where(np.isfinite(s), s, 0)
    
    n = x.shape[axis]
    indices = np.arange(n)
    
    # dx = np.max( np.take(x, indices[1:], axis=axis) - np.take(x, indices[:-1], axis=axis) )
    
    dx = (np.max(x) - np.min(x))/n
    
    freqs = np.fft.rfftfreq(n, dx)
    
    e = np.einsum('f,zyx->zyxf', 2j*np.pi*freqs, x)
    
    if axis==0:
        F = 1.0/np.sqrt(n)*np.einsum('zyx,zyxf->yxf', s, np.exp(e) )
    elif axis==1:
        F = 1.0/np.sqrt(n)*np.einsum('zyx,zyxf->zxf', s, np.exp(e) )
    elif axis==2:
        F = 1.0/np.sqrt(n)*np.einsum('zyx,zyxf->zyf', s, np.exp(e) )
    else:
        raise ValueError
    
    # F = np.mean( F, axis=axis ) #Get rid of the extra dimension
    
    Fm = np.mean( np.abs(F)**2, axis=(0,1) )
    
    return(Fm, freqs)
   
def plot_spectra(F1s, F2s, k, filename='./test.png'):
    
    l = k**(-5./3.)#*1e-7
    
    F1s = np.array(F1s)
    F2s = np.array(F2s)
    
    
    F1 = np.nanmean(F1s, axis=0)
    F2 = np.nanmean(F2s, axis=0)
    
    # F1 /= np.nanmax(F1)
    # F2 /= np.nanmax(F2)
    
    l *= 1e3/np.nanmax(l)
    
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
    plt.xlim(3e-7, 3e-4)
    plt.ylim(1e0, 1e4)
    plt.grid()
    
    plt.savefig(filename)
    plt.close()
    


# def profile_cut(u, axis=(1,2)):
#
#     sigma = [0]*3
#
#     for i in axis: sigma[i] = 1
#
#     u = gaussian_filter(u, sigma=sigma)
#
#     # nz, ny, nx = u.shape
#
#     index = u.shape[axis[1]]//2
#     u0 = np.take(u, index, axis=axis[1])
#
#     index = u.shape[axis[0]]//2
#     u0 = np.take(u0, index, axis=axis[0])
#
#     return u0
    
def main(path_ICON_model, 
         path_meteor_data,
         this_file,
         subfolder,
         log_index,
         units,
         gradients,
         plot_fields=False,
         save_cuts=False,
         ):
    
    units = 'm'
    # if float(subfolder[-5:]) < 8:
    #     units = 'km'
    
    scaling_factor = 1
    if units == 'km':
        scaling_factor = 1e-3
        
    # path_data = os.path.split( os.path.realpath(path_meteor_data) )[0]
    # path_PINN = os.path.join(path_data, 'winds', subfolder)
    
    path_PINN, filename = os.path.split(this_file)
    
    figpath = os.path.join(path_PINN, 'plots')
    
    if not os.path.exists(figpath):os.mkdir(figpath)
        
    figpath = os.path.join(figpath, 'full_%s' %log_index)
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    # else:
    #     return
    
    alt_range = 18
    lon_range = 8
    lat_range = 4
    
    decX = 2
    decY = 2
    decZ = 1
    
    # if not (plot_fields):
    #     decX = 5
    #     decY = 5
    #     decZ = 1
    
    
    nn = pinn.restore(this_file, log_index=log_index)
    
    lat_center = nn.lat_ref
    lon_center = nn.lon_ref
    alt_center = nn.alt_ref
    
    print("Meteor center (LLA): %2.1f, %2.1f, %2.1f" %(lon_center, lat_center, alt_center) )
    
    tmin, xmin, ymin, zmin = nn.lb
    tmax, xmax, ymax, zmax = nn.ub
    
    # tmax = tmin + 3*60*60
    
    meteor_sys = SMRReader(path_meteor_data)
    meteor_sys.set_spatial_center(lat_center, lon_center, alt_center)
    
    info = meteor_sys.read_next_file(enu_coordinates=True)
    df_meteor = meteor_sys.df
    
    mask  = np.abs(df_meteor['lons'] - lon_center) < lon_range/2.
    mask &= np.abs(df_meteor['lats'] - lat_center) < lat_range/2.
    # mask &= np.abs(df_meteor['alts'] - alt_center) < alt_range/2.
    
    df_meteor = df_meteor[mask]
    
    #Read model data in LLA coordinates
    model_sys = ICONReader(path_ICON_model,
                         alt_range = alt_range,
                         lon_range = lon_range,
                         lat_range = lat_range,
                         lat_center = lat_center,
                         lon_center = lon_center,
                         alt_center = alt_center,
                         )
    
    lats = model_sys.lat_grid[::decY]
    lons = model_sys.lon_grid[::decX]
    alts = model_sys.alt_grid[::decZ]
    
    ALT, LAT, LON = np.meshgrid(alts,lats,lons, indexing='ij')
    
    X,Y,Z = lla2xyh(LAT, LON, ALT,
                    lat_center = lat_center,
                    lon_center = lon_center,
                    alt_center = alt_center,
                    units=units,
                    )
    
    print("Dataset dimension:", ALT.shape )
    
    #Dimensions [alt, lat, lon] <> [z,y,x]
    dz, _, _ = derivative(Z)
    _, dy, _ = derivative(Y)
    _, _, dx = derivative(X)
    
    vmins = [-50,-50,-5, None]
    vmaxs = [ 50, 50, 5, None]
    
    vmins_der = [-2e-3,-2e-3,-2e-3, None]
    vmaxs_der = [ 2e-3, 2e-3, 2e-3, None]
    
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
    
    Su_xs = []
    Sv_xs = []
    Su_ys = []
    Sv_ys = []
    
    Sue_xs = []
    Sve_xs = []
    Sue_ys = []
    Sve_ys = []
    
    
    while True:
        df = model_sys.read_next_block()
        
        if df is None: break  
        
        time = int(df['epoch'])
        
        if time < tmin:
            print('x', end=' ')
            continue
        
        if time > tmax : #tmin + 0.1*(tmax-tmin):
            print('x', end=' ')
            break
        
        times.append(time)
        dt = datetime.datetime.utcfromtimestamp(time)
        
        u = df['u'][::decZ, ::decY, ::decX]
        v = df['v'][::decZ, ::decY, ::decX]
        w = df['w'][::decZ, ::decY, ::decX]
        # P = df['P'][::decZ, ::decY, ::decX]
        # rho = df['rho'][::decZ, ::decY, ::decX]
        
        # rho0 = np.median(rho)#, axis=(1,2))
        # drho = (rho-rho0)/rho0
    
        if gradients:
            w_z, _, _ = derivative(w)
            _, v_y, _ = derivative(v)
            _, _, u_x = derivative(u)
            
            w_zz, _, _ = derivative(w_z)
            _, v_yy, _ = derivative(v_y)
            _, _, u_xx = derivative(u_x)
            
            u_x = u_x/dx * scaling_factor
            v_y = v_y/dy * scaling_factor
            w_z = w_z/dz * scaling_factor
            
            u_xx = u_xx/dx**2 * scaling_factor**2
            v_yy = v_yy/dy**2 * scaling_factor**2
            w_zz = w_zz/dz**2 * scaling_factor**2
        
        shape = u.shape
        
        t = np.zeros_like(u) + time
        
        func = nn.infer
        if gradients: func = nn.infer_2nd_gradients
        
        outs = func(t.flatten(),
                     LON.flatten(), LAT.flatten(), ALT.flatten(),
                     x = X.flatten(), y=Y.flatten(), z=Z.flatten(),
                     filter_output=False,
                     )
        
        ue  = outs[:,0]
        ve  = outs[:,1]
        we  = outs[:,2]
        
        if outs.shape[1] > 3:
            drhoe = outs[:,3]
        
        ue = np.reshape(ue,shape)
        ve = np.reshape(ve,shape)
        we = np.reshape(we,shape)
        
        #mask data
        mask = np.isnan(ue)
        
        u[mask] = np.nan
        v[mask] = np.nan
        w[mask] = np.nan
        
        if gradients:
            ue_x  = outs[:,3] * scaling_factor
            ve_y  = outs[:,4] * scaling_factor
            we_z  = outs[:,5] * scaling_factor
            
            ue_xx  = outs[:,6] * scaling_factor**2
            ve_yy  = outs[:,7] * scaling_factor**2
            we_zz  = outs[:,8] * scaling_factor**2
            
            ue_x = np.reshape(ue_x,shape)
            ve_y = np.reshape(ve_y,shape)
            we_z = np.reshape(we_z,shape)
            
            ue_xx = np.reshape(ue_xx,shape)
            ve_yy = np.reshape(ve_yy,shape)
            we_zz = np.reshape(we_zz,shape)
        
        if drhoe is not None:
            drhoe = np.reshape(drhoe,shape)
        
        rmse_lon, rmse_lat, rmse_alt  = calc_rmse(fields = [u, v, w],
                                                  fields_est = [ue, ve, we],
                                                  vmaxs = vmaxs,
                                                  type='corr',
                                                  )
        rmse_alt_log.append(rmse_alt)
        rmse_lat_log.append(rmse_lat)
        rmse_lon_log.append(rmse_lon)
        
        #Mean values
        # func = np.nanmean
        # func = np.nanstd
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
        
        
        # Su_x, kx = calculate_spectra(u, X, axis=2)
        # Sv_x, kx = calculate_spectra(v, X, axis=2)
        #
        # Su_y, ky = calculate_spectra(u, Y, axis=1)
        # Sv_y, ky = calculate_spectra(v, Y, axis=1)
        #
        # Su_xs.append(Su_x)
        # Sv_xs.append(Sv_x)
        # Su_ys.append(Su_y)
        # Sv_ys.append(Sv_y)
        #
        # Su_x, kx = calculate_spectra(ue, X, axis=2)
        # Sv_x, kx = calculate_spectra(ve, X, axis=2)
        #
        # Su_y, ky = calculate_spectra(ue, Y, axis=1)
        # Sv_y, ky = calculate_spectra(ve, Y, axis=1)
        #
        # Sue_xs.append(Su_x)
        # Sve_xs.append(Sv_x)
        # Sue_ys.append(Su_y)
        # Sve_ys.append(Sv_y)
        
        if not plot_fields:
            continue
        
        df_t = df_meteor[ np.abs(df_meteor['times']-time) < 300]
        
        plot_field(lons, lats, alts,
                   fields = [u, v, w],
                   fields_est = [ue, ve, we],
                   titles=['u','v','w', 'rho'],
                   vmins = vmins,
                   vmaxs = vmaxs,
                   prefix='winds_%s' %dt.strftime('%Y%m%d_%H%M%S'),
                   path=figpath,
                   df_sampling=df_t,
                   coord='lla')
        
        if gradients:
            plot_field(lons, lats, alts,
                       fields = [u_x, v_y, w_z],
                       fields_est = [ue_x, ve_y, we_z],
                       titles=[r'$u_x$',r'$v_y$',r'$w_z$'],
                       vmins = vmins_der,
                       vmaxs = vmaxs_der,
                       prefix='grad_1st_%s' %dt.strftime('%Y%m%d_%H%M%S'),
                       path=figpath,
                       df_sampling=df_t,
                       coord='lla')
            
            plot_field(lons, lats, alts,
                       fields = [u_xx, v_yy, w_zz],
                       fields_est = [ue_xx, ve_yy, we_zz],
                       titles=[r'$u_{xx}$',r'$v_{yy}$',r'$w_{zz}$', 'rho'],
                       vmins = vmins_der2,
                       vmaxs = vmaxs_der2,
                       prefix='grad_2nd_%s' %dt.strftime('%Y%m%d_%H%M%S'),
                       path=figpath,
                       df_sampling=df_t,
                       coord='lla')
        
        if save_cuts:
            filename = os.path.join(figpath, 'winds_%s.h5' %dt.strftime('%Y%m%d_%H%M%S'))
            
            save_h5file(alts, lats, lons,
                        u=u, v=v, w=w,
                        ue=ue, ve=ve, we=we,
                        filename=filename)
            
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
    
    # figname = os.path.join(figpath, 'spectra_ux.png')
    # plot_spectra(Su_xs, Sue_xs, kx, figname)
    #
    # figname = os.path.join(figpath, 'spectra_vx.png')
    # plot_spectra(Sv_xs, Sve_xs, kx, figname)
    #
    # figname = os.path.join(figpath, 'spectra_uy.png')
    # plot_spectra(Su_ys, Sue_ys, ky, figname)
    #
    # figname = os.path.join(figpath, 'spectra_vy.png')
    # plot_spectra(Sv_ys, Sve_ys, ky, figname)
    
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
    
    vmins = [0]*3
    vmaxs = [1]*3
    cmap = "inferno"
    
    prefix = 'corr'
    
    figname1 = os.path.join(figpath, '%s_per_lon.png' %prefix)
    figname2 = os.path.join(figpath, '%s_per_lat.png' %prefix)
    figname3 = os.path.join(figpath, '%s_per_alt.png' %prefix)
    
    plot_rmses(times, lons,
             rmse_lon_log,
             labels=['u','v','w'],
             figname=figname1,
             cmap=cmap,
             vmins=vmins,
             vmaxs=vmaxs)
    
    plot_rmses(times, lats,
             rmse_lat_log,
             labels=['u','v','w'],
             figname=figname2,
             cmap=cmap,
             vmins=vmins,
             vmaxs=vmaxs)
    
    plot_rmses(times, alts,
             rmse_alt_log,
             labels=['u','v','w'],
             figname=figname3,
             cmap=cmap,
             vmins=vmins,
             vmaxs=vmaxs)
        
    
if __name__ == '__main__':
    
    import glob
    
    # path_ICON_model     = '/Users/radar/Data/IAP/Models/UA ICON/nest3_20160815'
    # path_data           = '/Users/radar/Data/IAP/VirtualRadar/ICON_+00+70+90'
    # path_data           = '/Users/radar/Data/IAP/SIMONe/Germany'
    #
    # path_meteor_data = os.path.join(path_data, 'Simone2018')
    # path_PINN = os.path.join(path_data, 'PINN_xyh_v1.0_m1e-10')
    #
    # model_name = 'model_4-06-64-TA_n00_1.0e+04-1.0e+04-1.0e-04-TF_1.0e+04_0.h5'
    # model_name = 'model_4-06-64-TA_n00_1.0e+03-1.0e+04-1.0e-03-TF_1.0e+04_0.h5'
    # model_name = 'model_4-06-64-TA_n00_1.0e+02-1.0e+04-1.0e-04-TF_1.0e+04_0.h5'
    # model_name = 'model_4-06-64-TA_n00_1.0e+04-1.0e+04-1.0e-03-TF_1.0e+04_0.h5'
    # # model_name = 'model_4-06-64-TA_n01_1.0e+04-1.0e+04-1.0e-04-TF_1.0e+04_0.h5' 
    # # model_name = 'model_4-06-64-TA_n01_1.0e+04-5.0e+03-1.0e-04-TF_1.0e+04_0.h5'
    #
    # # model_name = 'model_o4-h06-d64-TA_n0.0_a1.0e+04-i1.0e+04-lr1.0e-03-TF_NS1.0e+04_0.h5'
    #
    # model_name = 'model_o4-h06-d64-TA_n0.0_a1.0e+04-i1.0e+04-lr1.0e-03-TF_NS1.0e+04_0.h5'
    # model_name = 'model_o4-h06-d64-TA_n1.0_a1.0e+04-i1.0e+04-lr1.0e-03-TF_NS1.0e+04_0.h5'
    # model_name = 'model_o4-h06-d64-TA_n0.2_a1.0e+04-i1.0e+04-lr1.0e-03-TF_NS1.0e+04_0.h5'
    # model_name = 'model_o4-h06-d64-TA_n6.0_a1.0e+04-i1.0e+04-lr1.0e-03-TF_NS1.0e+04_0.h5'
    #
    # model_name = 'model_o3-h06-d64-TA_n0.0_a1.0e+04-i1.0e+04-lr1.0e-03-TF_NS1.0e+04_0.h5'
    #
    # #########New version ###
    # path_ICON_model     = '/Users/radar/Data/IAP/Models/UA ICON/nest3_20160816'
    # path_meteor_data = '/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160816/ICON_+00+70+90'
    # model_name = 'model_20160816-000000_n2.0_o3-h03-d64-TA_a1.0e+04-i1.4e+04-lr1.0e-02-TF_NS1.0e+04_0.h5'
    #
    # path_ICON_model     = '/Users/radar/Data/IAP/Models/UA ICON/nest3_20160815'
    # path_meteor_data = '/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815/ICON_+00+70+90'
    # model_name = 'model_20160815-000000_n2.0_o3-h06-d64-TA_a1.0e+04-i1.5e+04-lr1.0e-02-TF_NS1.0e+04_0.h5'
    # subfolder = 'PINN_xyh_vFNC_3.03n'
    
    # path_ICON_model     = '/Users/radar/Data/IAP/Models/UA ICON/nest3_20160816'
    # path_meteor_data    = "/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160816/ICON_-08+73+90"
    # model_name          = 'model_20160816-000000_w24_n1.0_NS[VV]_o3_atanh_l06_d512_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_1.00.h5'
   
    path_ICON_model  = '/Users/radar/Data/IAP/Models/UA ICON/nest3_20160815'
    path_meteor_data = '/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815/ICON_+00+70+90'
    # model_name       = 'model_20160815-000000_w03_n1.0_NS[VV_div]_o3_atanh_l06_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_None.h5'
    # model_name       = 'model_20160815-000000_w03_n1.0_NS[VV]_o3_atanh_l06_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_None.h5'
    # model_name      = 'model_20160815-000000_w24_n1.0_NS[VV_div]_o3_atanh_l06_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_None.h5'
    # model_name      = 'model_20160815-000000_w24_n1.0_NS[VV]_o3_atanh_l06_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_None.h5'
    # model_name      = 'model_20160815-000000_w24_n1.0_NS[VV]_o3_atanh_l12_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_None.h5'
    # subfolder       = 'resDOL1v2PINN_221.01'
    #
    # model_name      = 'model_20160815-000000_w24_n1.0_NS[VV_div]_o3_atanh_l12_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_None.h5'
    #
    # model_name      = 'model_20160815-000000_w24_n1.0_NS[VV]_o3_atanh_l12_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_uni_None.h5'
    # subfolder       = 'res2DOL1v2PINN_221.01'
    #
    # #Paper
    # model_name      = 'model_20160815-000000_w03_n1.0_NS[VV]_o3_atanh_l06_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS1.0e+04_0_uni_None.h5'
    # subfolder       = 'fcDOL1v2PINN_221.02_paper'
    #

    # model_name      = 'model_20160815-000000_w03_n1.0_NS[VV_noNu]_o3_asine_l05_d512_w1.0e+00_w1.0e-10_lr1.0e-03_NS5.0e+03_0_1_uni_0.5.h5'
    # subfolder       = 'fcDOL1v2PINN_251.00'
    #
    # model_name      = 'model_20160815-000000_w24_n6.0_NS[VV_hydro_noNu]_o3_asine_l05_d128_w1.0e+00_w1.0e-05_lr1.0e-03_NS1.0e+04_0_1_uni_1.5.h5'
    # subfolder       = 'fcDOL2v2PINN_313.11'
    #
    # model_name      = 'model_20160815-000000_w24_n6.0_NS[VV_noNu]_o3_asine_l05_d128_w1.0e+00_w1.0e-05_lr1.0e-03_NS1.0e+04_0_1_uni_1.5.h5'
    # subfolder       = 'fcDOL2v2PINN_313.01'
    #
    # model_name  =   'model_20160815-000000_w24_n6.0_NS[VP_div]_o3_asine_l04_d256_w1.0e+00_w1.0e+00_lr1.0e-03_NS5.0e+03_0_1_He_None.h5'
    # subfolder   =   'MRL1v3.00_ur=1e-4_PINN_1.31'
    #
    # model_name  =   'model_20160815-000000_w03_n0.0_NS[VP_div]_o3_asine_l04_n128_d016_w1.0e+00_w1.0e-03_lr1.0e-03_NS5.0e+03_0_0_He_None_ur=1.0e-03.h5'
    
    
    subfolder   =   'hWIND_VV_hydro_noNul03.02.256_w1.0e-05lr1.0e-03lf0ur1.0e-05T24'
    model_name      = None
    
    log_index       = None
    
    units           = 'm'
    gradients       = False
    plot_fields     = False
    save_cuts       = False
    
    path_data = os.path.split( os.path.realpath(path_meteor_data) )[0]
    path_PINN = os.path.join(path_meteor_data,  subfolder)
    
    if model_name is None:
        files = glob.glob(os.path.join(path_PINN, '*/*.h5'))
        files = sorted(files)
    else:
        files = [model_name] 
    
    
    for this_file in files:
        
        print("\nProcessing %s\n" %this_file)
        
        main(path_ICON_model, 
             path_meteor_data,
             this_file,
             subfolder,
             log_index,
             units,
             gradients,
             plot_fields,
             save_cuts,
             )