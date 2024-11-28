'''
Created on 30 Aug 2022

@author: mcordero
'''
import os, glob
import time, datetime
import numpy as np

import h5py

from scipy.ndimage import convolve1d

import tensorflow as tf
DTYPE='float32'

import matplotlib.pyplot as plt
# plt.rcParams['axes.facecolor'] = "#B0B0B0"

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.dates as mdates

from atmospheric_models.ICON import ICONReader
from georeference.geo_coordinates import lla2enu

from PINN import PINN
from radar.specular_meteor_radars.SMR import SMRReader
    
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

def plot_field(lons, lats, alts,
               fields,
               fields_est=None,
               prefix='',
               path='',
               figtitle='',
               cmap='RdBu_r',
               titles=None,
               vmins=None,
               vmaxs=None,
               df_sampling=None,
               alpha=1,
               xlabel='East-West (km)',
               ylabel='North-South (km)'
               ):
    
    nfields = len(fields)
    nz, ny, nx = fields[0].shape
    
    if titles is None: titles = ['']*nfields
    if vmins is None: vmins = [None]*nfields
    if vmaxs is None: vmaxs = [None]*nfields
    
    if fields_est is None:
        nrows = 1
    else:
        nrows = 2
        
    LON, LAT = np.meshgrid(lons, lats)
    
    for zi in range(nz):
        
        filename = os.path.join(path, "wind_field_%s_%2.1f.png" %(prefix, alts[zi]) )
        
        fig = plt.figure(figsize=(5*nfields,5*nrows))
        plt.suptitle(figtitle)
        
        if df_sampling is not None:
            df_sampling_z = df_sampling[ np.abs(df_sampling['heights'] - alts[zi]) < 1.2 ]
        
            samp_lons = df_sampling_z['lons'].values
            samp_lats = df_sampling_z['lats'].values
        
        for iplot in range(nfields):
            ax = fig.add_subplot(nrows, nfields, iplot+1, projection='3d')
        
            f = fields[iplot]
            vmin = vmins[iplot]
            vmax = vmaxs[iplot]
            
            if vmin is None: vmin = np.min(f[zi])
            if vmax is None: vmax = np.max(f[zi])
            
            # im = ax.pcolormesh(lons, lats, f[zi],
            #                   cmap=cmap,
            #                   vmin=vmin,
            #                   vmax=vmax)
            
            im = ax.plot_surface(LON, LAT, f[zi],
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax,
                                 alpha=alpha)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('%s %2.1f km' %(titles[iplot], alts[zi]) )
            
            ax.set_zlim(vmin, vmax)
            
            # ax.plot(samp_lons, samp_lats, 'mx')
            if df_sampling is not None:
                ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x')
            
            # ax.set_xlim(-2,2)
            # ax.set_ylim(0,2)
            
            plt.colorbar(im, ax=ax)
            
            if nrows>1:
                ax = fig.add_subplot(nrows, nfields, nfields+iplot+1, projection='3d')
        
                f = fields_est[iplot]
                #
                # im = ax.pcolormesh(lons, lats, f[zi],
                #                   cmap=cmap,
                #                   vmin=vmin,
                #                   vmax=vmax)
                
                im = ax.plot_surface(LON, LAT, f[zi],
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax,
                                 alpha=alpha)
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title('%s_est %2.1f km' %(titles[iplot], alts[zi]) )
                
                ax.set_zlim(vmin,vmax)
                # ax.plot(samp_lons, samp_lats, 'mx')
                if df_sampling is not None: 
                    ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x')
                
                plt.colorbar(im, ax=ax)
            
            if nrows>2:
                ax = fig.add_subplot(nrows, nfields, 2*nfields+iplot+1) #,projection='3d')
        
                f = fields[iplot] - fields_est[iplot]
                vmin = vmins[iplot]
                vmax = vmaxs[iplot]
                
                im = ax.pcolormesh(lons, lats, np.abs(f[zi]),
                                  cmap='inferno',
                                  vmin=0,
                                  vmax=10)
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title('%s error %2.1f km' %(titles[iplot], alts[zi]) )
                
                if df_sampling is not None:
                    ax.plot(samp_lons, samp_lats, 'mx')
                
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
    
    u_x /= x_x
    v_y /= y_y
    w_z /= z_z
    
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
        
def delete_files(path, pattern='*.png'):
    
    # Search files with .txt extension in current directory
    files = glob.glob0(path, pattern)
    
    # deleting the files with txt extension
    for file in files:
        os.remove(file)

def _plot_3d_map(x3d, y3d, z1d,
                 data, ax,
                 title="",
                 vmin=None, vmax=None,
                 xmin=None, xmax=None,
                 ymin=None, ymax=None,
                 zmin=None, zmax=None,
                 cmap_label='',
                 cmap = 'jet',
                 contour = None,
                 zdecimator=1):
    """
    Input:
        data    :    array-like (nz, ny, nx )
    """
    
    nranges, ny, nx = data.shape
    
    
#     plt.suptitle("Power Spectra " + title, size=18.0, y=1.0)
    
#     cmap = plt.get_cmap(cmap)
    
#     cmap_min = cmap(0)
#     cmap_max = cmap(255)
#     
#     cmap.set_bad(cmap_min, 1)
#     cmap.set_over(color=cmap_min, alpha=1)
#     cmap.set_under(color=cmap_max, alpha=1)
    
    for i in range(0, nranges, zdecimator):
        
        d = data[i, :, :]
        
#         ax.contourf(x3d[:,:,i], y3d[:,:,i], np.zeros_like(d), 2,
#                    zdir='z', offset=z1d[0,0,i],
#                    alpha=1.0,
#                    colors='w',
#                    vmin=vmin, vmax=vmax,
#                    antialiased=False,
#                     hatches=['////'])
#         ax.patch.set_color('.25')
        
        ax.contourf(x3d[i,:,:], y3d[i,:,:], d,
                   zdir='z', offset=z1d[i],
                   levels=np.linspace(vmin,vmax,50),
                   cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   antialiased=False
                    )
                  
    plt.tick_params(labelsize=22)
    ax.set_xlabel('\nEast-West (km)', fontsize=25, linespacing=2.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel('\nNorth-South (km)', fontsize=25, linespacing=2.8)
    ax.set_ylim(ymin, ymax)
    ax.set_zlabel('\nAltitude (km)', fontsize=25, linespacing=3)
    ax.set_zlim(zmin, zmax)
    # ax.dist = 10
    ax.view_init(elev=25, azim=270-30)    
#     ax.view_init(elev=27, azim=270-20)
    
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([vmin, vmax])
    m.set_clim(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(m, ax=ax, shrink=0.2, label=cmap_label, pad=0.0001) #, orientation='horizontal')
    
    cbar.ax.tick_params(labelsize=21) 
    
    ax.set_aspect('equalxy')
    ax.set_box_aspect([4,4,6])
    
#     plt.colorbar(cs, shrink=0.5, label=cmap_label, pad=0.0001)
    
    plt.title(title, size=28.0)
    
    return
        
def plot_3d_maps(x3d, y3d, z1d,
                 data_list,
                 title="",
                 title_list= ["SNR (dBs)", "Radial velocity (m/s)", "Delta radial vel (m/s)"],
                 vmin_list=None,
                 vmax_list=None,
                 xmin=None, xmax=None,
                 ymin=None, ymax=None,
                 filename=None,
                 cmap_label='',
                 cmap_list = None,
                 zdecimator=2):
    """
    Input:
        data_list    :    array-like (nplots, nranges, ny, nx)
    """
    nplots = len(data_list)
    _,ny,nx = data_list[0].shape
    
    nz = 3
    nx_subplot = nplots
    ny_subplot = 1

    xmin = np.min(x3d)
    xmax = np.max(x3d)
    ymin = np.min(y3d)
    ymax = np.max(y3d)
    zmin = np.min(z1d)
    zmax = np.max(z1d)
        
    fig = plt.figure(1, figsize=(8*nx_subplot, 3*nz), facecolor='w')
    fig.suptitle(title, fontsize = 30)
    
    cmap = 'seismic'
    
    for i in range(nplots):
        ax = fig.add_subplot(ny_subplot, nx_subplot, i+1, projection='3d')
        
        data = data_list[i].copy()
        
        if cmap_list is not None:
            cmap = cmap_list[i]
        
        if vmin_list[i] is not None:
            vmin = vmin_list[i]
        else:
            vmin = np.nanmin(data)
            
        if vmax_list[i] is not None:
            vmax = vmax_list[i]
        else:
            vmax = np.nanmax(data)
            
        
#         data = np.where(np.isnan(data), vmin_list[i], data)
        data = np.clip(data, vmin, vmax) 
        
        contour = None
        
        _plot_3d_map(x3d, y3d, z1d,
                 data, ax, title=title_list[i],
                 vmin=vmin, vmax=vmax,
                 xmin=xmin, xmax=xmax,
                 ymin=ymin, ymax=ymax,
                 zmin=zmin, zmax=zmax,
                 cmap_label=cmap_label,
                 cmap = cmap,
                 contour=contour,
                 zdecimator=zdecimator)
    
    plt.tight_layout()
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        
    return

def _plot_3d_vector(x, y,
                    u, v, w,
                    ax,
                     title="",
                     xmin=None, xmax=None,
                     ymin=None, ymax=None,
                     wmin=None, wmax=None,
                     scale=None,
                     cmap='RdYlBu_r',
                     cmap_label='',
                 ):
    """
    Input:
        x,y        :    x and y position (ny, nx)
        u,v,w      :    wind field. array-like (ny, nx )
        
        scale     :    Number of data units per arrow length unit,
                        e.g., m/s per plot width.
                        A smaller scale parameter makes the arrow longer.
    """
    
    ny, nx = u.shape
    
    # im = None
    im = ax.pcolormesh(x,y,w, cmap=cmap, vmin=wmin, vmax=wmax, alpha=0.3)
    
    im0 = ax.quiver(x, y, u, v,
                   # w,
                   # cmap=cmap,
                   # clim=[wmin, wmax],
                   scale=scale,
                   )
    
    ax.set(xlabel='East-West (km)', ylabel ='North-South (km)')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    ax.set_title(title)
    
    return(im)

def plot_3d_vectors(x3d, y3d, z1d,
                 data_list,
                 title="",
                 title_list= ["SNR (dBs)", "Radial velocity (m/s)", "Delta radial vel (m/s)"],
                 vmin_list=None,
                 vmax_list=None,
                 xmin=None, xmax=None,
                 ymin=None, ymax=None,
                 filename=None,
                 cmap_label='Vertical velocity (m/s)',
                 cmap = 'RdBu_r',
                 zdecimator=1,
                 scale=None):
    """
    Input:
        [u,v,w]    :    array-like (nplots, nranges, ny, nx)
    """
    u, v, w = data_list
    
    u = u[::zdecimator]
    v = v[::zdecimator]
    w = w[::zdecimator]
    z1d = z1d[::zdecimator]
    
    nz, ny, nx = u.shape
    
    nx_subplot = int( np.ceil( np.sqrt(nz) ) )
    ny_subplot = int( nz / nx_subplot )

    xmin = np.min(x3d)
    xmax = np.max(x3d)
    ymin = np.min(y3d)
    ymax = np.max(y3d)
    zmin = np.min(z1d)
    zmax = np.max(z1d)
    
    wmin = vmin_list[2]
    wmax = vmax_list[2]
    
    # plt.style.use('dark_background')
    fig, axs = plt.subplots(ny_subplot, nx_subplot,
                       sharex=True, sharey=True,
                       figsize=(5.2*nx_subplot, 5*ny_subplot))
    
    fig.suptitle(title)
    
    for i, ax in enumerate(axs.flat):
        
        if i >= nz: break
        
        im = _plot_3d_vector(x3d[i,:,:], y3d[i,:,:],
                        u[i,:,:], v[i,:,:], w[i,:,:],
                        ax,
                        title='h=%2.1f km' %z1d[i],
                        xmin=xmin, xmax=xmax,
                        ymin=ymin, ymax=ymax,
                        wmin=wmin, wmax=wmax,
                        scale=scale,
                        cmap=cmap,
                        cmap_label=cmap_label,
                        )
    
    for ax in axs.flat:
        ax.label_outer()
    
    plt.tight_layout()
    
    if im is not None:
        cbar = fig.colorbar(im,
                            ax=axs.ravel().tolist(),
                            shrink=0.99,
                            label='Vertical velocity (m/s)',
                            pad=0.01,
                            fraction=0.05,
                            aspect=50,
                            )
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        
    return

def plot_mean_winds(t, h, u, v, w,
                    figfile='./test.png',
                    vmins=None,
                    vmaxs=None,
                    cmap='seismic',
                    ylabel='Altitude (km)'):
    
    vmin = None
    vmax = None
    
    num = mdates.epoch2num(t)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    
    plt.figure(figsize=(8,6))

    ax = plt.subplot(311)
    ax.set_title('Zonal wind')
    ax.set_ylabel(ylabel)
    
    if vmins is not None: vmin = vmins[0]
    if vmaxs is not None: vmax = vmaxs[0]
    
    im = ax.pcolormesh(num, h, u.T, cmap=cmap, vmin=vmin, vmax=vmax)
    
    plt.colorbar(im, ax=ax)
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    
    ax = plt.subplot(312)
    ax.set_title('Meridional wind')
    ax.set_ylabel(ylabel)
    
    if vmins is not None: vmin = vmins[1]
    if vmaxs is not None: vmax = vmaxs[1]
    
    im = ax.pcolormesh(num, h, v.T, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    ax =plt.subplot(313)
    ax.set_title('Vertical wind')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('UTC (h)')
    
    if vmins is not None: vmin = vmins[2]
    if vmaxs is not None: vmax = vmaxs[2]
    
    im = ax.pcolormesh(num, h, w.T, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    
    plt.tight_layout()
    plt.savefig(figfile)
    plt.close()

def calculate_spectra(s, x, axis=2):
    '''
    Inputs:
        s    :    array-like. (nz, ny, nx)
        x    :    array-like. (nz, ny, nx)
    '''
    s = np.where(np.isfinite(s), s, 0)
    
    st = np.sum(s, axis=axis)
    mask = np.where(np.isfinite(st))
    
    # if axis==0:
    #     s = s #[:,mask[0], mask[1]]
    # elif axis==1:
    #     s = s[:, :, mask[1]]
    #     x = x[:, :, mask[1]]
    # elif axis==2:
    #     s = s[:, mask[1], :]
    #     x = x[:, mask[1], :]
    # else:
    #     raise ValueError
    
    max_d = np.max(x) - np.min(x)
    
    indices = np.arange(x.shape[axis])
    delta_x = np.abs( 2*np.max( np.take(x, indices[1:], axis=axis) - np.take(x, indices[:-1], axis=axis) ) )
    
    N =  int( 0.5*max_d/delta_x )
    k = 1.0/( delta_x*np.arange(1,N) )
    
    e = np.einsum('f,zyx->zyxf', -2j*np.pi*k, x)
    
    if axis==0:
        F = np.einsum('zyx,zyxf->yxf', s, np.exp(e) )
    elif axis==1:
        F = np.einsum('zyx,zyxf->zxf', s, np.exp(e) )
    elif axis==2:
        F = np.einsum('zyx,zyxf->zyf', s, np.exp(e) )
    else:
        raise ValueError
    
    # F = np.mean( F, axis=axis ) #Get rid of the extra dimension
    
    Fm = np.mean( np.abs(F)**2, axis=(0,1) )
    
    return(Fm, k)
   
def plot_spectra(Fs, k, filename='./test.png',
                 labels=['u', 'v', 'w']):
    
    l = k**(-5/3.)
    l /= np.nanmax(l)
    
    l3 = k**(-3.)
    l3 /= np.nanmax(l3)
    
    plt.figure(figsize=(8,4))
    
    # plt.subplot(211)
    for i,F in enumerate(Fs):
        F /= np.nanmax(F)
        plt.plot(k, F, 'o-', label=labels[i], alpha=0.5)
    
    plt.plot(k, 1e-1*l, 'm-.', label='-5/3')
    plt.plot(k, l3, 'r--', label='-3')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('wavenumber (log k)')
    plt.ylabel('log E(k)')
    plt.legend()
    plt.grid(True)
    
    # plt.subplot(212)
    # plt.plot(k, Fe, 'bo', label='v')
    # # plt.plot(k, l, 'r--')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.grid()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def calc_xyz_mean_winds(PINN_model, times, x_flat, y_flat, z_flat,
                    u_shape, valid_mask):
    
    u = np.zeros(u_shape, dtype=np.float32)
    v = np.zeros(u_shape, dtype=np.float32)
    w = np.zeros(u_shape, dtype=np.float32)
    
    ts = []
    us = []
    vs = []
    ws = []
    
    for time in times:
        print('m', end='')
        
        t = np.zeros_like(x_flat) + time
        
        X_test = tf.stack([t, x_flat, y_flat, z_flat], axis=1)
        
        outputs = PINN_model(X_test)
        
        u_flat = outputs[:,0].numpy()
        v_flat = outputs[:,1].numpy()
        w_flat = outputs[:,2].numpy()
        # P = w #outputs[:,3].numpy()
        
        u[:] = np.nan
        v[:] = np.nan
        w[:] = np.nan
        
        u[valid_mask] = u_flat
        v[valid_mask] = v_flat
        w[valid_mask] = w_flat
        
        u0 = np.nanmean(u, axis=(1,2))
        v0 = np.nanmean(v, axis=(1,2))
        w0 = np.nanmean(w, axis=(1,2))

        ts.append(time)
        us.append(u0)
        vs.append(v0)
        ws.append(w0)
    
    ts = np.array(ts) 
    us = np.array(us) 
    vs = np.array(vs) 
    ws = np.array(ws)    
    
    return(us, vs, ws)

def calc_4h_mean_winds(times, altitudes,
                       u0, v0, w0,
                       time_width=4*60*60, time_sigma=0.5,
                       alt_width=4e3, alt_sigma=0.5):
    '''
    Inputs:
        times         :    array-like. Time vector in seconds from 01/01/1950 (ntimes)
        u0, v0, w0    :    array-like. Mean winds (ntimes, naltitudes)
        width         :    window width in seconds
    '''
    
    #Filter in time
    dt = times[1] - times[0]
    N = int(time_width/dt)
    
    if N < 2:
        return (u0, v0, w0)
    
    M = (N-1)/2
    
    k = np.arange(N)
    
    w = np.exp(-0.5*((k-M)/M*time_sigma)**2)
    w /= np.sum(w)
    
    u4h = convolve1d(u0, w, axis=0, mode='nearest')
    v4h = convolve1d(v0, w, axis=0, mode='nearest')
    w4h = convolve1d(w0, w, axis=0, mode='nearest')
    
    #Filter in altitude
    dt = altitudes[1] - altitudes[0]
    N = int(alt_width/dt)
    
    if N<2:
        return(u4h, v4h, w4h)
    
    M = (N-1)/2
    
    k = np.arange(N)
    
    w = np.exp(-0.5*((k-M)/M*alt_sigma)**2)
    w /= np.sum(w)
    
    u4h = convolve1d(u4h, w, axis=1, mode='nearest')
    v4h = convolve1d(v4h, w, axis=1, mode='nearest')
    w4h = convolve1d(w4h, w, axis=1, mode='nearest')
    
    return(u4h, v4h, w4h)
    
def keograms(filename,
               figpath,
               tstep = 300,
                xstep = 10e3,
                ystep = 10e3,
                x0 = None,
                y0 = None,
                z0 = 90e3,
                vmins = [-100,-100,-15, None],
                vmaxs = [ 100, 100, 15, None],
                ext='png',
                type='residuals',
                plot_mean=False,
                log_file=None,
                time_width=0,
                cmap='jet'
                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    PINN_model, PINN_attrs = PINN.load_model(filename, log_file=log_file)
    
    tmin, xmin, ymin, zmin = PINN_model.lb
    tmax, xmax, ymax, zmax = PINN_model.ub
    
    tmin = np.ceil(tmin)
    xmin = np.ceil(xmin)*0.8
    ymin = np.ceil(ymin)*0.8
    
    x_radius = 1.01*np.abs(xmin) #(xmax - xmin)/2.0
    y_radius = 1.01*np.abs(ymin) #(ymax - ymin)/2.0
    
    x_center = 0 #(xmax + xmin)/2.0
    y_center = 0 #(ymax + ymin)/2.0
    
    times   = np.arange(tmin, tmax, tstep, dtype=np.float32)
    x       = np.arange(xmin, -xmin, xstep, dtype=np.float32)
    y       = np.arange(ymin, -ymin, ystep, dtype=np.float32)
    
    if x0 is None: x0 = 0
    if y0 is None: y0 = 0
    if z0 is None: z0 = 90
    
    x0       = np.atleast_1d(x0)
    y0       = np.atleast_1d(y0)
    z0       = np.atleast_1d(z0)
    
    dt = datetime.datetime.utcfromtimestamp(times[0])
    dt_str = dt.strftime('%Y%m%d')
    
    #X
    figfile_mean = os.path.join(figpath, 'keo_x_%s.%s' %(dt_str, ext) )
    
    T, X, Y, Z = np.meshgrid(times, x, y0, z0, indexing='ij')
    shape = T.shape
    
    t_flat = T.flat
    x_flat = X.flat
    y_flat = Y.flat
    z_flat = Z.flat
    
    X_test = tf.stack([t_flat, x_flat, y_flat, z_flat], axis=1)
    
    outputs = PINN_model(X_test)
    
    u = outputs[:,0].numpy()
    v = outputs[:,1].numpy()
    w = outputs[:,2].numpy()
    # P = w #outputs[:,3].numpy()
    
    u = u.reshape(shape)
    v = v.reshape(shape)
    w = w.reshape(shape)
    
    u = np.mean(u, axis=(2,3))
    v = np.mean(v, axis=(2,3))
    w = np.mean(w, axis=(2,3))
    
    u, v, w = calc_4h_mean_winds(times, x, u, v, w,
                                 time_width=time_width,
                                 alt_width=0)
    
    if type == 'residuals':
        u0, v0, w0 = calc_4h_mean_winds(times, x, u, v, w,
                                 time_width=4*60*60,
                                 alt_width=0)
        
        u -= u0
        v -= v0
        w -= w0
        
    plot_mean_winds(times, x*1e-3, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='East-West (km)')

    #Y
    figfile_mean = os.path.join(figpath, 'keo_y_%s.%s' %(dt_str, ext) )
    
    T, X, Y, Z = np.meshgrid(times, x0, y, z0, indexing='ij')
    shape = T.shape
    
    t_flat = T.flat
    y_flat = Y.flat
    z_flat = Z.flat
    x_flat = X.flat
    
    X_test = tf.stack([t_flat, x_flat, y_flat, z_flat], axis=1)
    
    outputs = PINN_model(X_test)
    
    u = outputs[:,0].numpy()
    v = outputs[:,1].numpy()
    w = outputs[:,2].numpy()
    # P = w #outputs[:,3].numpy()
    
    u = u.reshape(shape)
    v = v.reshape(shape)
    w = w.reshape(shape)
    
    u = np.mean(u, axis=(1,3))
    v = np.mean(v, axis=(1,3))
    w = np.mean(w, axis=(1,3))
    
    u, v, w = calc_4h_mean_winds(times, y, u, v, w,
                                 time_width=time_width,
                                 alt_width=0)
    
    if type == 'residuals':
        u0, v0, w0 = calc_4h_mean_winds(times, x, u, v, w,
                                 time_width=4*60*60,
                                 alt_width=0)
        
        u -= u0
        v -= v0
        w -= w0
        
    plot_mean_winds(times, y*1e-3, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='North-South (km)')

def getting_and_plotting_winds(filename,
                               figpath,
                               tstep = 300,
                                xstep = 10e3,
                                ystep = 10e3,
                                zstep = 2e3,
                                xmin = None,
                                ymin = None,
                                zmin = None,
                                vmins = [-100,-100,-15, None],
                                vmaxs = [ 100, 100, 15, None],
                                ext='png',
                                type='residuals',
                                plot_mean=False,
                                log_file=None,
                                plot_vor_div=False,
                                plot_wind_vectors=False,
                                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    PINN_model, PINN_attrs = load_model(filename, log_file=log_file)
    
    tmin0, xmin0, ymin0, zmin0 = PINN_model.lb
    tmax0, xmax0, ymax0, zmax0 = PINN_model.ub
    
    tmin = np.ceil(tmin0)
    
    if xmin is None: xmin = np.ceil(xmin0)
    else: xmin = max(xmin, xmin0)
    
    if ymin is None: ymin = np.ceil(ymin0)
    else: ymin = max(ymin, ymin0)
    
    if zmin is None: zmin = np.ceil(zmin0)
    zmin = max(zmin, zmin0)
    
    times   = np.arange(tmin, tmax0, tstep)
    x       = np.arange(xmin, -xmin, xstep)
    y       = np.arange(ymin, -ymin, ystep)
    z       = np.arange(zmin, zmax0, zstep)
    
    times = times.astype(np.float32)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    
    Z, Y, X = np.meshgrid(z,y,x, indexing='ij')
    shape = X.shape 
    
    #3D to 1D
    x_radius = 1.01*np.abs(xmin) #(xmax - xmin)/2.0
    y_radius = 1.01*np.abs(ymin) #(ymax - ymin)/2.0
    
    x_center = 0 #(xmax + xmin)/2.0
    y_center = 0 #(ymax + ymin)/2.0
    mask = np.where( ( (X-x_center)**2/x_radius**2 + (Y-y_center)**2/y_radius**2) <= 1 )
    
    x_flat = X[mask]
    y_flat = Y[mask]
    z_flat = Z[mask]
    
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = dt.strftime('%Y%m%d')
    
    figfile_mean = os.path.join(figpath, 'mean_wind_%s.%s' %(dt_str, ext) ) 
    figfile_4h_mean = os.path.join(figpath, 'mean_4h_wind_%s.%s' %(dt_str, ext) ) 
    figfile_res_mean = os.path.join(figpath, 'mean_res_wind_%s.%s' %(dt_str, ext) )   
    
    if (type == 'residuals') or plot_mean:
        u0,   v0,  w0 = calc_xyz_mean_winds(PINN_model, times, x_flat, y_flat, z_flat, X.shape, mask)
        u4h, v4h, w4h = calc_4h_mean_winds(times, z, u0, v0, w0)
        
        ur = u0 - u4h
        vr = v0 - v4h
        wr = w0 - w4h
        
        if plot_mean:
            plot_mean_winds(times, z*1e-3, u0, v0, w0,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs)
            
            plot_mean_winds(times, z*1e-3, u4h, v4h, w4h,
                            figfile=figfile_4h_mean,
                            vmins=vmins, vmaxs=vmaxs)
            
            plot_mean_winds(times, z*1e-3, ur, vr, wr,
                            figfile=figfile_res_mean,
                            vmins=vmins, vmaxs=vmaxs)
            
            return
    
    # Fu = 0
    # Fv = 0
    # Fw = 0
    
    figpath_3D = os.path.join(figpath, 'wind_field_%s' %dt_str)
    if not os.path.exists(figpath_3D):
        os.mkdir(figpath_3D)
    
    h5file = os.path.join(figpath, 'wind_field_%s.h5' %dt_str)
    
    f = h5py.File(h5file, 'w')
    
    f.create_dataset('x', data=X, compression="gzip")
    f.create_dataset('y', data=Y, compression="gzip")
    f.create_dataset('z', data=Z, compression="gzip")
    
    N = len(times)
    
    for idx, time in enumerate(times):
        print('.', end='')
        
        t = np.zeros_like(x_flat) + time
        dt = datetime.datetime.utcfromtimestamp(time)
        
        X_test = tf.stack([t, x_flat, y_flat, z_flat], axis=1)
        
        # outputs = PINN_model(X_test)
        #
        # u_flat = outputs[:,0].numpy()
        # v_flat = outputs[:,1].numpy()
        # w_flat = outputs[:,2].numpy()
        # # P = w #outputs[:,3].numpy()
        
        outputs = PINN_model.predict_grads(X_test)
        
        u_flat = outputs[:,0].numpy()
        v_flat = outputs[:,1].numpy()
        w_flat = outputs[:,2].numpy()
        
        u_x_flat = outputs[:,3].numpy()
        u_y_flat = outputs[:,4].numpy()
        u_z_flat = outputs[:,5].numpy()
        
        v_x_flat = outputs[:,6].numpy()
        v_y_flat = outputs[:,7].numpy()
        v_z_flat = outputs[:,8].numpy()
        
        w_x_flat = outputs[:,9].numpy()
        w_y_flat = outputs[:,10].numpy()
        w_z_flat = outputs[:,11].numpy()
        
        div_flat = u_x_flat + v_y_flat
        vort_flat = v_x_flat - u_y_flat
        
        u = np.zeros_like(X) + np.nan
        v = np.zeros_like(X) + np.nan
        w = np.zeros_like(X) + np.nan
        
        div  = np.zeros_like(X) + np.nan
        vort = np.zeros_like(X) + np.nan
        
        u[mask] = u_flat
        v[mask] = v_flat
        w[mask] = w_flat
        
        div[mask] = div_flat
        vort[mask] = vort_flat
        
        plot_field(x*1e-3, y*1e-3, z*1e-3,
                   fields = [u, v, w],
                   titles=['u','v','w', 'P'],
                   figtitle='%s'%dt.strftime('%Y/%m/%d %H:%M:%S'),
                   vmins = vmins,
                   vmaxs = vmaxs,
                   prefix='field_%s'%dt.strftime('%Y%m%d_%H%M%S'),
                   path=figpath_3D)
        
        g = f.create_group('%d' %time)
        g.create_dataset('u', data=u, compression="gzip")
        g.create_dataset('v', data=v, compression="gzip")
        g.create_dataset('w', data=w, compression="gzip")
        
        if type == 'residuals':
            u -= u4h[idx,:,None,None]
            v -= v4h[idx,:,None,None]
            w -= w4h[idx,:,None,None]
            
        if plot_vor_div:
            figfile = os.path.join(figpath_3D, 'vordiv_map_%s.%s' %(dt.strftime('%Y%m%d_%H%M%S'), ext) )
            
            plot_3d_maps(X*1e-3, Y*1e-3, z*1e-3,
                         [vort,div,w],
                         title=r'%2.1f$^\circ$N, %2.1f$^\circ$E: %s' %(PINN_attrs['lat_center'],
                                                        PINN_attrs['lon_center'],
                                                        dt.strftime('%Y.%m.%d %H:%M:%S') ),
                         title_list=['vort','div','w'],
                         vmin_list=[-2e-2, -1e-2, -5],
                         vmax_list=[ 2e-2,  1e-2,  5],
                         filename=figfile,
                         zdecimator=2)
        
        if plot_wind_vectors:
            figfile = os.path.join(figpath_3D, 'wind_vec_%s.%s' %(dt.strftime('%Y%m%d_%H%M%S'), ext) )
            
            plot_3d_vectors(X*1e-3, Y*1e-3, z*1e-3,
                         [u,v,w],
                         title=r'%2.1f$^\circ$N, %2.1f$^\circ$E: %s' %(PINN_attrs['lat_center'],
                                                        PINN_attrs['lon_center'],
                                                        dt.strftime('%Y.%m.%d %H:%M:%S') ),
                         title_list=['u','v','w'],
                         vmin_list=vmins,
                         vmax_list=vmaxs,
                         filename=figfile,
                         scale=3000*vmaxs[0]/150)
        
        # F, k = calculate_spectra(u, x=X, axis=2)
        # Fu = Fu + F
        #
        # F, k = calculate_spectra(v, x=X, axis=2)
        # Fv = Fv + F
        #
        # F, k = calculate_spectra(w, x=X, axis=2)
        # Fw = Fw + F
    
    f.close()

    # spec_figname = os.path.join(figpath, 'spectra_%s.png' %dt_str)
    # plot_spectra( [Fu/N, Fv/N, Fw/N], k, filename=spec_figname)
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Germany", help='Data path')
    parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Virtual/ICON_20160816", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Argentina/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Norway/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Condor/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Piura/", help='Data path')
    parser.add_argument('-m', '--model', dest='model', default=None, help='neural network model')
    parser.add_argument('-s', '--subfolder', dest='subfolder', default="fPINN_fc2.3", help='subfolder where the neural network model is stored')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='figures extension')
    parser.add_argument('-t', '--type', dest='type', default='full', help='plot type. Either "residuals" or "full" wind')
    parser.add_argument('-l', '--log-file', dest='log_file', default=11000, help='select the i-th weights file from the log folder')
    
    
    args = parser.parse_args()
    
    path  = args.dpath
    model_name = args.model
    ext = args.ext
    type = args.type
    log_file = args.log_file
    
    vmins = [-50, -50, -1]
    vmaxs = [ 50,  50,  1]
                
    path_PINN = os.path.join(path, args.subfolder)
    
    if model_name is None:
        models = glob.glob1(path_PINN, 'model*.h5')
        models = sorted(models)
    else:
        models = [  model_name ]
    
    for model in models:
        id_name = model[-11:-3]
        
        filename = os.path.join(path_PINN, model)
        figpath = os.path.join(path_PINN, 'plot_%s_%s_%s' %(type, ext, log_file) ) #+os.path.splitext(model_name)[0])
        
        getting_and_plotting_winds(filename, figpath, ext=ext, tstep=600,
                                    type=type,
                                    # plot_mean=True,
                                    # xmin=-30e3,
                                    # ymin=-30e3,
                                    zstep=1e3,
                                    # zmin=84e3,
                                    log_file=log_file,
                                    vmins=vmins,
                                    vmaxs=vmaxs,
                                   )
        
        keograms(filename, figpath, ext=ext,
                 xstep=10e3,
                 ystep=10e3,
                 tstep=300,
                 # z=np.arange(89e3,91.5e3,0.5e3), #Vortex
                 x0 = np.arange(-50e3,50.1e3,2e3),
                 y0 = np.arange(-50e3,50.1e3,2e3),
                 z0 = np.arange(91e3,94.5e3,0.5e3), #Tonga
                 time_width=0*60,
                 vmins=vmins,
                 vmaxs=vmaxs,
                                    type=type,
                                    log_file=log_file,
                                    cmap='seismic'
                                   )
        