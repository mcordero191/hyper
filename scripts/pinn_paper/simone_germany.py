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
from radar.specular_meteor_radars.Hari import read_H5File
DTYPE='float32'

import matplotlib.pyplot as plt
# plt.rcParams['axes.facecolor'] = "#B0B0B0"

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.dates as mdates

from atmospheric_models.ICON import ICONReader
from georeference.geo_coordinates import lla2enu, lla2xyh

from pinn import pinn_v2 as pinn
from radar.specular_meteor_radars.SMR import SMRReader

from utils.plotting import plot_mean_winds, plot_3d_vectors
    
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
               xlabel='Longitude',
               ylabel='Latitude'
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
        
        filename = os.path.join(path, "wind_field_%2.1f_%s.png" %(alts[zi], prefix) )
        
        fig = plt.figure(figsize=(5*nfields,5*nrows))
        plt.suptitle(figtitle)
        
        if df_sampling is not None:
            df_sampling_z = df_sampling[ np.abs(df_sampling['heights'] - alts[zi]) < 1.2 ]
        
            samp_lons = df_sampling_z['lons'].values
            samp_lats = df_sampling_z['lats'].values
        
        for iplot in range(nfields):
            ax = fig.add_subplot(nrows, nfields, iplot+1)#, projection='3d')
        
            f = fields[iplot]
            vmin = vmins[iplot]
            vmax = vmaxs[iplot]
            
            if vmin is None: vmin = np.min(f[zi])
            if vmax is None: vmax = np.max(f[zi])
            
            im = ax.pcolormesh(lons, lats, f[zi],
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax)
            
            # im = ax.plot_surface(LON, LAT, f[zi],
            #                      cmap=cmap,
            #                      vmin=vmin,
            #                      vmax=vmax,
            #                      alpha=alpha)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('%s %2.1fkm' %(titles[iplot], alts[zi]) )
            
            # ax.set_zlim(vmin, vmax)
            
            # ax.plot(samp_lons, samp_lats, 'mx')
            if df_sampling is not None:
                ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x')
            
            # ax.set_xlim(-2,2)
            # ax.set_ylim(0,2)
            
            plt.colorbar(im, ax=ax)
            
            if nrows>1:
                ax = fig.add_subplot(nrows, nfields, nfields+iplot+1)#, projection='3d')
        
                f = fields_est[iplot]
                #
                im = ax.pcolormesh(lons, lats, f[zi],
                                  cmap=cmap,
                                  vmin=vmin,
                                  vmax=vmax)
                
                # im = ax.plot_surface(LON, LAT, f[zi],
                #                  cmap=cmap,
                #                  vmin=vmin,
                #                  vmax=vmax,
                #                  alpha=alpha)
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title('%s_est %2.1f km' %(titles[iplot], alts[zi]) )
                
                # ax.set_zlim(vmin,vmax)
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

def plot_heatmaps(lons, lats, alts,
               fields,
               prefix='',
               path='',
               figtitle='',
               cmap='RdBu_r',
               titles=None,
               vmins=None,
               vmaxs=None,
               df_sampling=None,
               alpha=1,
               xlabel='Longitude',
               ylabel='Latitude'
               ):
    
    nfields = len(fields)
    nz, ny, nx = fields[0].shape
    
    if titles is None: titles = ['']*nfields
    if vmins is None: vmins = [None]*nfields
    if vmaxs is None: vmaxs = [None]*nfields
    
    nrows = int(  np.sqrt(nz) )
    ncols = int( nz / nrows +0.9) 
        
    LON, LAT = np.meshgrid(lons, lats)
    
    for iplot in range(nfields):
        
        f = fields[iplot]
        vmin = vmins[iplot]
        vmax = vmaxs[iplot]
        
        if vmin is None: vmin = np.min(f)
        if vmax is None: vmax = np.max(f)
            
        fig, axs = plt.subplots(nrows, ncols, figsize=(5.5*ncols,5*nrows))
        plt.suptitle(figtitle)
            
        filename = os.path.join(path, "%s_%s.png" %(titles[iplot][:3], prefix ))
        
        axs_flat = axs.flat[::-1]
        
        for zi in range(nz):
            
            ax = axs_flat[zi]
            
            im = ax.pcolormesh(lons, lats, f[zi],
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax)
            
            # im = ax.contourf(lons, lats, f[zi],
            #                  levels=10,
            #                   cmap=cmap,
            #                   vmin=vmin,
            #                   vmax=vmax)
            
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('%s %2.1f km' %(titles[iplot], alts[zi]) )
            
            plt.colorbar(im, ax=ax)
            
            # ax.scatter(16.04, 69.3, marker='o', c='r')
                
            
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


def plot_mean_winds_grads(t, h, ug,
                        figfile='./test.png',
                        vmins=None,
                        vmaxs=None,
                        cmap='seismic',
                        xlabel='UTC (h)',
                        ylabel='Altitude (km)',
                        titles=[
                                r'$u_0$ (m/s)', r'$v_0$ (m/s)', r'$w_0$ (m/s)',
                                r'$u_x$ (m/s/km)', r'$v_x$ (m/s/km)', r'$w_x$ (m/s/km)', 
                                r'$u_y$ (m/s/km)', r'$v_y$ (m/s/km)', r'$w_y$ (m/s/km)', 
                                r'$u_z$ (m/s/km)', r'$v_z$ (m/s/km)', r'$w_z$ (m/s/km)',
                                ]
                        ):
    '''
    ug    :    (
    '''
    vmin = None
    vmax = None
    
    num = mdates.epoch2num(t)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    formatter = mdates.ConciseDateFormatter(locator)
    
    fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(16,8))
    
    axs_flat = axs.flat
    
    for i in range(12):
        
        ax = axs_flat[i]
        ax.set_title(titles[i])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
        if vmins is not None: vmin = vmins[i]
        if vmaxs is not None: vmax = vmaxs[i]
        
        d = ug[i].T
        if i > 2: d = 1e3*d
        
        im = ax.pcolormesh(num, h, d, cmap=cmap, vmin=vmin, vmax=vmax)
        
        plt.colorbar(im, ax=ax)
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle='--')
        
    
    for ax in axs_flat:
        ax.label_outer()
    
    plt.tight_layout(pad=0)
    
    ax.annotate('@mu',
                     xy = (1.03, -0.2),
                     xycoords='axes fraction',
                     ha='left',
                     va="top",
                     fontsize=8,
                     weight='bold')
    
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

def calc_xyz_mean_winds(nn, times, x_flat, y_flat, z_flat,
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
        
        nn_output = nn.infer(t, x_flat, y_flat, z_flat)
        
        u[:] = np.nan
        v[:] = np.nan
        w[:] = np.nan
        
        u[valid_mask] = nn_output[:,0]
        v[valid_mask] = nn_output[:,1]
        w[valid_mask] = nn_output[:,2]
        
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

def calc_mean_winds(x,
                    y,
                    *u,
                    x_width=4*60*60,
                    x_sigma=0.5,
                    y_width=4,
                    y_sigma=0.5,
                    x_axis=0,
                    y_axis=1):
    '''
    Inputs:
        times         :    array-like. Time vector in seconds from 01/01/1950 (ntimes)
        *u = [u0, v0, w0]    :    array-like. Mean winds (ntimes, naltitudes)
        width         :    window width in seconds
    '''
    
    u = np.squeeze(u)
    
    if len(u.shape) != 3:
        raise ValueError("The array must have at least three dimensions with more than one element, %s" %u.shape)
    
    #Filter in time
    dx = x[1] - x[0]
    N = int(x_width/dx)
    
    
    if N < 2:
        return(u)
    
    M = (N-1)/2
    
    k = np.arange(N)
    w = np.exp(-0.5*((k-M)/M*x_sigma)**2)
    
    # w = np.ones(N)
    w /= np.sum(w)
    
    u4h = np.empty_like(u)
    
    for i in range(u.shape[0]):
        u4h[i] = convolve1d(u[i], w, axis=x_axis, mode='nearest')
    
    # v4h = convolve1d(v0, w, axis=x_axis, mode='nearest')
    # w4h = convolve1d(w0, w, axis=x_axis, mode='nearest')
    
    #Filter in altitude
    dy = y[1] - y[0]
    N = int(y_width/dy)
    
    if N<2:
        return(u4h)
    
    M = (N-1)/2
    
    k = np.arange(N)
    
    w = np.exp(-0.5*((k-M)/M*y_sigma)**2)
    
    # w = np.ones(N)
    w /= np.sum(w)
    
    for i in range(u.shape[0]):
        u4h[i] = convolve1d(u4h[i], w, axis=y_axis, mode='nearest')
        
    # u4h = convolve1d(u4h, w, axis=y_axis, mode='nearest')
    # v4h = convolve1d(v4h, w, axis=y_axis, mode='nearest')
    # w4h = convolve1d(w4h, w, axis=y_axis, mode='nearest')
    
    return(u4h)
    
def keograms(filename,
               figpath,
               tstep = None, #min
                xstep = 0.2, #degrees
                ystep = 0.2, #degrees
                zstep = 0.5,
                x0 = None, #degrees
                y0 = None,
                z0 = None, #km
                xrange = 4, #degrees
                yrange = 4, #degrees
                zrange = None, #km
                vmins = [-100,-100,-15, None],
                vmaxs = [ 100, 100, 15, None],
                ext='png',
                type='residuals',
                plot_mean=False,
                log_file=None,
                time_width=0*60*60 ,
                cmap='jet',
                t0=0*60*60,
                trange=None,#3*60*60,
                df_ref=None,
                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    tmin = nn.lb[0].numpy() + t0
    if trange is None: tmax = nn.ub[0].numpy() #- t0
    else: tmax = tmin+trange
    
    if tstep is None: tstep = (tmax - tmin)/(24*6)
    if zrange is None: zrange = (nn.ub[3].numpy() - nn.lb[3].numpy())*1e-3
    
    lon_ref = nn.lon_ref
    lat_ref = nn.lat_ref
    alt_ref = nn.alt_ref
    
    xmin = lon_ref - xrange/2.
    xmax = lon_ref + xrange/2.
    
    ymin = lat_ref - yrange/2.
    ymax = lat_ref + yrange/2.
    
    zmin = alt_ref - zrange/2.
    zmax = alt_ref + zrange/2.
    
    times   = np.arange(tmin, tmax, tstep)
    x       = np.arange(xmin, xmax, xstep)
    y       = np.arange(ymin, ymax, ystep)
    z       = np.arange(zmin, zmax, zstep)
    
    if x0 is None: x0 = lon_ref
    if y0 is None: y0 = lat_ref
    if z0 is None: z0 = alt_ref
    
    x0       = np.atleast_1d(x0)
    y0       = np.atleast_1d(y0)
    z0       = np.atleast_1d(z0)
    
    dt = datetime.datetime.utcfromtimestamp(times[0])
    dt_str = dt.strftime('%Y%m%d_%H%M%S')
    
    #X
    figfile_mean = os.path.join(figpath, 'keo_x_%s_%2.1f_%2.1f.%s' %(dt_str, y0[0], z0[0], ext) )
    
    T, X, Y, Z = np.meshgrid(times, x, y0, z0, indexing='ij')
    shape = T.shape
    
    t_flat = T.flatten()
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    nn_output = nn.infer(t_flat, x_flat, y_flat, z_flat)
    
    u = nn_output[:,0]
    v = nn_output[:,1]
    w = nn_output[:,2]
    
    u = u.reshape(shape)
    v = v.reshape(shape)
    w = w.reshape(shape)
    
    u = np.mean(u, axis=(2,3))
    v = np.mean(v, axis=(2,3))
    w = np.mean(w, axis=(2,3))
    
    u, v, w = calc_mean_winds(times, x, u, v, w,
                                 x_width=time_width,
                                 y_width=0)
    
    if type == 'residuals':
        u0, v0, w0 = calc_mean_winds(times, x, u, v, w,
                                 x_width=4*60*60,
                                 y_width=0 )
        
        u -= u0
        v -= v0
        w -= w0
        
    plot_mean_winds(times, x, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='Longitude')

    #Y
    figfile_mean = os.path.join(figpath, 'keo_y_%s_%2.1f_%2.1f.%s' %(dt_str, x0[0], z0[0], ext) )
    
    T, X, Y, Z = np.meshgrid(times, x0, y, z0, indexing='ij')
    shape = T.shape
    
    t_flat = T.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    x_flat = X.flatten()
    
    outputs = nn.infer(t_flat, x_flat, y_flat, z_flat)
    
    u = outputs[:,0]
    v = outputs[:,1]
    w = outputs[:,2]
        
    u = u.reshape(shape)
    v = v.reshape(shape)
    w = w.reshape(shape)
    
    u = np.mean(u, axis=(1,3))
    v = np.mean(v, axis=(1,3))
    w = np.mean(w, axis=(1,3))
    
    u, v, w = calc_mean_winds(times, y, u, v, w,
                                 x_width=time_width,
                                 y_width=0)
    
    if type == 'residuals':
        u0, v0, w0 = calc_mean_winds(times, y, u, v, w,
                                 x_width=4*60*60,
                                 y_width=0)
        
        u -= u0
        v -= v0
        w -= w0
        
    plot_mean_winds(times, y, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='Latitude')
    
    #Z
    figfile_mean = os.path.join(figpath, 'keo_z_%s_%2.1f_%2.1f.%s' %(dt_str, x0[0], y0[0], ext) )
    
    T, X, Y, Z = np.meshgrid(times, x0, y0, z, indexing='ij')
    shape = T.shape
    
    t_flat = T.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    x_flat = X.flatten()
    
    outputs = nn.infer(t_flat, x_flat, y_flat, z_flat)
    
    u = outputs[:,0]
    v = outputs[:,1]
    w = outputs[:,2]
        
    u = u.reshape(shape)
    v = v.reshape(shape)
    w = w.reshape(shape)
    
    u = np.mean(u, axis=(1,2))
    v = np.mean(v, axis=(1,2))
    w = np.mean(w, axis=(1,2))
    
    u, v, w = calc_mean_winds(times, z, u, v, w,
                                 x_width=time_width,
                                 y_width=0)
    
    if type == 'residuals':
        u0, v0, w0 = calc_mean_winds(times, z, u, v, w,
                                 x_width=4*60*60,
                                 y_width=0)
        
        u -= u0
        v -= v0
        w -= w0
        
    plot_mean_winds(times, z, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='Altitude (km)',
                            df_ref=df_ref,
                            )
    
def getting_and_plotting_winds(filename,
                               figpath,
                               tstep = None, #min
                                xstep = 0.1, #degrees
                                ystep = 0.1, #degrees
                                zstep = 2, #km
                                xrange = 8, #degrees
                                yrange = 3, #degrees
                                zrange = None, #km
                                zmin=None,
                                vmins = [-100,-100,-10, None],
                                vmaxs = [ 100, 100, 10, None],
                                ext='png',
                                type='full',
                                plot_mean=False,
                                log_file=None,
                                plot_fields=False,
                                plot_vor_div=False,
                                plot_wind_vectors=True,
                                t0=0,
                                calc_grads=False,
                                df_meteor=None,
                                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    if plot_vor_div:
        calc_grads = True
    
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    tmin = nn.lb[0].numpy() + t0
    tmax = nn.ub[0].numpy() 
    
    if tstep is None: tstep = np.round( (tmax - tmin)/(24*4) )
    if zrange is None: zrange = (nn.ub[3].numpy() - nn.lb[3].numpy())*1e-3
    
    lon_ref = nn.lon_ref
    lat_ref = nn.lat_ref
    alt_ref = nn.alt_ref
    
    xmin = lon_ref - xrange/2.
    xmax = lon_ref + xrange/2.
    
    ymin = lat_ref - yrange/2.
    ymax = lat_ref + yrange/2.
    
    if zmin is None: zmin = alt_ref - zrange/2.
    zmax = zmin + zrange
    
    # zmin = int(zmin)
    
    times   = np.arange(tmin, tmax, tstep)
    x       = np.arange(xmin, xmax, xstep)
    y       = np.arange(ymin, ymax, ystep)
    z       = np.arange(zmin, zmax, zstep)
    
    Z, Y, X = np.meshgrid(z,y,x, indexing='ij')
    shape = X.shape 
    
    #Rotate X and Y
    # theta = np.pi/4
    # Xmag = (X-lon_ref)/(xrange/2)
    # Ymag = (Y-lat_ref)/(yrange/2)
    #
    # Xp = Xmag*np.cos(theta) - Ymag*np.sin(theta)
    # Yp = Xmag*np.sin(theta) + Ymag*np.cos(theta)
    #
    # Xp = Xp*xrange/2 + lon_ref
    # Yp = Yp*yrange/2 + lat_ref
    
    mask = np.where( ( (X-lon_ref)**2/(xrange)**2 + (Y-lat_ref)**2/(yrange)**2) <= 1 )
    # mask = np.where( ( (X-lon_ref)**2/(xrange/2)**2 + (Y-lat_ref)**2/(yrange/2)**2) >=0.1 )
    
    x_flat = X[mask]
    y_flat = Y[mask]
    z_flat = Z[mask]
    
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = dt.strftime('%Y%m%d')
    
    figfile_mean = os.path.join(figpath, 'mean_wind_%s.%s' %(dt_str, ext) ) 
    figfile_4h_mean = os.path.join(figpath, 'mean_4h_wind_%s.%s' %(dt_str, ext) ) 
    figfile_res_mean = os.path.join(figpath, 'mean_res_wind_%s.%s' %(dt_str, ext) )   
    
    if (type == 'residuals') or plot_mean:
        u0,   v0,  w0 = calc_xyz_mean_winds(nn, times, x_flat, y_flat, z_flat, X.shape, mask)
        u4h, v4h, w4h = calc_mean_winds(times, z, u0, v0, w0,
                                        x_width=4*60*60, y_width=4)
        
        ur = u0 - u4h
        vr = v0 - v4h
        wr = w0 - w4h
        
        if plot_mean:
            
            h5file = os.path.join(figpath, 'mean_wind_%s.h5' %dt_str)
            
            with h5py.File(h5file, 'w') as fp:
    
                fp.create_dataset('epoch', data=times)
                fp.create_dataset('altitudes', data=z)
                fp.create_dataset('u0', data=u0)
                fp.create_dataset('v0', data=v0)
                fp.create_dataset('w0', data=w0)
                
            plot_mean_winds(times, z, u0, v0, w0,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs)
            
            plot_mean_winds(times, z, u4h, v4h, w4h,
                            figfile=figfile_4h_mean,
                            vmins=vmins, vmaxs=vmaxs)
            
            plot_mean_winds(times, z, ur, vr, wr,
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
    
    with h5py.File(h5file, 'w') as fp:
    
        fp.create_dataset('x', data=X)
        fp.create_dataset('y', data=Y)
        fp.create_dataset('z', data=Z)
        
        N = len(times)
        
        for idx, time in enumerate(times):
            print('.', end='')
                
            t = np.zeros_like(x_flat) + time
            dt = datetime.datetime.utcfromtimestamp(time)
            
            # outputs = nn(X_test)
            #
            # u_flat = outputs[:,0]
            # v_flat = outputs[:,1]
            # w_flat = outputs[:,2]
            # # P = w #outputs[:,3]
            
            if calc_grads:
                outputs = nn.infer_gradients(t, x_flat, y_flat, z_flat)
                
                u_x_flat = outputs[:,3]
                u_y_flat = outputs[:,4]
                u_z_flat = outputs[:,5]
                
                v_x_flat = outputs[:,6]
                v_y_flat = outputs[:,7]
                v_z_flat = outputs[:,8]
                
                w_x_flat = outputs[:,9]
                w_y_flat = outputs[:,10]
                w_z_flat = outputs[:,11]
                
                u_x = np.zeros_like(X) + np.nan
                v_x = np.zeros_like(X) + np.nan
                w_x = np.zeros_like(X) + np.nan
                
                u_y = np.zeros_like(X) + np.nan
                v_y = np.zeros_like(X) + np.nan
                w_y = np.zeros_like(X) + np.nan
                
                u_z = np.zeros_like(X) + np.nan
                v_z = np.zeros_like(X) + np.nan
                w_z = np.zeros_like(X) + np.nan
                
                u_x[mask] = u_x_flat*1e3 #from m/s/m to m/s/km
                v_x[mask] = v_x_flat*1e3
                w_x[mask] = w_x_flat*1e3
                
                u_y[mask] = u_y_flat*1e3
                v_y[mask] = v_y_flat*1e3
                w_y[mask] = w_y_flat*1e3
                
                u_z[mask] = u_z_flat*1e3
                v_z[mask] = v_z_flat*1e3
                w_z[mask] = w_z_flat*1e3
                
                div  = u_x + v_y #+ w_z
                vort = v_x - u_y
                
            else:
                outputs = nn.infer(t, x_flat, y_flat, z_flat)
            
            u_flat = outputs[:,0]
            v_flat = outputs[:,1]
            w_flat = outputs[:,2]
            
            u = np.zeros_like(X) + np.nan
            v = np.zeros_like(X) + np.nan
            w = np.zeros_like(X) + np.nan
            
            u[mask] = u_flat
            v[mask] = v_flat
            w[mask] = w_flat
            
            
            g = fp.create_group('%d' %time)
            g.create_dataset('u', data=u)
            g.create_dataset('v', data=v)
            g.create_dataset('w', data=w)
            
            if calc_grads:
                g.create_dataset('u_x', data=u_x)
                g.create_dataset('v_x', data=v_x)
                g.create_dataset('w_x', data=w_x)
                
                g.create_dataset('u_y', data=u_y)
                g.create_dataset('v_y', data=v_y)
                g.create_dataset('w_y', data=w_y)
                
                g.create_dataset('u_z', data=u_z)
                g.create_dataset('v_z', data=v_z)
                g.create_dataset('w_z', data=w_z)
                
                
            if type == 'residuals':
                u -= u4h[idx,:,None,None]
                v -= v4h[idx,:,None,None]
                w -= w4h[idx,:,None,None]
            
            if plot_fields:
                plot_field(x, y, z,
                           fields = [u, v, w],
                           titles=['u','v','w', 'P'],
                           figtitle='%s'%dt.strftime('%Y/%m/%d %H:%M:%S'),
                           vmins = vmins,
                           vmaxs = vmaxs,
                           prefix='field_%s'%dt.strftime('%Y%m%d_%H%M%S'),
                           path=figpath_3D)
            
                
            if plot_vor_div:
                # figfile = os.path.join(figpath_3D, 'vordiv_map_%s.%s' %(dt.strftime('%Y%m%d_%H%M%S'), ext) )
                #
                # plot_3d_maps(X, Y, z,
                #              [vort, div, w_z],
                #              title=r'%2.1f$^\circ$N, %2.1f$^\circ$E: %s' %(lat_ref,
                #                                             lon_ref,
                #                                             dt.strftime('%Y.%m.%d %H:%M:%S') ),
                #              title_list=['Abs. rel. vort. (m/s/km)','hor. div. (m/s/km)','w_z (m/s/km)'],
                #              vmin_list=[0, -20, -15],
                #              vmax_list=[ 15,  20,  15],
                #              cmap_list=['jet', 'seismic_r', 'seismic'],
                #              filename=figfile,
                #              zdecimator=1)
                
                # div[0,:,:] = np.nanmean(div, axis=0)
                
                plot_heatmaps(x, y, z,
                           fields = [vort, div],#, w_z],
                           titles=['Relative vorticity (m/s/km)','Horizontal Divergence (m/s/km)', r'$w_z$ (m/s/km)'],
                           figtitle='%s'%dt.strftime('%Y/%m/%d %H:%M:%S'),
                           vmins = [-5, -5, -5],
                           vmaxs = [ 5,  5,  5],
                           prefix='%s'%dt.strftime('%Y%m%d_%H%M%S'),
                           path=figpath_3D)
                
            
            if plot_wind_vectors:
                
                df_meteor_t = None
                if df_meteor is not None:
                    mask1 = np.abs(df_meteor['times'] - time) < 5*60
                    df_meteor_t = df_meteor[mask1]
                
                
                figfile = os.path.join(figpath_3D, 'wind_vec_%s.%s' %(dt.strftime('%Y%m%d_%H%M%S'), ext) )
                
                # u[0,:,:] = np.nanmean(u, axis=0)
                # v[0,:,:] = np.nanmean(v, axis=0)
                # w[0,:,:] = np.nanmean(w, axis=0)
    
                plot_3d_vectors(X, Y, z,
                             [u,v,w],
                             title=r'%s' %( dt.strftime('%Y.%m.%d %H:%M:%S') ),
                             title_list=['u','v','w'],
                             vmin_list=vmins,
                             vmax_list=vmaxs,
                             filename=figfile,
                             scale=3000*vmaxs[0]/300,
                             df_meteor=df_meteor_t)
                
                # figfile = os.path.join(figpath_3D, 'h_wind_vec_%s.%s' %(dt.strftime('%Y%m%d_%H%M%S'), ext) )
                #
                # plot_3d_vectors_hprofile(X, Z, y,
                #              [u,v,w],
                #              title=r'%s' %( dt.strftime('%Y.%m.%d %H:%M:%S') ),
                #              title_list=['u','v','w'],
                #              vmin_list=vmins,
                #              vmax_list=vmaxs,
                #              filename=figfile,
                #              scale=3000*vmaxs[0]/200)
                
            # F, k = calculate_spectra(u, x=X, axis=2)
            # Fu = Fu + F
            #
            # F, k = calculate_spectra(v, x=X, axis=2)
            # Fv = Fv + F
            #
            # F, k = calculate_spectra(w, x=X, axis=2)
            # Fw = Fw + F

    # spec_figname = os.path.join(figpath, 'spectra_%s.png' %dt_str)
    # plot_spectra( [Fu/N, Fv/N, Fw/N], k, filename=spec_figname)

def plot_alt_profiles(datetimes, lons, lats, alts, u, v, w, figpath='./'):
    
    nlons = len(lons)
    nlats = len(lats)
    
    dt = datetimes[0]
    
    figname = os.path.join(figpath, 'wind_alt_profile_%s_%2.1f_%2.1f.png' %( dt.strftime("%Y%m%d_%H%M%S"), lons[0], lats[0]) )
    
    fig, axs = plt.subplots(nlons, nlats, figsize=(nlats*7, nlons*7), squeeze=False)
    
    newline = "\n"
    fig.suptitle(r'SIMONe Norway: %s @ %2.1f$^\circ$N, %2.1f$^\circ$E %s $\Delta t, \Delta h, \Delta x \approx$10min, 1km, 20km' %( dt.strftime("%Y-%m-%d"), lats[0], lons[0], newline) )
    
    linestyles = ['-', '--', 'dotted']
    
    for j in range(nlons):
        for k in range(nlats):
            
            ax = axs[j,k]
            
            for i, dt in enumerate(datetimes):
                
                alpha = 0.4
                linestyle = linestyles[i]
                
                h_str = dt.strftime("%H:%M UT")
                
                label_u = "(%s)" %h_str
                label_v = None
                label_w = None
                
                if i == 0:
                    alpha = 1
                    linestyle = '-'
                    
                    label_u = "Zonal (%s)" %h_str
                    label_v = "Meridional (%s)" %h_str
                    label_w = 'Verticalx10 (%s)' %h_str
                
                ax.plot(u[i,j,k], alts, 'k', linestyle=linestyle, alpha=alpha, label=label_u)
                ax.plot(v[i,j,k], alts, 'b', linestyle=linestyle, alpha=alpha, label=label_v)
                # ax.plot(10*w[i,j,k], alts, 'g', linestyle=linestyle, alpha=alpha, label=label_w)
            
    ax.set_xlabel('Amplitude (m/s)')
    ax.set_ylabel('Height (km)')
    ax.grid(True)
    plt.legend()
    
    ax.set_xlim(-100,100)
    # ax.set_ylim(80,160)
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.close('all')
    
    return

def altitude_profiles(filename,
                       figpath,
                       datetimes,
                       lats,
                       lons,
                       delta_h=1, #km
                        vmins = [-100,-100,-15, None],
                        vmaxs = [ 100, 100, 15, None],
                        log_file=None,
                        ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    datetimes   =   np.atleast_1d(datetimes)
    lats        =   np.atleast_1d(lats)
    lons        =   np.atleast_1d(lons)
    
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    hmin = nn.lb[3].numpy()*1e-3
    hmax = nn.ub[3].numpy()*1e-3
    
    alts = np.arange(hmin, hmax, delta_h)
    epochs = [ (dt - datetime.datetime(1970,1,1)).total_seconds() for dt in datetimes]
    
    T, X, Y, Z = np.meshgrid(epochs, lons, lats, alts, indexing='ij')
    
    shape = T.shape
    
    outputs = nn.infer(T.flatten(), X.flatten(), Y.flatten(), Z.flatten())
    
    u = outputs[:,0].reshape(shape)
    v = outputs[:,1].reshape(shape)
    w = outputs[:,2].reshape(shape)
    
    # save_alt_profiles(epochs, lons, lats, alts, u, v, w)
    plot_alt_profiles(datetimes, lons, lats, alts, u, v, w, figpath=figpath)

def get_mean_winds_and_grads(nn, times, x_flat, y_flat, z_flat,
                    u_shape, valid_mask):
    
    ug = np.zeros(( (12,)+u_shape), dtype=np.float32)
    
    ugs = []
    
    print('')
    for time in times:
        print('m', end='')
        
        t = np.zeros_like(x_flat) + time
        
        nn_output = nn.infer_gradients(t, x_flat, y_flat, z_flat)
        
        ug[:] = np.nan
        
        for i in range(12):
            ug[i][valid_mask] = nn_output[:,i]
        
        ug0 = np.nanmean(ug, axis=(2,3))
        # ug0 = np.nanstd(ug, axis=(2,3))
        
        ugs.append(ug0)
    
    ugs = np.array(ugs)  
    
    ugs = np.transpose(ugs, (1,0,2))
    
    return(ugs)

def calc_kspectra(uvw, x, y, dx = 1e4, dy=1e4, N=100):
    
    xmax = np.max(x) - np.min(x)
    ymax = np.max(y) - np.min(y)
    
    Nx = xmax/dx
    Ny = ymax/dy
    
    dkx = 1.0/xmax
    dky = 1.0/ymax
    
    kx = (np.arange(Nx) - 0.5)*dkx
    ky = (np.arange(Ny) - 0.5)*dky
    
    
    return
    
def horizontal_spectra(filename,
               figpath,
               tstep = None, #min
                xstep = 1e4, #m
                ystep = 1e4, #m
                zstep = 500, #m
                ext='png',
                log_file=None,
                t0=0,
                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    tmin = nn.lb[0].numpy() + t0
    tmax = nn.ub[0].numpy()
    
    xmin = nn.lb[1].numpy()
    xmax = nn.ub[1].numpy()
    
    ymin = nn.lb[2].numpy()
    ymax = nn.ub[2].numpy()
    
    zmin = nn.lb[3].numpy()
    zmax = nn.ub[3].numpy()
    
    if tstep is None: tstep = (tmax - tmin)/(24*6)
    
    times   = np.arange(tmin, tmax, tstep)
    x       = np.arange(xmin, xmax, xstep)
    y       = np.arange(ymin, ymax, ystep)
    z       = np.arange(zmin, zmax, zstep)
    
    Y, X = np.meshgrid(y,x, indexing='ij')
    shape = X.shape 
    
    X = X.ravel()
    Y = Y.ravel()
    
    Hu = 0
    Hv = 0
    Hw = 0
    
    for time in times:
        print('x', end='')
        
        for zi in z:
            
            T = np.zeros_like(x) + time
            Z = np.zeros_like(x) + zi
            
            nn_output = nn.infer(T, x=X, y=Y, z=Z)
            
            Hu += calc_kspectra(nn_output, x=X, y=Y)
            # Hv += calc_spectra(nn_output[:,1], x=X, y=Y)
            # Hw += calc_spectra(nn_output[:,2], x=X, y=Y)
            
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = dt.strftime('%Y%m%d_%H%M%S')
    
    
    
def mean_winds(filename,
               figpath,
               tstep = None, #min
                xstep = 0.5, #degrees
                ystep = 0.25, #degrees
                zstep = 1, #km
                xrange = 4, #degrees
                yrange = 3, #degrees
                zrange = None, #km
                vmins = [-100,-100,-15, None],
                vmaxs = [ 100, 100, 15, None],
                ext='png',
                type='residuals',
                plot_mean=False,
                log_file=None,
                plot_fields=False,
                plot_vor_div=False,
                plot_wind_vectors=True,
                df_meteor=None,
                t0=0,
                df_ref=None,
                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    tmin = nn.lb[0].numpy() + t0
    tmax = nn.ub[0].numpy()
    
    if tstep is None: tstep = (tmax - tmin)/(24*6)
    if zrange is None: zrange = (nn.ub[3].numpy() - nn.lb[3].numpy())*1e-3
    
    lon_ref = nn.lon_ref
    lat_ref = nn.lat_ref
    alt_ref = nn.alt_ref
    
    xmin = lon_ref - xrange/2.
    xmax = lon_ref + xrange/2.
    
    ymin = lat_ref - yrange/2.
    ymax = lat_ref + yrange/2.
    
    zmin = alt_ref - zrange/2.
    zmax = alt_ref + zrange/2.
    
    zmin = int(zmin+1)
    
    times   = np.arange(tmin, tmax, tstep)
    x       = np.arange(xmin, xmax, xstep)
    y       = np.arange(ymin, ymax, ystep)
    z       = np.arange(zmin, zmax, zstep)
    
    Z, Y, X = np.meshgrid(z,y,x, indexing='ij')
    shape = X.shape 
    
    mask = np.where( ( (X-lon_ref)**2/(xrange/2.)**2 + (Y-lat_ref)**2/(yrange/2.)**2) <= 1 )
    
    x_flat = X[mask]
    y_flat = Y[mask]
    z_flat = Z[mask]
    
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = dt.strftime('%Y%m%d_%H%M%S')
    
    ug0 = get_mean_winds_and_grads(nn, times, x_flat, y_flat, z_flat, X.shape, mask)
    
    u1h = calc_mean_winds(times, z, ug0, x_width=1 * 60 * 60, y_width=1)
    
    if type == 'residuals':
        u4h = calc_mean_winds(times, z, ug0)#, y_width=0)
        u1h = u1h - u4h
    # vr = v0 - v4h
    # wr = w0 - w4h
    
    h5file = os.path.join(figpath, 'mean_wind_grad_%s.h5' %dt_str)
        
    with h5py.File(h5file, 'w') as fp:

        fp.create_dataset('epoch', data=times)
        fp.create_dataset('altitudes', data=z)
        fp.create_dataset('u0', data=ug0[0])
        fp.create_dataset('v0', data=ug0[1])
        fp.create_dataset('w0', data=ug0[2])
        
        fp.create_dataset('u_x', data=ug0[3])
        fp.create_dataset('v_x', data=ug0[4])
        fp.create_dataset('w_x', data=ug0[5])
        
        fp.create_dataset('u_y', data=ug0[6])
        fp.create_dataset('v_y', data=ug0[7])
        fp.create_dataset('w_y', data=ug0[8])
        
        fp.create_dataset('u_z', data=ug0[9])
        fp.create_dataset('v_z', data=ug0[10])
        fp.create_dataset('w_z', data=ug0[11])
        
            
    if plot_mean:
        
        figfile_mean = os.path.join(figpath, 'mean_wind_grads_%s.%s' %(dt_str, ext) ) 
        
        plot_mean_winds_grads(times, z, u1h,
                        figfile=figfile_mean,
                        vmins=vmins,
                        vmaxs=vmaxs)
        
        figfile_mean = os.path.join(figpath, 'mean_wind_%s.%s' %(dt_str, ext) ) 
        plot_mean_winds(times, z,
                        u1h[0], u1h[1], u1h[2],
                        figfile=figfile_mean,
                        vmins=vmins,
                        vmaxs=vmaxs,
                        df_ref=df_ref)
        return

def wind_at_samples(filename,
                   figpath,
                    ext='png',
                    df_meteor=None,
                    ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    times   = df_meteor['times'].values
    lons       = df_meteor['lons'].values
    lats       = df_meteor['lats'].values
    alts       = df_meteor['heights'].values
    
    u       = df_meteor['u'].values
    v       = df_meteor['v'].values
    w       = df_meteor['w'].values
    
    dops    = df_meteor['dops'].values
    
    dt = datetime.datetime.utcfromtimestamp(times[0])
    dt_str = dt.strftime('%Y%m%d')
    
    outputs, x, y, z = nn.infer(times, lons, lats, alts, return_xyz=True)
    
    ue = outputs[:,0]
    ve = outputs[:,1]
    we = outputs[:,2]
    
    h5file = os.path.join(figpath, 'winds_DNS_%s.h5' %dt_str)
    
    with h5py.File(h5file, 'w') as fp:
    
        fp.create_dataset('t', data=times)
        fp.create_dataset('x', data=x)
        fp.create_dataset('y', data=y)
        fp.create_dataset('z', data=z)
        
        fp.create_dataset('u', data=u)
        fp.create_dataset('v', data=v)
        fp.create_dataset('w', data=w)
            
        fp.create_dataset('ue', data=ue)
        fp.create_dataset('ve', data=ve)
        fp.create_dataset('we', data=we)
    
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Germany", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Virtual/ICON_20160815", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Argentina/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Norway/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Condor/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/JRO/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Piura/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/NewMexico/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Virtual/DNS_Simone2018/", help='Data path')
    
    parser.add_argument('-m', '--model', dest='model', default=None, help='neural network model')
    parser.add_argument('-s', '--subfolder', dest='subfolder', default="fcDOL1v2PINN_221.07", help='subfolder where the neural network model is stored')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='figures extension')
    parser.add_argument('-t', '--type', dest='type', default='full', help='plot type. Either "residuals" or "full" wind')
    parser.add_argument('-l', '--log-file', dest='log_file', default=None, help='select the i-th weights file from the log folder')
    
    # parser.add_argument('--meteor-path', dest='mpath', default='/Users/mcordero/Data/IAP/SIMONe/Virtual/ICON_20160815/ICON_+00+70+90', help='Data path')
    # parser.add_argument('--meteor-path', dest='mpath', default='/Users/mcordero/Data/IAP/SIMONe/Virtual/DNS_Simone2018/DNSx10_+12+53+91/', help='Data path')
    # parser.add_argument('--meteor-path', dest='mpath', default='/Users/mcordero/Data/IAP/SIMONe/Norway/ExtremeEvent/', help='Data path')
    parser.add_argument('--meteor-path', dest='mpath', default='/Users/mcordero/Data/IAP/SIMONe/Germany/Simone2018/', help='Data path')

    filename = '/Users/mcordero/Dropbox/Work/IAP/Papers/PINN/Data/10.22000-456/data/dataset/CharuvilEPS2022/windssimone_2018.h5'
    
    df_hari = read_H5File(filename)
    
    args = parser.parse_args()
    
    path  = args.dpath
    mpath = args.mpath
    model_name = args.model
    ext = args.ext
    type = args.type
    log_file = args.log_file
    
    xrange = 8
    yrange = 5
    
    xrange = 2.6
    yrange = 2.6
    
    xrange = 5
    yrange = 2.5
    
    #SIMONe Germany 2018
    xrange = 7
    yrange = 4
    
    t0 = 0*60*60
    
    vmins = [-150, -150, -10, -0.4, -0.4, -0.04, -0.4, -0.4, -0.04, -25, -25, -2.5]
    vmaxs = [ 150,  150,  10,  0.4,  0.4,  0.04,  0.4,  0.4,  0.04,  25,  25,  2.5]
    
    if type == 'residuals':
        vmins[0:3] = [-100, -100, -5]
        vmaxs[0:3] = [ 100,  100,  5]
    
    path_PINN = os.path.join(path, args.subfolder)
    
    if model_name is None:
        models = glob.glob1(path_PINN, 'model*[!s].h5')
        models = sorted(models)
    else:
        models = [  model_name ]
    
    meteor_sys = None
    if mpath is not None:
        meteor_sys = SMRReader(mpath)
            
    for model in models[2:]:
        
        id_name = model[-11:-3]
        filename = os.path.join(path_PINN, model)
        
        if not os.path.isfile(filename):
            continue
        
        figpath = os.path.join(path_PINN, 'ploty_test_%s' %model[:-3])
        
        if not os.path.exists(figpath):
            os.mkdir(figpath)
            
        figpath_type = os.path.join(figpath, '%s_%s_std' %(type, log_file) ) #+os.path.splitext(model_name)[0])
    
        keograms(filename,
                 figpath_type, ext=ext,
                 # ystep=0.1,
                 # z0=np.arange(90,92,0.25), #Vortex
                 xstep=0.1,
                 ystep=0.1,
                 zstep=0.5,
                 tstep=300,
                 xrange=xrange,
                 yrange=yrange,
                 time_width=0*60*60,
                 vmins=vmins,
                 vmaxs=vmaxs,
                 type=type,
                 log_file=log_file,
                 cmap='seismic',
                 t0=t0,
                # trange=60*60,
                # x0=16.04,
                # y0=69.3,
                # x0=17.5,
                # y0=70,
                # z0=92.0,
                # df_ref=df_hari,
               )
        
        # wind_at_samples(filename, figpath_type, df_meteor=df_meteor)
        #
        # continue
        #
        mean_winds(filename, figpath_type,
                   ext=ext,
                    type=type,
                    plot_mean=True,
                    t0=t0,
                    xrange=xrange,
                    yrange=yrange,
                    tstep=300,
                    # xmin=-30,
                    # ymin=-30,
                    # xstep=0.05,
                    # ystep=0.05,
                    zstep=0.5,
                    # zmin=84,
                    log_file=log_file,
                    vmins=vmins,
                    vmaxs=vmaxs,
                    df_ref=df_hari, 
                   )
        # continue
        