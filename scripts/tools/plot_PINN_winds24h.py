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
from georeference.geo_coordinates import lla2enu, lla2xyh

from pinn import pinn
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
    
    nrows = int( np.ceil( np.sqrt(nz) ) )
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
        
        axs_flat = axs.flat
        
        for zi in range(nz):
            
            ax = axs_flat[zi]
            
            im = ax.pcolormesh(lons, lats, f[zi],
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax)
            
            
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
    ax.set_xlabel('\nLongitude', fontsize=25, linespacing=2.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel('\nLatitude', fontsize=25, linespacing=2.8)
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
    ax.set_box_aspect([4,6,8])
    
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
    
    im = None
    im = ax.pcolormesh(x,y,w, cmap=cmap, vmin=wmin, vmax=wmax, alpha=0.3)
    
    im0 = ax.quiver(x, y, u, v,
                   # w,
                   # cmap=cmap,
                   # clim=[wmin, wmax],
                   scale=scale,
                   color='k'
                   )
    
    # ax.scatter(15.82, 69.45, marker='x', color='r')
    ax.scatter(16.04, 69.3, marker='o', c='r')
    
    ax.set(xlabel='Longitude', ylabel ='Latitude')
    
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
    ny_subplot = int( nz / nx_subplot +0.9) 

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
                       figsize=(5.5*nx_subplot, 5.0*ny_subplot),
                       squeeze=False)
    
    fig.suptitle(title, fontsize=14, weight='bold')
    
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
    
    ax.annotate('@Miguel Urco',
                     xy = (1.0, -0.1),
                     xycoords='axes fraction',
                     ha='right',
                     va="center",
                     fontsize=12,
                     weight='bold')
    
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
    ax.grid(True, linestyle='--')
    
    ax = plt.subplot(312)
    ax.set_title('Meridional wind')
    ax.set_ylabel(ylabel)
    
    if vmins is not None: vmin = vmins[1]
    if vmaxs is not None: vmax = vmaxs[1]
    
    im = ax.pcolormesh(num, h, v.T, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, linestyle='--')
    
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
    ax.grid(True, linestyle='--')
    
    plt.tight_layout(pad=0, rect=[0.02,0.0,1.05,0.99])
    
    ax.annotate('@Miguel Urco',
                     xy = (1.03, -0.2),
                     xycoords='axes fraction',
                     ha='left',
                     va="top",
                     fontsize=8,
                     weight='bold')
    plt.savefig(figfile)
    plt.close()

def plot_mean_winds_grads(t, h, ug,
                        figfile='./test.png',
                        vmins=None,
                        vmaxs=None,
                        cmap='seismic',
                        xlabel='UTC (h)',
                        ylabel='Altitude (km)',
                        titles=[
                                r'$u_0$ (m/s)', r'$v_0$ (m/s)', r'$w_0$ (m/s)',
                                r'$u_x$ (m/s/km)', r'$u_y$ (m/s/km)', r'$u_z$ (m/s/km)',
                                r'$v_x$ (m/s/km)', r'$v_y$ (m/s/km)', r'$v_z$ (m/s/km)',
                                r'$w_x$ (m/s/km)', r'$w_y$ (m/s/km)', r'$w_z$ (m/s/km)',
                                ]
                        ):
    
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
        
        d = ug[:,:,i].T
        if i > 2: d = 1e3*d
        
        im = ax.pcolormesh(num, h, d, cmap=cmap, vmin=vmin, vmax=vmax)
        
        plt.colorbar(im, ax=ax)
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle='--')
        
    
    for ax in axs_flat:
        ax.label_outer()
    
    plt.tight_layout(pad=0)
    
    ax.annotate('@Miguel Urco',
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

def calc_xyz_mean_winds(filenames, times, X, Y, Z, t0):
    
    ug = get_winds(filenames, times, X, Y, Z, t0)
    
    u = ug[:,:,:,:,0]
    v = ug[:,:,:,:,1]
    w = ug[:,:,:,:,2]
    
    u0 = np.nanmean(u, axis=(2,3))
    v0 = np.nanmean(v, axis=(2,3))
    w0 = np.nanmean(w, axis=(2,3))
    
    return(u0, v0, w0)

def calc_mean_winds(times, altitudes,
                       u0, v0, w0,
                       time_width=1*60*60, time_sigma=0.5,
                       alt_width=4, alt_sigma=0.5):
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
               tstep = None, #min
                xstep = 0.2, #degrees
                ystep = 0.2, #degrees
                zstep = 0.5,
                x0 = None, #degrees
                y0 = None,
                z0 = None, #km
                xrange = 6, #degrees
                yrange = 3, #degrees
                zrange = None, #km
                vmins = [-100,-100,-15, None],
                vmaxs = [ 100, 100, 15, None],
                ext='png',
                type='residuals',
                plot_mean=False,
                log_file=None,
                time_width=0,
                cmap='jet',
                t0=0*60*60,
                trange=None,#3*60*60,
                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    tmin = nn.lb[0].numpy() + t0
    if trange is None: tmax = nn.ub[0].numpy() - t0
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
                                 time_width=time_width,
                                 alt_width=0)
    
    if type == 'residuals':
        u0, v0, w0 = calc_mean_winds(times, x, u, v, w,
                                 time_width=4*60*60,
                                 alt_width=0)
        
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
                                 time_width=time_width,
                                 alt_width=0)
    
    if type == 'residuals':
        u0, v0, w0 = calc_mean_winds(times, y, u, v, w,
                                 time_width=4*60*60,
                                 alt_width=0)
        
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
                                 time_width=time_width,
                                 alt_width=0)
    
    if type == 'residuals':
        u0, v0, w0 = calc_mean_winds(times, z, u, v, w,
                                 time_width=4*60*60,
                                 alt_width=0)
        
        u -= u0
        v -= v0
        w -= w0
        
    plot_mean_winds(times, z, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='Altitude (km)')
    
def getting_and_plotting_winds(filenames,
                               figpath,
                               tstep = None, #min
                                xstep = 0.1, #degrees
                                ystep = 0.1, #degrees
                                zstep = 2, #km
                                xrange = 4, #degrees
                                yrange = 3, #degrees
                                zrange = None, #km
                                trange=24*60*60,
                                vmins = [-100,-100,-15, None],
                                vmaxs = [ 100, 100, 15, None],
                                ext='png',
                                type='residuals',
                                plot_mean=False,
                                log_file=None,
                                plot_fields=False,
                                plot_vor_div=True,
                                plot_wind_vectors=True,
                                t0=0,
                                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    nn = pinn.PINN()
    nn.restore(filenames[0], log_index=log_file)
    
    tmin = nn.lb[0].numpy()
    tmax = nn.ub[0].numpy() + trange
    
    if tstep is None: tstep = (tmax - tmin)/(24)
    if zrange is None: zrange = (nn.ub[3].numpy() - nn.lb[3].numpy())*1e-3
    
    lon_ref = nn.lon_ref
    lat_ref = nn.lat_ref
    alt_ref = nn.alt_ref
    
    xmin = lon_ref - xrange/2.
    xmax = lon_ref + xrange/2.
    
    ymin = lat_ref - yrange/2.
    ymax = lat_ref + yrange/2.
    
    zmin = 84 #alt_ref - zrange/2.
    zmax = zmin + zrange
    
    # zmin = int(zmin)
    
    times   = np.arange(tmin, tmax, tstep)
    x       = np.arange(xmin, xmax, xstep)
    y       = np.arange(ymin, ymax, ystep)
    z       = np.arange(zmin, zmax, zstep)
    
    Z, Y, X = np.meshgrid(z,y,x, indexing='ij')
    shape = X.shape 
    
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = dt.strftime('%Y%m%d')
    
    figfile_mean = os.path.join(figpath, 'mean_wind_%s.%s' %(dt_str, ext) ) 
    figfile_4h_mean = os.path.join(figpath, 'mean_4h_wind_%s.%s' %(dt_str, ext) ) 
    figfile_res_mean = os.path.join(figpath, 'mean_res_wind_%s.%s' %(dt_str, ext) )   
    
    if (type == 'residuals') or plot_mean:
        u0,   v0,  w0 = calc_xyz_mean_winds(filenames, times, X, Y, Z, t0)
        u4h, v4h, w4h = calc_mean_winds(times, z, u0, v0, w0, alt_width=0)
        
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
    
        fp.create_dataset('x', data=X, compression="gzip")
        fp.create_dataset('y', data=Y, compression="gzip")
        fp.create_dataset('z', data=Z, compression="gzip")
        
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
            
            if plot_vor_div:
                outputs = nn.infer_gradients(t, x_flat, y_flat, z_flat)
                
                u_x_flat = outputs[:,3]
                u_y_flat = outputs[:,4]
                # u_z_flat = outputs[:,5]
                
                v_x_flat = outputs[:,6]
                v_y_flat = outputs[:,7]
                # v_z_flat = outputs[:,8]
                
                # w_x_flat = outputs[:,9]
                # w_y_flat = outputs[:,10]
                w_z_flat = outputs[:,11]
                
                div_flat = u_x_flat + v_y_flat
                vort_flat = v_x_flat - u_y_flat
                
                div  = np.zeros_like(X) + np.nan
                vort = np.zeros_like(X) + np.nan
                w_z = np.zeros_like(X) + np.nan
                
                div[mask] = div_flat*1e3
                vort[mask] = vort_flat*1e3
                w_z[mask]   = w_z_flat*1e3
                
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
            g.create_dataset('u', data=u, compression="gzip")
            g.create_dataset('v', data=v, compression="gzip")
            g.create_dataset('w', data=w, compression="gzip")
            
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
                
                plot_heatmaps(x, y, z,
                           fields = [vort, div, w_z],
                           titles=['Relative vorticity (m/s/km)','Horizontal Divergence (m/s/km)', r'$w_z$ (m/s/km)'],
                           figtitle='%s'%dt.strftime('%Y/%m/%d %H:%M:%S'),
                           vmins = [-5, -5, -5],
                           vmaxs = [ 5,  5,  5],
                           prefix='%s'%dt.strftime('%Y%m%d_%H%M%S'),
                           path=figpath_3D)
                
            
            if plot_wind_vectors:
                figfile = os.path.join(figpath_3D, 'wind_vec_%s.%s' %(dt.strftime('%Y%m%d_%H%M%S'), ext) )
                
                plot_3d_vectors(X, Y, z,
                             [u,v,w],
                             title=r'%s' %( dt.strftime('%Y.%m.%d %H:%M:%S') ),
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

    # spec_figname = os.path.join(figpath, 'spectra_%s.png' %dt_str)
    # plot_spectra( [Fu/N, Fv/N, Fw/N], k, filename=spec_figname)

def plot_alt_profiles(datetimes, lons, lats, alts, u, v, w, figpath='./'):
    
    nlons = len(lons)
    nlats = len(lats)
    
    dt = datetimes[0]
    
    figname = os.path.join(figpath, 'wind_alt_profile_%s_%2.1f_%2.1f.png' %( dt.strftime("%Y%m%d_%H%M%S"), lons[0], lats[0]) )
    
    fig, axs = plt.subplots(nlons, nlats, figsize=(nlats*7, nlons*7), squeeze=False)
    fig.suptitle('SIMONe Norway <>(10 min - 1 km): %s @ %2.1f N, %2.1f E' %( dt.strftime("%Y-%m-%d"), lats[0], lons[0] ) )
    
    for j in range(nlons):
        for k in range(nlats):
            
            ax = axs[j,k]
            
            for i, dt in enumerate(datetimes):
                
                alpha = i*0.3
                linestyle = '--'
                
                h_str = dt.strftime("%H:%M")
                
                label_u = "(%s)" %h_str
                label_v = None
                label_w = None
                
                if i == 0:
                    alpha = 1
                    linestyle = '-'
                    
                    label_u = "Zonal (%s)" %h_str
                    label_v = "Meridional (%s)" %h_str
                    label_w = 'Verticalx10 (%s)' %h_str
                
                ax.plot(u[i,j,k], alts, 'b', linestyle=linestyle, alpha=alpha, label=label_u)
                ax.plot(v[i,j,k], alts, 'r', linestyle=linestyle, alpha=alpha, label=label_v)
                ax.plot(10*w[i,j,k], alts, 'g', linestyle=linestyle, alpha=alpha, label=label_w)
            
            ax.set_xlabel('Amplitude (m/s)')
            ax.set_ylabel('Height (km)')
            ax.grid(True)
            plt.legend()
            ax.set_xlim(-100,100)
    
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
    epochs = [dt.timestamp() for dt in datetimes]
    
    T, X, Y, Z = np.meshgrid(epochs, lons, lats, alts, indexing='ij')
    
    shape = T.shape
    
    outputs = nn.infer(T.flatten(), X.flatten(), Y.flatten(), Z.flatten())
    
    u = outputs[:,0].reshape(shape)
    v = outputs[:,1].reshape(shape)
    w = outputs[:,2].reshape(shape)
    
    # save_alt_profiles(epochs, lons, lats, alts, u, v, w)
    plot_alt_profiles(datetimes, lons, lats, alts, u, v, w, figpath=figpath)


def get_winds(filenames, times, X, Y, Z, t0=0*60):
    
    nz, ny, nx = X.shape
    
    x = np.ravel(X)
    y = np.ravel(Y)
    z = np.ravel(Z)
    
    nn = pinn.PINN()
    
    ind = 0
    nn.restore(filenames[ind], log_index=log_file)
    
    tmin = nn.lb[0].numpy() + t0
    tmax = nn.ub[0].numpy() - t0
    
    u = np.zeros( (times.shape[0], x.shape[0], 3), dtype=np.float32 ) + np.nan
    
    for i, time in enumerate(times):
        print('m', end='')
        
        if time < tmin:
            continue
        
        if time > tmax: 
            ind += 1
            nn.restore(filenames[ind], log_index=log_file)
    
            tmin = nn.lb[0].numpy() + t0
            tmax = nn.ub[0].numpy() - t0
        
        t = np.zeros_like(x) + time
        
        u[i] = nn.infer(t, x, y, z)
    
    u = np.reshape(u, (-1,nz,ny,nx,3) )
    
    return(u)

def get_winds_and_grads(filenames, times, X, Y, Z, t0=0*60):
    
    nz, ny, nx = X.shape
    
    x = np.ravel(X)
    y = np.ravel(Y)
    z = np.ravel(Z)
    
    nn = pinn.PINN()
    
    ind = 0
    nn.restore(filenames[ind], log_index=log_file)
    
    tmin = nn.lb[0].numpy() + t0
    tmax = nn.ub[0].numpy() - t0
    
    ug = np.zeros( (times.shape[0], x.shape[0], 12), dtype=np.float32 ) + np.nan
    
    for i, time in enumerate(times):
        print('m', end='')
        
        if time < tmin:
            continue
        
        if time > tmax: 
            ind += 1
            
            if ind >= len(filenames): break
            
            nn.restore(filenames[ind], log_index=log_file)
    
            tmin = nn.lb[0].numpy() + t0
            tmax = nn.ub[0].numpy() - t0
        
        t = np.zeros_like(x) + time
        
        ug[i] = nn.infer_gradients(t, x, y, z)
    
    ug = np.reshape(ug, (-1,nz,ny,nx,12) )
    
    return(ug)

def mean_winds(filenames,
               figpath,
               tstep = None, #min
                xstep = 0.1, #degrees
                ystep = 0.1, #degrees
                zstep = 2, #km
                xrange = 4, #degrees
                yrange = 3, #degrees
                zrange = None, #km
                trange = 12*60*60,
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
                t0 = 30*60,
                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    nn = pinn.PINN()
    nn.restore(filenames[0], log_index=log_file)
    
    tmin = nn.lb[0].numpy()
    tmax = nn.ub[0].numpy() + trange
    
    if tstep is None: tstep = (tmax - tmin)/(24*2)
    if zrange is None: zrange = (nn.ub[3].numpy() - nn.lb[3].numpy())*1e-3
    
    lon_ref = nn.lon_ref
    lat_ref = nn.lat_ref
    alt_ref = nn.alt_ref
    
    xmin = lon_ref - xrange/2.
    xmax = lon_ref + xrange/2.
    
    ymin = lat_ref - yrange/2.
    ymax = lat_ref + yrange/2.
    
    zmin = 84 #alt_ref - zrange/2.
    zmax = zmin + zrange
    
    times   = np.arange(tmin, tmax, tstep)
    x       = np.arange(xmin, xmax, xstep)
    y       = np.arange(ymin, ymax, ystep)
    z       = np.arange(zmin, zmax, zstep)
    
    Z, Y, X = np.meshgrid(z,y,x, indexing='ij')
    # shape = X.shape 
    
    # mask = np.where( ( (X-lon_ref)**2/(xrange/2.)**2 + (Y-lat_ref)**2/(yrange/2.)**2) <= 1 )
    #
    # x_flat = X[mask]
    # y_flat = Y[mask]
    # z_flat = Z[mask]
    
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = dt.strftime('%Y%m%d_%H%M%S')
    
    ug0 = get_winds_and_grads(filenames, times, X, Y, Z, t0)
    
    ug0 = np.nanmean(ug0, axis=(2,3))
    
    u4h, v4h, w4h = calc_mean_winds(times, z, ug0[:,:,0], ug0[:,:,1], ug0[:,:,2], alt_width=1)
    ug0[:,:,0] -= u4h
    ug0[:,:,1] -= v4h
    ug0[:,:,2] -= w4h
    
    
    h5file = os.path.join(figpath, 'mean_wind_grad_%s.h5' %dt_str)
        
    with h5py.File(h5file, 'w') as fp:

        fp.create_dataset('epoch', data=times)
        fp.create_dataset('altitudes', data=z)
        fp.create_dataset('u0', data=ug0[:,:,0])
        fp.create_dataset('v0', data=ug0[:,:,1])
        fp.create_dataset('w0', data=ug0[:,:,2])
        
        fp.create_dataset('u_x', data=ug0[:,:,3])
        fp.create_dataset('u_y', data=ug0[:,:,4])
        fp.create_dataset('u_z', data=ug0[:,:,5])
        
        fp.create_dataset('v_x', data=ug0[:,:,6])
        fp.create_dataset('v_y', data=ug0[:,:,7])
        fp.create_dataset('v_z', data=ug0[:,:,8])
        
        fp.create_dataset('w_x', data=ug0[:,:,9])
        fp.create_dataset('w_y', data=ug0[:,:,10])
        fp.create_dataset('w_z', data=ug0[:,:,11])
        
            
    if plot_mean:
        
        figfile_mean = os.path.join(figpath, 'mean_wind_grads_%s.%s' %(dt_str, ext) ) 
        
        plot_mean_winds_grads(times, z, ug0,
                        figfile=figfile_mean,
                        vmins=vmins,
                        vmaxs=vmaxs)
        
        return
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Germany", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Virtual/ICON_20160816", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Argentina/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Norway/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Condor/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Piura/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Virtual/DNS_Simone2018/", help='Data path')
    
    parser.add_argument('-m', '--model', dest='model', default=None, help='neural network model')
    parser.add_argument('-s', '--subfolder', dest='subfolder', default="fPINN_45.20", help='subfolder where the neural network model is stored')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='figures extension')
    parser.add_argument('-t', '--type', dest='type', default='full', help='plot type. Either "residuals" or "full" wind')
    parser.add_argument('-l', '--log-file', dest='log_file', default=1000, help='select the i-th weights file from the log folder')
    
    parser.add_argument('--meteor-path', dest='mpath', default=None, help='Data path')
    
    args = parser.parse_args()
    
    path  = args.dpath
    mpath = args.mpath
    model_name = args.model
    ext = args.ext
    type = args.type
    log_file = args.log_file
    
    t0 = 0#15*60
    
    vmins = [-80, -80, -10, -0.4, -0.4, -25, -0.4, -0.4, -25, -0.04, -0.04, -2.5]
    vmaxs = [ 80,  80,  10,  0.4,  0.4,  25,  0.4,  0.4,  25,  0.04,  0.04,  2.5]
    
    # vmins = None
    # vmaxs = None
    
    df_meteor = None
    if mpath is not None:
        meteor_sys = SMRReader(path_meteor_data)
        meteor_sys.read_next_file()
        df_meteor = meteor_sys.df
    
    path_PINN = os.path.join(path, args.subfolder)
    
    if model_name is None:
        models = glob.glob1(path_PINN, 'model*.h5')
        models = sorted(models)
    else:
        models = [  model_name ]
    
    figpath = os.path.join(path_PINN, 'plot_%s' %models[0][:10])
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    figpath_type = os.path.join(figpath, '%s_%s_std' %(type, log_file) ) #+os.path.splitext(model_name)[0])
        
    filenames = [os.path.join(path_PINN, model) for model in models]
    
    mean_winds(filenames,
               figpath_type,
               ext=ext,
                type=type,
                plot_mean=True,
                # xmin=-30,
                # ymin=-30,
                # xstep=0.05,
                # ystep=0.05,
                zstep=0.5,
                # zmin=84,
                log_file=log_file,
                vmins=vmins,
                vmaxs=vmaxs,
                df_meteor=df_meteor,
               )
    
    # keograms(filenames,
    #          figpath_type, ext=ext,
    #          # ystep=0.1,
    #          # z0=np.arange(90,95,0.3), #Vortex
    #          # x0=16.04,
    #          # y0=69.3,
    #          # z0=89,
    #          xstep=0.05,
    #          ystep=0.05,
    #          zstep=0.25,
    #          # time_width=300,
    #          vmins=vmins,
    #          vmaxs=vmaxs,
    #         type=type,
    #         log_file=log_file,
    #         cmap='seismic',
    #         t0=t0,
    #
    #        )
    #
    # getting_and_plotting_winds(filenames,
    #                            figpath_type, ext=ext,
    #                             type=type,
    #                             plot_mean=False,
    #                             # xmin=-30,
    #                             # ymin=-30,
    #                             zstep=4,
    #                             zrange=16,
    #                             # zmin=84,
    #                             log_file=log_file,
    #                             vmins=vmins,
    #                             vmaxs=vmaxs,
    #                             t0=t0,
    #                            )