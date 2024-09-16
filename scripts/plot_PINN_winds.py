'''
Created on 30 Aug 2022

@author: radar
'''
import os, glob
import time, datetime
import numpy as np

import h5py

from scipy.ndimage import convolve1d, gaussian_filter

import tensorflow as tf
DTYPE='float32'

import matplotlib.pyplot as plt
# plt.rcParams['axes.facecolor'] = "#B0B0B0"

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.dates as mdates


# from atmospheric_models.ICON import ICONReader
from georeference.geo_coordinates import lla2enu, lla2xyh

from pinn import hyper as pinn
from radar.smr.smr_file import SMRReader
    
from utils.plotting import plot_3d_vectors, plot_mean_winds, plot_mean_winds_grads
from utils.io import save_h5file

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

def save_daemodel(x, y, z, filename, vmin=-1, vmax=1, cmap='RdBu_r', rotation_angles=(-np.pi/2, np.pi, 0)):
    
    import matplotlib.colors as mcolors
    # from trimesh import Trimesh
    import trimesh
    
    x -= x.mean()
    y -= y.mean()
    
    # Normalize Z values and apply colormap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    colors = cmap(norm(z.ravel()))

    # # Mask NaN values
    # mask = ~np.isnan(z)
    #
    # # Apply mask to x, y, z
    # x = x[mask]
    # y = y[mask]
    # z = z[mask]
    #
    # # Recreate a grid assuming mask results in a rectangular shape
    # num_points = len(x)
    # grid_size = 18  # Approximate grid size (assuming square grid)
    #
    # x = x.reshape(grid_size, -1)
    # y = y.reshape(grid_size, -1)
    # z = z.reshape(grid_size, -1)
    
    # Create vertices
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    
    # Create faces
    rows, cols = x.shape  # Recalculate based on masked data
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v1 = i * cols + j
            v2 = v1 + 1
            v3 = v1 + cols
            v4 = v3 + 1
            if v1 in range(vertices.shape[0]) and v2 in range(vertices.shape[0]) and v3 in range(vertices.shape[0]) and v4 in range(vertices.shape[0]):
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
    faces = np.array(faces)
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Apply scaling
    # mesh.apply_scale(scale)

    # Apply rotation
    rotation_matrix = trimesh.transformations.euler_matrix(rotation_angles[0], rotation_angles[1], rotation_angles[2], 'sxyz')
    mesh.apply_transform(rotation_matrix)
    
    # # Remove duplicate faces
    # mesh.update_faces(mesh.unique_faces())
    #
    # # Remove degenerate faces (those with zero area)
    # mesh.remove_degenerate_faces()
    #
    # # Fill any holes in the mesh
    # mesh.fill_holes()
    #
    # # Merge close vertices to prevent overlapping
    # mesh.merge_vertices()
    #
    # # Recompute normals to ensure correct shading
    # mesh.rezero()  # Center the mesh if needed
    # mesh.fix_normals()  # Recalculate normals

    vertex_colors = colors[:, :3]
    mesh.visual.vertex_colors = vertex_colors
    
    # Simplify the mesh, targeting a specific number of faces (e.g., 1000 faces)
    # mesh = mesh.simplify_quadratic_decimation(1000)

    # Export to a 3D file format
    mesh.export(filename+".dae", scale=1.0, rotation_angles=(np.pi/2, 0, 0)) 
    
    # mesh.show()
    
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
        # plt.suptitle(figtitle)
        
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
                                 alpha=alpha,
                                 linewidth=1, 
                                 antialiased=True,
                                 rcount=100,
                                 ccount=100,
                                 )
            
            ax.set_zlim(2*vmin, 2*vmax)
            
            # ax.set_facecolor('white')
            # ax.axis("off")
            # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # ax.w_zaxis.line.set_visible(False)


            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # ax.set_title('%s %2.1fkm' %(titles[iplot], alts[zi]) )
            
            # ax.plot(samp_lons, samp_lats, 'mx')
            if df_sampling is not None:
                ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x', color='k')
            
            # ax.set_xlim(-2,2)
            # ax.set_ylim(0,2)
            
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes(position='right', size="5%", pad=0.05)

            plt.colorbar(im, ax=ax, label="m/s", shrink=0.2, aspect=20*0.2)
            
            if nrows>1:
                ax = fig.add_subplot(nrows, nfields, nfields+iplot+1)#, projection='3d')
        
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
                ax.set_zlim(vmin, vmax)
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title('%s_est %2.1f km' %(titles[iplot], alts[zi]) )
                
                # 
                if df_sampling is not None: 
                    ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x', color='k')
                
                plt.colorbar(im, ax=ax, label="m/s")
            
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
                
                plt.colorbar(im, ax=ax, label="m/s")
        
        plt.tight_layout()
        
        # for ii in range(0,360,10):
        #     ax.view_init(elev=ii, azim=150)
        #     plt.savefig(filename+"*%d.png" %ii, transparent=True, dpi=500)
        
        # ax.view_init(elev=10, azim=150)
        
        plt.savefig(filename, transparent=True, dpi=500)
        plt.close()
        
        save_daemodel(LON, LAT, f[zi], filename)

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
        
        if vmin_list is not None:
            vmin = vmin_list[i]
        else:
            vmin = np.nanmin(data)
            
        if vmax_list is not None:
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

def plot_3d_vectors_hprofile(x3d, z3d, y1d,
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
                 ydecimator=6,
                 scale=None):
    """
    Input:
        [u,v,w]    :    array-like (nplots, nranges, ny, nx)
    """
    u, v, w = data_list
    
    u = u[:,::ydecimator]
    v = v[:,::ydecimator]
    w = w[:,::ydecimator]
    y1d = y1d[::ydecimator]
    
    nz, ny, nx = u.shape
    
    nx_subplot = int( np.ceil( np.sqrt(ny) ) )
    ny_subplot = int( ny / nx_subplot +0.9) 

    xmin = np.min(x3d)
    xmax = np.max(x3d)
    ymin = np.min(y1d)
    ymax = np.max(y1d)
    zmin = np.min(z3d)
    zmax = np.max(z3d)
    
    wmin = vmin_list[2]
    wmax = vmax_list[2]
    
    # plt.style.use('dark_background')
    fig, axs = plt.subplots(ny_subplot, nx_subplot,
                       sharex=True, sharey=True,
                       figsize=(5.5*nx_subplot, 5.0*ny_subplot),
                       squeeze=False)
    
    fig.suptitle(title, fontsize=14, weight='bold')
    
    for i, ax in enumerate(axs.flat):
        
        if i >= ny: break
        
        im = _plot_3d_vector(x3d[:,i,:], z3d[:,i,:],
                        u[:,i,:], v[:,i,:], w[:,i,:],
                        ax,
                        title='y=%2.1f' %y1d[i],
                        xmin=xmin, xmax=xmax,
                        ymin=zmin, ymax=zmax,
                        wmin=wmin, wmax=wmax,
                        scale=scale,
                        cmap=cmap,
                        cmap_label=cmap_label,
                        )
    
    for ax in axs.flat:
        ax.label_outer()
    
    plt.tight_layout()
    
    ax.annotate('@mu',
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

# def plot_mean_winds(t, h, u, v, w,
#                     figfile='./test.png',
#                     vmins=None,
#                     vmaxs=None,
#                     cmap='seismic',
#                     ylabel='Altitude (km)',
#                     histogram=True):
#
#     vmin = None
#     vmax = None
#
#     num = mdates.epoch2num(t)
#     locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
#     formatter = mdates.ConciseDateFormatter(locator)
#
#     ncols = 1
#     if histogram:
#         ncols = 2
#
#     us = [u, v, w]
#     titles = ['Zonal wind', 'Meridional wind', 'Vertical wind']
#
#     # fig, axs = plt.subplots(3, ncols, figsize=(8,6), squeeze=False, sharex=False)
#
#     axs = []
#     fig = plt.figure(figsize=(8,6))
#
#     for i in range(3):
#
#         data = us[i]
#         if vmins is not None: vmin = vmins[i]
#         if vmaxs is not None: vmax = vmaxs[i]
#
#         # ax = axs[i,0]
#         ax = plt.subplot2grid( (3,2+ncols), (i,0), colspan=3, rowspan=1 )
#
#         ax.set_title(titles[i])
#         ax.set_ylabel(ylabel)
#
#         im = ax.pcolormesh(num, h, data.T, cmap=cmap, vmin=vmin, vmax=vmax)
#
#         plt.colorbar(im, ax=ax)#, label='m/s')
#
#         ax.xaxis.set_major_locator(locator)
#         ax.xaxis.set_major_formatter(formatter)
#         # ax.xaxis.set_ticklabels([])
#
#         ax.grid(True, linestyle='--')
#
#         axs.append(ax)
#
#         if histogram:
#             # ax0 = axs[i,1]
#             ax0 = plt.subplot2grid( (3,2+ncols), (i,3), colspan=1, rowspan=1 )
#
#             if i == 0: ax0.set_title('Histogram')
#
#             ax0.set_xlabel('%s (m/s)' %titles[i])
#
#             d = data[np.isfinite(data)]
#             ax0.hist(d, bins=20, density=True, color='grey', alpha=0.5)
#             ax0.grid(True, linestyle='--')
#             ax0.set_xlim(vmin, vmax)
#
#             axs.append(ax0)
#
#
#     ax.set_xlabel('UTC (h)')
#
#
#     plt.tight_layout(pad=0, rect=[0.02,0.0,0.98,0.99])
#
#     # axs[-1].annotate('@mu',
#     #                  xy = (1.03, -0.2),
#     #                  xycoords='axes fraction',
#     #                  ha='left',
#     #                  va="top",
#     #                  fontsize=8,
#     #                  weight='bold')
#
#     plt.savefig(figfile)
#     plt.close()
    
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
        
        outputs = nn.infer(t, x_flat, y_flat, z_flat, filter_output=False)
        
        u[:] = np.nan
        v[:] = np.nan
        w[:] = np.nan
        
        u[valid_mask] = outputs[:,0]
        v[valid_mask] = outputs[:,1]
        w[valid_mask] = outputs[:,2]
        
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
    wx = int(x_width/dx)
    
    #Filter in altitude
    dy = y[1] - y[0]
    wy = int(y_width/dy)
    
    if (wx <= 1) and (wy <= 1):
        return(u)
    
    u_filtered = np.empty_like(u)
    
    for i in range(u.shape[0]):
        u_filtered[i] = gaussian_filter(u[i], sigma=(wx, wy), mode='nearest', truncate=1)
    
    # plt.subplot(121)
    # plt.pcolormesh(u[i])
    #
    # plt.subplot(122)
    # plt.pcolormesh(u_filtered[i])
    #
    # plt.show()
    
    # M = (N-1)/2
    #
    # k = np.arange(N)
    # w = np.exp(-0.5*((k-M)/M*x_sigma)**2)
    #
    # w = np.exp(-((tf-m)/sig)**2/2)/(sig*np.sqrt(2*np.pi))
    #
    # w = np.ones(N)
    # w /= np.sum(w)
    #
    # u4h = np.empty_like(u)
    #
    # for i in range(u.shape[0]):
    #     u4h[i] = convolve1d(u[i], w, axis=x_axis, mode='nearest')
    #
    # # v4h = convolve1d(v0, w, axis=x_axis, mode='nearest')
    # # w4h = convolve1d(w0, w, axis=x_axis, mode='nearest')
    #
    # #Filter in altitude
    # dy = y[1] - y[0]
    # N = int(y_width/dy)
    #
    # if N<2:
    #     return(u4h)
    #
    # M = (N-1)/2
    #
    # k = np.arange(N)
    #
    # w = np.exp(-0.5*((k-M)/M*y_sigma)**2)
    #
    # w = np.ones(N)
    # w /= np.sum(w)
    #
    # for i in range(u.shape[0]):
    #     u4h[i] = convolve1d(u4h[i], w, axis=y_axis, mode='nearest')
        
    # u4h = convolve1d(u4h, w, axis=y_axis, mode='nearest')
    # v4h = convolve1d(v4h, w, axis=y_axis, mode='nearest')
    # w4h = convolve1d(w4h, w, axis=y_axis, mode='nearest')
    
    return(u_filtered)
    
def getting_and_plotting_winds(filename,
                               figpath,
                               tstep = None, #min
                                xstep = 0.4, #degrees
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
                                save=False,
                                meteor_path=None
                                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    if plot_vor_div:
        calc_grads = True
        
    try:
        nn = pinn.restore(filename, log_index=log_file)
    except:
        return
        nn = pinn.PINN()
        nn.restore(filename, log_index=log_file)
        
    tmin = nn.lb[0] + t0
    tmax = nn.ub[0]
    
    x_lb = nn.lb[2]
    y_lb = nn.lb[3]
    z_lb = nn.lb[1]
    
    x_ub = nn.ub[2]
    y_ub = nn.ub[3]
    z_ub = nn.ub[1]
    
    
    if trange is not None: tmax = tmin+trange
    if tstep is None: tstep = (tmax - tmin)/(24*6)
    
    if xrange is None: xrange = (x_ub - x_lb)/40e3
    if yrange is None: yrange = (y_ub - y_lb)/90e3
    if zrange is None: zrange = (z_ub - z_lb)*1e-3
    
    lon_ref = nn.lon_ref + 0.3
    lat_ref = nn.lat_ref + 0.2
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
    
    mask = np.where( ( (X-lon_ref)**2/(xrange)**2 + (Y-lat_ref)**2/(yrange)**2) <= 5 )
    # mask = np.where( ( (X-lon_ref)**2/(xrange/2)**2 + (Y-lat_ref)**2/(yrange/2)**2) >=0.1 )
    
    x_flat = X[mask]
    y_flat = Y[mask]
    z_flat = Z[mask]
    
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = dt.strftime('%Y%m%d')
    
    
    if meteor_path is not None:
        meteor_sys = SMRReader(mpath)
        meteor_sys.read_next_file()
        # meteor_sys.filter()
        df_meteor = meteor_sys.df
        
    
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
    
    if save:
        with h5py.File(h5file, 'w') as fp:
        
            fp.create_dataset('x', data=X)
            fp.create_dataset('y', data=Y)
            fp.create_dataset('z', data=Z)
        
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
            u_y_flat = outputs[:,6]
            u_z_flat = outputs[:,9]
            
            v_x_flat = outputs[:,4]
            v_y_flat = outputs[:,7]
            v_z_flat = outputs[:,10]
            
            w_x_flat = outputs[:,5]
            w_y_flat = outputs[:,8]
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
            
            u_x[mask] = u_x_flat #from m/s/m to m/s/km
            v_x[mask] = v_x_flat
            w_x[mask] = w_x_flat
            
            u_y[mask] = u_y_flat
            v_y[mask] = v_y_flat
            w_y[mask] = w_y_flat
            
            u_z[mask] = u_z_flat
            v_z[mask] = v_z_flat
            w_z[mask] = w_z_flat
            
            div  = u_x + v_y #+ w_z
            vort = v_x - u_y
            
        else:
            outputs = nn.infer(t, x_flat, y_flat, z_flat, filter_output=True)
        
        u_flat = outputs[:,0]
        v_flat = outputs[:,1]
        w_flat = outputs[:,2]
        
        u = np.zeros_like(X) + np.nan
        v = np.zeros_like(X) + np.nan
        w = np.zeros_like(X) + np.nan
        
        u[mask] = u_flat
        v[mask] = v_flat
        w[mask] = w_flat
        
        if save:
            with h5py.File(h5file, 'w') as fp:
                
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
        
        df_meteor_t = None
        if df_meteor is not None:
            mask1 = np.abs(df_meteor['times'] - time) < 5*60
            df_meteor_t = df_meteor[mask1]
                
        if plot_fields:
            # plot_field(x, y, z,
            #            fields = [u, v, w],
            #            titles=['u','v','w', 'P'],
            #            figtitle='%s'%dt.strftime('%Y/%m/%d %H:%M:%S'),
            #            vmins = vmins,
            #            vmaxs = vmaxs,
            #            prefix='field_%s'%dt.strftime('%Y%m%d_%H%M%S'),
            #            path=figpath_3D)
            
            plot_field(x, y, z,
                       fields = [w],
                       titles=['w'],
                       figtitle='%s'%dt.strftime('%Y/%m/%d %H:%M:%S'),
                       vmins = vmins[2:],
                       vmaxs = vmaxs[2:],
                       prefix='field_%s'%dt.strftime('%Y%m%d_%H%M%S'),
                       path=figpath_3D,
                       df_sampling=df_meteor_t)
            
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
                       vmins = vmins[-3:],
                       vmaxs = vmaxs[-3:],
                       prefix='%s'%dt.strftime('%Y%m%d_%H%M%S'),
                       path=figpath_3D)
            
        
        if plot_wind_vectors:
            
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
                         scale=3000*vmaxs[0]/400,
                         df_meteor=df_meteor_t)
            
            # figfile = os.path.join(figpath_3D, 'wind_map_%s.%s' %(dt.strftime('%Y%m%d_%H%M%S'), ext) )
            #
            # plot_3d_maps(X, Y, z,
            #              [u,v,w], 
            #              # title, 
            #              # title_list, 
            #              # vmin_list, 
            #              # vmax_list, 
            #              # xmin, 
            #              # xmax, 
            #              # ymin, 
            #              # ymax, 
            #              filename=figfile, 
            #              # cmap_label, 
            #              # cmap_list, 
            #              # zdecimator,
            #              )
            
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
                ax.plot(10*w[i,j,k], alts, 'g', linestyle=linestyle, alpha=alpha, label=label_w)
            
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
    
    nn = pinn.restore(filename, log_index=log_file)
    
    hmin = nn.lb[3]*1e-3
    hmax = nn.ub[3]*1e-3
    
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
        
        outputs = nn.infer_gradients(t, x_flat, y_flat, z_flat)
        
        ug[:] = np.nan
        
        for i in range(12):
            ug[i][valid_mask] = outputs[:,i]
        
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
    
    tmin = nn.lb[0] + t0
    tmax = nn.ub[0]
    
    xmin = nn.lb[1]
    xmax = nn.ub[1]
    
    ymin = nn.lb[2]
    ymax = nn.ub[2]
    
    zmin = nn.lb[3]
    zmax = nn.ub[3]
    
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
            
            outputs = nn.infer(T, x=X, y=Y, z=Z)
            
            Hu += calc_kspectra(outputs, x=X, y=Y)
            # Hv += calc_spectra(outputs[:,1], x=X, y=Y)
            # Hw += calc_spectra(outputs[:,2], x=X, y=Y)
            
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = dt.strftime('%Y%m%d_%H%M%S')
    
def keograms(filename,
               figpath,
               tstep = None, #min
                xstep = 0.2, #degrees
                ystep = 0.2, #degrees
                zstep = 0.5,
                x0 = None, #degrees
                y0 = None,
                z0 = None, #km
                xrange = None, #degrees
                yrange = None, #degrees
                zrange = None, #km
                vmins = [-100,-100,-15, None],
                vmaxs = [ 100, 100, 15, None],
                ext='pdf',
                type='residuals',
                plot_mean=False,
                log_file=None,
                time_width=0*60*60,
                time_width_mean=1*60*60,
                h_width=0,
                h_width_mean=1,
                cmap='jet',
                t0=1,
                trange=None,#3*60*60,
                histogram=False,
                grads=False,
                grad_titles = [r'Hor. div. [m/s/m]', r'Rel. vort. [m/s/m]', r'Div. [m/s/m]']
                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    # try:
    nn = pinn.restore(filename, log_index=log_file)
    # except:
    #     nn = pinn.PINN()
    #     nn.restore(filename, log_index=log_file)
    
    tmin = nn.lb[0] + t0 
    tmax = nn.ub[0] 
    
    x_lb = nn.lb[2]
    y_lb = nn.lb[3]
    z_lb = nn.lb[1]
    
    x_ub = nn.ub[2]
    y_ub = nn.ub[3]
    z_ub = nn.ub[1]
    
    if trange is not None: tmax = tmin+trange
    if tstep is None: tstep = (tmax - tmin)/(24*6)
    
    if xrange is None: xrange = (x_ub - x_lb)/40e3
    if yrange is None: yrange = (y_ub - y_lb)/90e3
    if zrange is None: zrange = (z_ub - z_lb)*1e-3
    
    lon_ref = nn.lon_ref
    lat_ref = nn.lat_ref
    alt_ref = nn.alt_ref
    
    xmin = np.floor(lon_ref - xrange/2.)
    xmax = np.ceil(lon_ref + xrange/2.)
    
    ymin = np.floor(lat_ref - yrange/2.)
    ymax = np.ceil(lat_ref + yrange/2.)
    
    zmin = np.floor(alt_ref - zrange/2.)
    zmax = np.ceil(alt_ref + zrange/2.)
    
    times   = np.arange(tmin, tmax, tstep)
    x       = np.arange(xmin, xmax, xstep)
    y       = np.arange(ymin, ymax, ystep)
    z       = np.arange(zmin, zmax, zstep)
    
    if type == 'residuals':
        
        X, Y, Z = np.meshgrid(z, x, y, indexing='ij')
        
        shape = X.shape
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        mask = np.where(np.isfinite(X))
        
        u0, v0, w0 = calc_xyz_mean_winds(nn, times, x_flat, y_flat, z_flat, u_shape=shape, valid_mask=mask)
        u0, v0, w0 = calc_mean_winds(times, x, u0, v0, w0,
                                 x_width=time_width_mean,
                                 y_width=h_width_mean )
        
    if x0 is None: x0 = lon_ref
    if y0 is None: y0 = lat_ref
    if z0 is None: z0 = alt_ref
    
    x0       = np.atleast_1d(x0)
    y0       = np.atleast_1d(y0)
    z0       = np.atleast_1d(z0)
    
    idx_x0 = np.argmin(np.abs(x-x0))
    idx_y0 = np.argmin(np.abs(y-y0))
    idx_z0 = np.argmin(np.abs(z-z0))
    
    dt = datetime.datetime.utcfromtimestamp(times[0])
    dt_str = dt.strftime('%Y%m%d') #_%H%M%S')
        
    #X
    T, X, Y, Z = np.meshgrid(times, x, y0, z0, indexing='ij')
    shape = T.shape
    
    t_flat = T.flatten()
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    outputs = nn.infer_gradients(t_flat, x_flat, y_flat, z_flat)
    
    u = outputs[:,0]
    v = outputs[:,1]
    w = outputs[:,2]
    
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
        # u0, v0, w0 = calc_mean_winds(times, x, u, v, w,
        #                          x_width=time_width_mean,
        #                          y_width=0 )
        
        u -= u0[:,None,idx_z0]
        v -= v0[:,None,idx_z0]
        w -= w0[:,None,idx_z0]
    
    figfile_mean = os.path.join(figpath, 'keo_x_%s_%2.1f_%2.1f.%s' %(dt_str, y0[0], z0[0], ext) )
    
    plot_mean_winds(times, x, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='Longitude',
                            histogram=False,
                            )
    
    filename_x = os.path.join(figpath, 'winds_%s_TvsX_[%+03.1fºE,%+03.1fkm].h5' %(dt_str, y0[0], z0[0]) )
    
    save_h5file(z0[0], x, y0[0],
                u, v, w,
                filename=filename_x,
                times=times)
    
    if grads == True:
        u_x = outputs[:,3]
        v_x = outputs[:,4]
        w_x = outputs[:,5]
        
        u_y = outputs[:,6]
        v_y = outputs[:,7]
        w_y = outputs[:,8]
        
        u_z = outputs[:,9]
        v_z = outputs[:,10]
        w_z = outputs[:,11]
        
        div = u_x + v_y
        vor = v_x - u_y
        w_z = div + w_z
        
        div = div.reshape(shape)
        vor = vor.reshape(shape)
        w_z = w_z.reshape(shape)
        
        div = np.mean(div, axis=(2,3))
        vor = np.mean(vor, axis=(2,3))
        w_z = np.mean(w_z, axis=(2,3))
        
        div, vor, w_z = calc_mean_winds(times, x, div, vor, w_z,
                                     x_width=time_width,
                                     y_width=0)
        
        if type == 'residuals':
            div0, vor0, w_z0 = calc_mean_winds(times, x, div, vor, w_z,
                                     x_width=time_width_mean,
                                     y_width=0 )
            
            div -= div0
            vor -= vor0
            w_z -= w_z0
        
        figfile = os.path.join(figpath, 'keo_x_grad_%s_%2.1f_%2.1f.%s' %(dt_str, y0[0], z0[0], ext) )
        
        
        plot_mean_winds(times, x, div, vor, w_z,
                                titles=grad_titles,
                                figfile=figfile,
                                vmins=vmins[-3:],
                                vmaxs=vmaxs[-3:],
                                cmap=cmap,
                                ylabel='Longitude',
                                histogram=False,
                                )

    #Y
    figfile_mean = os.path.join(figpath, 'keo_y_%s_%2.1f_%2.1f.%s' %(dt_str, x0[0], z0[0], ext) )
    
    T, X, Y, Z = np.meshgrid(times, x0, y, z0, indexing='ij')
    shape = T.shape
    
    t_flat = T.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    x_flat = X.flatten()
    
    outputs = nn.infer_gradients(t_flat, x_flat, y_flat, z_flat)
    
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
        # u0, v0, w0 = calc_mean_winds(times, y, u, v, w,
        #                          x_width=time_width_mean,
        #                          y_width=0)
        
        u -= u0[:,None,idx_z0]
        v -= v0[:,None,idx_z0]
        w -= w0[:,None,idx_z0]
        
    plot_mean_winds(times, y, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='Latitude',
                            histogram=False,
                            )
    
    filename_y = os.path.join(figpath, 'winds_%s_TvsY_[%+03.1fºN,%+03.1fkm].h5' %(dt_str, x0[0], z0[0]) )
    
    
    save_h5file(z0[0], x0[0], y,
                u, v, w,
                filename=filename_y,
                times=times)
    
    if grads == True:
        
        u_x = outputs[:,3]
        v_x = outputs[:,4]
        w_x = outputs[:,5]
        
        u_y = outputs[:,6]
        v_y = outputs[:,7]
        w_y = outputs[:,8]
        
        u_z = outputs[:,9]
        v_z = outputs[:,10]
        w_z = outputs[:,11]
        
        div = u_x + v_y
        vor = v_x - u_y
        w_z = div + w_z
        
        div = div.reshape(shape)
        vor = vor.reshape(shape)
        w_z = w_z.reshape(shape)
        
        div = np.mean(div, axis=(1,3))
        vor = np.mean(vor, axis=(1,3))
        w_z = np.mean(w_z, axis=(1,3))
        
        div, vor, w_z = calc_mean_winds(times, y, div, vor, w_z,
                                     x_width=time_width,
                                     y_width=0)
        
        if type == 'residuals':
            div0, vor0, w_z0 = calc_mean_winds(times, y, div, vor, w_z,
                                     x_width=time_width_mean,
                                     y_width=0 )
            
            div -= div0
            vor -= vor0
            w_z -= w_z0
        
        figfile = os.path.join(figpath, 'keo_y_grad_%s_%2.1f_%2.1f.%s' %(dt_str, x0[0], z0[0], ext) )
        
        plot_mean_winds(times, y, div, vor, w_z,
                                titles=grad_titles,
                                figfile=figfile,
                                vmins=vmins[-3:],
                                vmaxs=vmaxs[-3:],
                                cmap=cmap,
                                ylabel='Latitude',
                                histogram=False,
                                )
        
    #Z
    figfile_mean = os.path.join(figpath, 'keo_z_%s_%2.1f_%2.1f.%s' %(dt_str, x0[0], y0[0], ext) )
    
    T, X, Y, Z = np.meshgrid(times, x0, y0, z, indexing='ij')
    shape = T.shape
    
    t_flat = T.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    x_flat = X.flatten()
    
    outputs = nn.infer_gradients(t_flat, x_flat, y_flat, z_flat)
    
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
                                 y_width=h_width)
    
    if type == 'residuals':
        # u0, v0, w0 = calc_mean_winds(times, z, u, v, w,
        #                          x_width=time_width_mean,
        #                          y_width=h_width_mean)
        
        u -= u0
        v -= v0
        w -= w0
        
    plot_mean_winds(times, z, u, v, w,
                            figfile=figfile_mean,
                            vmins=vmins, vmaxs=vmaxs,
                            cmap=cmap,
                            ylabel='Altitude (km)',
                            histogram=histogram,
                            )
    
    filename_z = os.path.join(figpath, 'winds_%s_TvsZ_[%+03.1fºE,%+03.1fºN].h5' %(dt_str, x0[0], y0[0]) )
    
    
    save_h5file(z, x0[0], y0[0],
                u, v, w,
                filename=filename_z,
                times=times)
    
    if grads == True:
        
        u_x = outputs[:,3]
        v_x = outputs[:,4]
        w_x = outputs[:,5]
        
        u_y = outputs[:,6]
        v_y = outputs[:,7]
        w_y = outputs[:,8]
        
        u_z = outputs[:,9]
        v_z = outputs[:,10]
        w_z = outputs[:,11]
        
        div = u_x + v_y
        vor = v_x - u_y
        w_z = div + w_z
        
        div = div.reshape(shape)
        vor = vor.reshape(shape)
        w_z = w_z.reshape(shape)
        
        div = np.mean(div, axis=(1,2))
        vor = np.mean(vor, axis=(1,2))
        w_z = np.mean(w_z, axis=(1,2))
        
        div, vor, w_z = calc_mean_winds(times, z, div, vor, w_z,
                                     x_width=time_width,
                                     y_width=h_width)
        
        if type == 'residuals':
            div0, vor0, w_z0 = calc_mean_winds(times, z, div, vor, w_z,
                                     x_width=time_width_mean,
                                     y_width=h_width_mean )
            
            div -= div0
            vor -= vor0
            w_z -= w_z0
        
        figfile = os.path.join(figpath, 'keo_z_grad_%s_%2.1f_%2.1f.%s' %(dt_str, x0[0], y0[0], ext) )
        
        plot_mean_winds(times, z, div, vor, w_z,
                                titles=grad_titles,
                                figfile=figfile,
                                vmins=vmins[-3:],
                                vmaxs=vmaxs[-3:],
                                cmap=cmap,
                                ylabel='Altitude',
                                histogram=False,
                                )
    
        filename_z = os.path.join(figpath, 'wind_grads_%s_TvsZ_[%+03.1fºE,%+03.1fºN].h5' %(dt_str, x0[0], y0[0]) )
    
    
        save_h5file(z, x0[0], y0[0],
                    div, vor, w_z,
                    filename=filename_z,
                    times=times,
                    labels=["div", "vor", "wz"])
    
def mean_winds(filename,
               figpath,
               tstep = None, #min
                xstep = 0.2, #degrees
                ystep = 0.2, #degrees
                zstep = 1, #km
                xrange = None, #degrees
                yrange = None, #degrees
                zrange = None, #km
                trange = None,
                vmins = [-100,-100,-15, None],
                vmaxs = [ 100, 100, 15, None],
                ext='pdf',
                type='residuals',
                plot_mean=False,
                log_file=None,
                plot_fields=False,
                plot_vor_div=False,
                plot_wind_vectors=True,
                df_meteor=None,
                t0=1,
                cmap='RdBu_r',
                time_width=0*60*60,
                time_width_mean=1*60*60,
                h_width=0,
                h_width_mean=1,
                ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    try:
        nn = pinn.restore(filename, log_index=log_file)
    except:
        nn = pinn.PINN()
        nn.restore(filename, log_index=log_file)
        
    tmin = nn.lb[0] + t0
    tmax = nn.ub[0]
    
    x_lb = nn.lb[2]
    y_lb = nn.lb[3]
    z_lb = nn.lb[1]
    
    x_ub = nn.ub[2]
    y_ub = nn.ub[3]
    z_ub = nn.ub[1]
    
    
    if trange is not None: tmax = tmin+trange
    if tstep is None: tstep = (tmax - tmin)/(24*6)
    
    if xrange is None: xrange = (x_ub - x_lb)/40e3
    if yrange is None: yrange = (y_ub - y_lb)/90e3
    if zrange is None: zrange = (z_ub - z_lb)*1e-3
    
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
    
    mask = nn.invalid_mask(tmax*np.ones_like(X), X, Y, Z)
    
    x_flat = X[mask]
    y_flat = Y[mask]
    z_flat = Z[mask]
    
    dt = datetime.datetime.utcfromtimestamp(tmin)
    dt_str = "%s_%d_%d" %(dt.strftime('%Y%m%d_%H%M%S'), time_width_mean/3600, h_width_mean)
    
    ug0 = get_mean_winds_and_grads(nn, times, x_flat, y_flat, z_flat, X.shape, mask)
    
    ug0 = calc_mean_winds(times, z, ug0, x_width=time_width, y_width=h_width)
    
    if type == 'residuals':
        u4h = calc_mean_winds(times, z, ug0, x_width=time_width_mean, y_width=h_width_mean)
        ug0 = ug0 - u4h
    # else:
    #     ug0 = u4h
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
        
        #Divergence
        ug0[5] = ug0[3] + ug0[7]
        
        #Vorticity
        ug0[8] = ug0[4] - ug0[6]
        
        plot_mean_winds_grads(times, z, ug0,
                        figfile=figfile_mean,
                        vmins=vmins,
                        vmaxs=vmaxs,
                        cmap=cmap)
        
        figfile_mean = os.path.join(figpath, 'mean_wind_%s.%s' %(dt_str, ext) ) 
        plot_mean_winds(times, z,
                        ug0[0], ug0[1], ug0[2],
                        figfile=figfile_mean,
                        vmins=vmins,
                        vmaxs=vmaxs,
                        cmap=cmap,
                        )
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
    
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/Germany", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815", help='Data path')
    parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/Argentina/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/Norway/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/Condor/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/JRO/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/Piura/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/NewMexico/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/", help='Data path')
    
    parser.add_argument('-m', '--model', dest='model', default=None, help='neural network model')
    parser.add_argument('-s', '--subfolder', dest='subfolder', default="nnRESPINN_15.11", help='subfolder where the neural network model is stored')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='figures extension')
    parser.add_argument('-t', '--type', dest='type', default='wind', help='plot type. Either "residuals" or "full" wind')
    parser.add_argument('-l', '--log-file', dest='log_file', default=None, help='select the i-th weights file from the log folder')
    
    # parser.add_argument('--meteor-path', dest='mpath', default=None, help='Data path')
    
    # parser.add_argument('--meteor-path', dest='mpath', default='/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815/ICON_+00+70+90', help='Data path')
    # parser.add_argument('--meteor-path', dest='mpath', default='/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/DNSx10_+12+53+91/', help='Data path')
    parser.add_argument('--meteor-path', dest='mpath', default='/Users/radar/Data/IAP/SIMONe/Norway/ExtremeEvent/', help='Data path')
    
    
    args = parser.parse_args()
    
    path  = args.dpath
    mpath = args.mpath
    model_name = args.model
    ext = args.ext
    type = args.type
    log_file = args.log_file
    
    xrange = 12
    yrange = 4
    
    cmap = 'seismic'
    
    t0 = 1#6*60*60
    trange = None#1.5*60*60
    
    # t0 = 16*60*60
    # trange = 1.5*60*60
    
    time_width_mean = 1*60*60
    h_width_mean = 1
    
    # vmins = np.array([-100, -100, -10, -1.0, -1.0, -0.1, -1.0, -1.0, -0.1, -25, -25, -5])
    
    vmins = np.array([-100, -100, -5, -4.0e-4, -4.0e-4, -4e-4, -4.0e-4, -4.0e-4, -4e-4, -2e-3, -2e-3, -2e-3])
    # vmins = np.array([-20, -20, -1, -2.0e-3, -2.0e-3, -2e-3, -2.0e-3, -2.0e-3, -2e-3, -25e-3, -25e-3, -2e-3])
    # vmins = 0.1*np.array([-80, -80, -5, -1.0e-3, -1.0e-3, -1e-3, -1.0e-3, -1.0e-3, -1e-3, -30e-3, -30e-3, -1e-3])
     #[ 150,  150,  10,  0.4,  0.4,  0.04,  0.4,  0.4,  0.04,  25,  25,  2.5]
    
    # if type == 'residuals':
    #     vmins[0:3] = [-50, -50, -1]
    
    vmaxs = -vmins
    
    path_PINN = os.path.join(path, "winds", args.subfolder)
    
    if model_name is None:
        models = glob.glob1(path_PINN, 'h*[!s].h5')
        models = sorted(models)
    else:
        models = [  model_name ]
            
    for model in models[:]:
        
        id_name = model[-11:-3]
        filename = os.path.join(path_PINN, model)
        
        if not os.path.isfile(filename):
            continue
        
        figpath = os.path.join(path_PINN, "plots")
                               
        if not os.path.exists(figpath):
            os.mkdir(figpath)
            
        figpath = os.path.join(figpath, '%s' %model[:-3])
        
        if not os.path.exists(figpath):
            os.mkdir(figpath)
            
        figpath_type = os.path.join(figpath, '%s_%s_%02d' %(type,log_file, t0/3600) ) #+os.path.splitext(model_name)[0])
        
        
        # 69.45 N, 15.83 E, around 90 km
        
        # altitude_profiles(filename, figpath_type,
        #                   datetimes=[
        #                              datetime.datetime(2023,3,23,21,0,0),
        #                              # datetime.datetime(2023,3,23,20,0,0),
        #                              datetime.datetime(2023,3,23,20,30,0),
        #                              datetime.datetime(2023,3,23,21,30,0)
        #                              ],
        #                   lats=69.45,
        #                   lons=15.83,
        #                   delta_h=0.25,
        #                   log_file=log_file)
        
        keograms(filename,
                 figpath_type, ext=ext,
                 # ystep=0.1,
                # z0=np.arange(90,92,0.25), #Vortex
                 xstep=0.05,
                 ystep=0.05,
                 zstep=0.25,
                 tstep=5*60,
                 xrange=xrange,
                 yrange=yrange,
                 time_width=0*60,
                 h_width=0,
                 time_width_mean=time_width_mean,
                 h_width_mean=h_width_mean,
                 vmins=vmins,
                 vmaxs=vmaxs,
                 type=type,
                 log_file=log_file,
                 cmap=cmap,
                 t0=t0,
                 trange=trange,
                 #MAARSY
                # x0=16.04,
                # y0=69.25,
                # y0=70,
                # x0=-106.5,
                # y0=33.5,
                # x0=17.5,
                # y0=70,
                # z0=85.0,
                ##Vortex,
                # x0=15.75,
                # y0=69.45,
                # y0=70.35,
                # x0=14.25,
                histogram=False,
                # grads=True,
               )
        
        continue
        
        mean_winds(filename, figpath_type,
                   ext=ext,
                    type=type,
                    plot_mean=True,
                    t0=t0,
                    xrange=xrange,
                    yrange=yrange,
                    trange=trange,
                    tstep=15*60,
                    # xmin=-30,
                    # ymin=-30,
                    xstep=1.0,
                    ystep=1.0,
                    zstep=0.5,
                    # zmin=84,
                    log_file=log_file,
                    vmins=vmins,
                    vmaxs=vmaxs,
                    cmap=cmap,
                    time_width=0*60,
                    h_width=0,
                    h_width_mean=h_width_mean,
                    time_width_mean=time_width_mean,
                   )
        #
        # continue
            
        getting_and_plotting_winds(filename, figpath_type, ext=ext,
                                    type=type,
                                    plot_mean=False,
                                    plot_fields=False,
                                    plot_vor_div=False,
                                    plot_wind_vectors=True,
                                    save=True,
                                    # xmin=-30,
                                    # ymin=-30,
                                    xrange=xrange,
                                    yrange=yrange,
                                    zstep=1,
                                    zrange=6,
                                    zmin=83,
                                    log_file=log_file,
                                    vmins=vmins,
                                    vmaxs=vmaxs,
                                    t0=t0,
                                    calc_grads=False,
                                    meteor_path=mpath,
                                   )
        
        # getting_and_plotting_mean_winds(filename, figpath_type, ext=ext,
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