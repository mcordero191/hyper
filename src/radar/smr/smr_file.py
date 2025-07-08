'''
Created on 2 Sep 2022

@author: mcordero
'''
import os, glob
import h5py
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import matplotlib.colors as colors

from sklearn.preprocessing import LabelEncoder

from radar.smr.standard_methods import mean_wind_grad

from georeference.geo_coordinates import lla2enu
from utils.clustering import hierarchical_cluster

from utils.histograms import ax_2dhist_simple
from utils.plotting import epoch2num
from utils.plotting import plot_mean_winds

stations = {}
stations['collm']       = [51.31, 13.0, 10e-3]
stations['bornim']      = [52.44, 13.02, 10e-3]
stations['bor']         = [52.44, 13.02, 10e-3]
stations['ntz']         = [53.33, 13.07, 10e-3]
stations['jruh']        = [54.63, 13.37, 10e-3]
stations['kborn']       = [54.10, 11.8, 10e-3]
stations['breege']      = [54.62, 13.37, 10e-3]
stations['guderup']     = [55.00, 9.86, 10e-3]
stations['mechelsdorf'] = [54.12, 11.67, 10e-3]
stations['salzwedel']   = [52.75, 10.90, 10e-3]
stations['andenes']     = [69.27, 16.04, 10e-3]
stations['tromso']      = [69.58, 19.22, 10e-3]
stations['alta']        = [69.96, 23.29, 10e-3]
stations['straumen']    = [67.40, 15.6, 10e-3]

def get_xyz_bounds(x, y, z, factor=3.0):
    
    from scipy import stats
    
    x0 = np.median(x)
    y0 = np.median(y)
    z0 = np.median(z)
    
    xsigma = stats.median_abs_deviation(x)
    ysigma = stats.median_abs_deviation(y)
    zsigma = stats.median_abs_deviation(z)
    
    mask_x = np.abs(x - x0) < factor*xsigma
    mask_y = np.abs(y - y0) < factor*ysigma
    mask_z = np.abs(z - z0) < factor*zsigma
    
    valid = (mask_x & mask_y & mask_z)
    
    return (valid)
    
    # xbins = np.arange( np.min(x), np.max(x)+30e3, 30e3)
    # ybins = np.arange( np.min(y), np.max(y)+30e3, 30e3)
    # zbins = np.arange( np.min(z), np.max(z)+1e3, 1e3)
    #
    # hist, edges = np.histogramdd( np.array([x, y]).T, (xbins, ybins) )
    #
    # valid = np.where(hist >= 30)
    #
    # idx0 = np.min(valid, axis=1)
    # idx1 = np.max(valid, axis=1)
    #
    # xmin = edges[0][idx0[0]]
    # xmax = edges[0][idx1[0]+1]
    #
    # ymin = edges[1][idx0[1]]
    # ymax = edges[1][idx1[1]+1]
    #
    # hist, edges = np.histogramdd( np.array([z,]).T, (zbins,) )
    #
    # valid = np.where(hist >= 30)
    #
    # idx0 = np.min(valid, axis=1)
    # idx1 = np.max(valid, axis=1)
    #
    # zmin = edges[0][idx0[0]]
    # zmax = edges[0][idx1[0]+1]
    #
    #
    # return( xmax-xmin, ymax-ymin, zmax-zmin )

from scipy.spatial import KDTree

def remove_close_clusters(df, spatial_threshold=0.5e3, temporal_threshold=5):
    """
    Removes points that are too close in both time and space from a DataFrame.

    Parameters:
    - df: pandas DataFrame with columns 't', 'x', 'y', 'z'
    - spatial_threshold: minimum allowed spatial distance between points
    - temporal_threshold: minimum allowed temporal distance between points

    Returns:
    - Filtered pandas DataFrame with points that are not clustered.
    """
    
    # Sort the DataFrame by time
    df_sorted = df.sort_values(by='times').reset_index(drop=True)
    
    # Extract the spatial coordinates and time values
    coordinates = df_sorted[['x', 'y', 'z']].to_numpy()
    times = df_sorted['times'].to_numpy()
    links = df_sorted['link'].to_numpy()
    
    # Use KDTree for efficient spatial queries
    spatial_tree = KDTree(coordinates)
    
    keep_indices = []
    discard = np.zeros(len(df_sorted), dtype=bool)
    
    for i in range(len(df_sorted)):
        
        if discard[i]:
            continue
        
        # Get indices of nearby points within the spatial threshold
        spatial_neighbors = spatial_tree.query_ball_point(coordinates[i], spatial_threshold)
        
        for neighbor in spatial_neighbors:
            
            if neighbor == i:
                continue
            
            if links[neighbor] != links[i]:
                continue
            
            # Check the temporal distance
            temporal_distance = abs(times[i] - times[neighbor])
            
            if temporal_distance < temporal_threshold:
                discard[neighbor] = True
                # discard[i] = True
                break
        
        if discard[i]:
            continue
        
        keep_indices.append(i)
    
    # df_removed = df_sorted.drop(keep_indices)
    #
    # t0 = df_sorted['times']
    # x0 = df_sorted['z']
    #
    # t = df_removed['times']
    # x = df_removed['z']
    #
    # plt.plot(t0, x0, "bo", alpha=0.3)
    # plt.plot(t, x, "rx", alpha=0.3)
    # plt.show()
    
    # Return filtered DataFrame
    return df_sorted.iloc[keep_indices]

    
def filter_data(df, tini=0, dt=24,
                sevenfold=False,
                overlapping_time = 2*60*60,
                central_date=None,
                ena_clustering=1,
                **kwargs
                ):
    
    if central_date is not None:
        tbase = central_date
    elif tini >= 0:
        tbase = pd.to_datetime( df['t'].min() )
    else:
        tbase = pd.to_datetime( df['t'].max() )
        
    t = tbase + pd.to_timedelta(tini, unit='h') # Center time for reconstruction
    
    tmin = t - pd.to_timedelta(overlapping_time, unit='s')
    tmax = t + pd.to_timedelta(dt*60*60 + overlapping_time, unit='s')        
    
    ##################################################
    
    
    valid  = (df['t'] >= tmin) & (df['t'] <= tmax)
    df = df[valid]
    
    if df.size == 0:
        return(df)
    
    ##################################################
    
    if not(sevenfold):
        #Filter synthetic measurements
        if 'SMR_like' in df.keys():
            valid = (df['SMR_like'] == 1) 
            df = df[valid]
            
    ##################################################
    
    dxy = np.sqrt( df['dcosx']**2 + df['dcosy']**2 )
    zenith = np.arcsin(dxy)*180/np.pi
    df = df[zenith < 70]

    if df.size == 0:
        return(df)
    
    ########################
    
    if ena_clustering:
        
        links = df['link'].values
        
        le = LabelEncoder()
        le.fit(links)
        nlinks = 1#len(le.classes_)
        
        for _ in range(nlinks):
        #########################
        ### Outliers removal ##########
            
            dxy = np.sqrt( df['dcosx'].values**2 + df['dcosy'].values**2 )
            zenith = np.arcsin(dxy)*180/np.pi
            
            if np.max(zenith) <= 0:
                break
            
            dop = df['dops'].values
            links = df['link'].values
            ids = le.transform(links)
            
            X = np.stack([  dop,
                            # kz,
                            zenith,
                            ids,
                          ],
                          axis=1)
            
            valid = hierarchical_cluster(X)
            
            if np.count_nonzero(~valid) == 0:
                break
            
            # print("Removing %d meteor samples ..." %(df.size - valid.size))
            df = df[valid]
        
        if df.size == 0:
            return(df)
    #####################################################
    
    # x = df['x']
    # y = df['y']
    # z = df['z']
    #
    # attrs = df.attrs
    # xmid = attrs['lon_center']
    # ymid = attrs['lat_center']
    # zmid = attrs['alt_center']
    
    # if lon_center is not None: xmid = lon_center
    # else: xmid = np.round( x.mean(), 1)
    #
    # if lat_center is not None: ymid = lat_center
    # else: ymid = np.round( y.mean(), 1)
    #
    # xmid = 0
    # ymid = 0
    # zmid = np.round( z.mean(), 1)
    #
    # # print('Filter middle point:', xmid, ymid, zmid)
    #
    # dlon0, dlat0, dh0 = get_xyz_bounds(x, y, z)
    #
    # if (dlon is None): dlon = dlon0
    # if (dlat is None): dlat = dlat0
    # if (dh is None)  : dh   = dh0
    #
    # # Set boundary
    # xmin = xmid - dlon/2
    # xmax = xmid + dlon/2
    # ymin = ymid - dlat/2
    # ymax = ymid + dlat/2
    # zmin = zmid - dh/2
    # zmax = zmid + dh/2
    #
    # valid  = (x >= xmin) & (x <= xmax)
    # valid &= (y >= ymin) & (y <= ymax) 
    # valid &= (z >= zmin) & (z <= zmax)
    
        #tbase = pd.to_datetime(dfraw['t'][0].date())
    
    lons = df['lons']
    lats = df['lats']
    alts = df['heights']
    
    valid = get_xyz_bounds(lons, lats, alts)
    
    df = df[valid]
    
    ########################
    
    # plot_hist(df)
    
    ########################
    
    return(df)


def plot_delta_sampling(df, path):

    from scipy.spatial import distance
    
    t = df['times'].values
    
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    
    X = np.vstack((x,y,z)).T

    d = distance.cdist(X, X, 'euclidean')
    ind = np.triu_indices_from(d)
    
    dr = d[ind]
    
    t = t.reshape(-1,1)
    
    d = distance.cdist(t, t, 'euclidean')
    ind = np.triu_indices_from(d)
    
    dt = d[ind]
    
    ax = plt.subplot(1,1,1)
    ax_2dhist_simple(dt/60, dr*1e-3, ax)
    
    plt.xlabel('Delta time (min)')
    plt.ylabel('Distance (km)')
    
    figname = os.path.join(path, 'sampling_dt_dr.png')
    plt.savefig(figname)
    plt.close()
        
def plot_spatial_sampling(df, path, suffix='', lla=True, cmap='jet',
                        label_colorbar='meteor counts',
                        vmin=1, vmax=1e3,
                        bins=15):
    
    norm=colors.LogNorm(vmin, vmax)
    
    vmin = None
    vmax = None
    
    t = df['times'].values
    
    total_counts = len(t)
    
    
    t_num = epoch2num(t)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    formatter = mdates.ConciseDateFormatter(locator)
    
    tlabel = 'UTC'
    
    if lla:
        x = df['lons'].values
        y = df['lats'].values
        z = df['heights'].values
        
        xlabel = 'Longitude'
        ylabel = 'Latitude'
        zlabel = 'Altitude (km)'
    else:
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        
        xlabel = 'X (m)'
        ylabel = 'Y (m)'
        zlabel = 'Z (m)'
    
    N = int( (t.max() - t.min())/(10*60) )//30
    N = max(1,N)
    tstep = N*10*60
    
    bins_t = np.arange(t.min(), t.max(), tstep)
    bins_t_num = epoch2num(bins_t)
    
    bins_x = bins #np.arange(x.min()-0.03, x.max(), 0.06*5)
    bins_y = bins #np.arange(y.min()-0.03, y.max(), 0.06*5)
    bins_z = bins #np.arange(z.min(), z.max()+2, 2)
    
    filename = os.path.join(path, 'dist_%s.png' %(suffix) )
    
    # _, axs = plt.subplots(2, 3, figsize=(12,10))
    
    _ = plt.figure(figsize=(12,10))
    plt.suptitle('Meteor counts: %d' %total_counts)
    
    ax0 = plt.subplot2grid((3,3), (0,0), rowspan=1, colspan=2)
    ax1 = plt.subplot2grid((3,3), (1,0), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid((3,3), (2,0), rowspan=1, colspan=2)
    
    ax3 = plt.subplot2grid((3,3), (2,2), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((3,3), (0,2), rowspan=1, colspan=1)
    ax5 = plt.subplot2grid((3,3), (1,2), rowspan=1, colspan=1)
    
    
    # ax0 = axs[0,1]
    
    h,_,_,im = ax3.hist2d(x, y, bins=(bins_x, bins_y), cmap=cmap,
                          # vmin=vmin, vmax=vmax,
                          norm=norm
                          )
    # ax3.scatter(x, y, alpha=0.05)
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    # ax3.grid()
    plt.colorbar(im, ax=ax3, label=label_colorbar)
    
    if vmin is None: vmin = np.min(h)
    if vmax is None: vmax = np.max(h)
    
    h, _, _, im = ax0.hist2d(t_num, x, bins=(bins_t_num, bins_x), cmap=cmap,
                          # vmin=vmin, vmax=vmax,
                          norm=norm
                          )
    
    ax0.set_xlabel(tlabel)
    ax0.set_ylabel(xlabel)
    # ax0.grid()
    # plt.colorbar(im, ax=ax0, label=label_colorbar)
    
    ax0.xaxis.set_major_locator(locator)
    ax0.xaxis.set_major_formatter(formatter)
    
    
    
    # ax1 = axs[0,1]
    
    _,_,_,im = ax1.hist2d(t_num, y, bins=(bins_t_num, bins_y), cmap=cmap,
                          # vmin=vmin, vmax=vmax,
                          norm=norm
                          )
    ax1.set_xlabel(tlabel)
    ax1.set_ylabel(ylabel)
    # ax1.grid()
    # plt.colorbar(im, ax=ax1, label=label_colorbar)
    
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    
    # ax2 = axs[0,2]
    
    _,_,_,im = ax2.hist2d(t_num, z, bins=(bins_t_num, bins_z), cmap=cmap,
                          # vmin=vmin, vmax=vmax,
                          norm=norm
                          )
    ax2.set_xlabel(tlabel)
    ax2.set_ylabel(zlabel)
    # ax2.grid()
    # plt.colorbar(im, ax=ax2, label=label_colorbar)
    
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    
    
    # ax3 = axs[1,0]
    
    
    
    
    # ax4 = axs[1,1]
    
    _,_,_,im = ax4.hist2d(x, z, bins=(bins_x, bins_z),
                          cmap=cmap,
                          # vmin=vmin, vmax=vmax,
                          norm=norm
                          )
    # ax4.scatter(x, z, alpha=0.05)
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel(zlabel)
    # ax4.grid()
    plt.colorbar(im, ax=ax4, label=label_colorbar)
    
    # ax5 = axs[1,2]
    
    _,_,_,im = ax5.hist2d(y,z, bins=(bins_y, bins_z), cmap=cmap,
                          # vmin=vmin, vmax=vmax,
                          norm=norm
                          )
    # ax5.scatter(y, z, alpha=0.05)
    ax5.set_xlabel(ylabel)
    ax5.set_ylabel(zlabel)
    # ax5.grid()
    plt.colorbar(im, ax=ax5, label=label_colorbar)
    
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close("all")

def plot_2point_sampling(df, path,
                         suffix='',
                         lla=False,
                         cmap='inferno',
                         vmin=None,
                         vmax=None,
                         bins=20,
                         dtype=np.float16):
    
    N = -1
    
    t = df['times'].values
    t = (t-t.min()).astype(dtype)[:N]
    
    if lla:
        x = df['lons'].values.astype(dtype)[:N]
        # y = df['lats'].values.astype(dtype)[:N]
        # z = df['heights'].values.astype(dtype)[:N]
        #
        # xlabel = 'Longitude'
        # ylabel = 'Latitude'
        # zlabel = 'Altitude (km)'
    else:
        x = df['x'].values.astype(dtype)[:N]*1e-3
        # y = df['y'].values.astype(dtype)[:N]*1e-3
        # z = df['z'].values.astype(dtype)[:N]*1e-3
        #
        # xlabel = 'X (m)'
        # ylabel = 'Y (m)'
        # zlabel = 'Z (m)'
        
    #
    # indices = np.arange(t.shape[0])
    #
    # dts = []
    # dxs = []
    # dys = []
    # dzs = []
    #
    # for i,j in itertools.combinations(indices, 2):
    #     dt = t[j] - t[i]
    #     dx = x[j] - x[i]
    #     dy = y[j] - y[i]
    #     dz = z[j] - z[i]
    #
    #     dts.append(dt)
    #     dxs.append(dx)
    #     dys.append(dy)
    #     dzs.append(dz)
    
    idx = np.tril_indices(n=t.shape[0])
        
    df = {}
    df['dt'] = np.abs( (t[:,None] - t[None,:])[idx] )
    df['dx'] = np.abs( (x[:,None] - x[None,:])[idx] )
    # df['dy'] = np.abs(y[:,None] - y[None,:])
    # df['dz'] = np.abs(z[:,None] - z[None,:])
    
    df = pd.DataFrame.from_dict(df)
    
    import plotly.express as px
    
    filename = os.path.join(path, '%s_%d.png' %(suffix, lla) )
    
    fig = px.density_heatmap(df, x="dt", y="dx",
                             marginal_x="histogram",
                             marginal_y="histogram",
                             nbinsx=30,
                             nbinsy=30,
                             )
    
    fig.write_image(filename, format='png') 
    
def plot_Doppler_sampling(df_unfiltered, df, path="./",
                          lla=True, cmap='jet',
                          label_colorbar='meteor counts',
                          vmin=-10, vmax=10,
                          k_min=-1.3, k_max=1.3,
                          umin=-50, umax=50,
                          bins=40,
                          density=False,
                          noise=None,
                          suffix="",
                          ):
    
    # bins_u = np.linspace(umin, umax, bins)
    # bins_w = np.linspace(vmin, vmax, bins)
    
    
    # t = df['times'].values
    
    # t_num = epoch2num(t)
    # locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    # formatter = mdates.ConciseDateFormatter(locator)
    
    # tlabel = 'UTC'
    
    kx = df['braggs_x'].values                     #[N]
    ky = df['braggs_y'].values
    kz = df['braggs_z'].values
    
    dops = df['dops'].values
    derr = df['dop_errs'].values
    
    dops_unf = df_unfiltered['dops'].values
    # derr_unf = df_unfiltered['dop_errs'].values
    
    std = np.std(dops)
    std_unf = np.std(dops_unf)
    
    # u = df['u'].values                              #[N]
    # v = df['v'].values                              #[N]
    # w = df['w'].values 
    
    dxy = np.sqrt( df['dcosx'].values**2 + df['dcosy'].values**2 )
    zenith = np.arcsin(dxy)*180/np.pi
    
    filename = os.path.join(path, 'sampling_Dopp_%s.png' %(suffix) )
    
    _, axs = plt.subplots(2, 3, figsize=(9,6))
    
    ###### Doppler ###
    
    ax0 = axs[0,0]
    
    ax0.set_title(r'Original dataset: f (std=%3.2f)' %std_unf)
    ax0.hist(dops_unf, bins,
                    label='Doopler'
                     )    
    
    # ax0.hist(derr_unf, bins,
    #             label='Noise',
    #             color='r',
    #             alpha=0.3,
    #              ) 
        
    ax0.set_ylabel('Counts')
    ax0.set_xlabel('Doppler (Hz)')
    # ax0.set_xlim(vmin, vmax)
    ax0.set_yscale('log')
    ax0.grid(True)
    
    
    ax0 = axs[0,1]
    
    ax0.set_title(r'Filtered data: f (std=%3.2f)' %std)
    ax0.hist(dops, bins,
                    label='Doopler'
                     )    
    
    # ax0.hist(derr, bins,
    #             label='Noise',
    #             color='r',
    #             alpha=0.3,
    #              ) 
        
    ax0.set_xlabel('Doppler (Hz)')
    # ax0.set_xlim(vmin, vmax)
    ax0.set_yscale('log')
    ax0.grid(True)
    
    ###########################
    ax0 = axs[0,2]
    
    ax_2dhist_simple(derr, zenith,
                     ax0,
                     bins,
                     log_scale=True,
                     )
    
    ax0.set_ylabel('Zenith (deg)')
    ax0.set_xlabel('Doppler error (Hz)')
    
    
    # ax0 = axs[0,1]
    # ax0.set_title(r'$u*k_x$')
    # ax0.hist(up, bins_w) #, label=r'$u*k_x$')
    # ax0.set_xlabel('m/s')
    # # ax0.set_xlim(vmin, vmax)
    # ax0.grid(True)
    #
    # ax0 = axs[0,2]
    # ax0.set_title(r'$v*k_y$')
    # ax0.hist(vp, bins_w) #, label=r'$u*k_x$')
    # ax0.set_xlabel('m/s')
    # # ax0.set_xlim(vmin, vmax)
    # ax0.grid(True)
    #
    # ax0 = axs[0,3]
    # ax0.set_title(r'$w*k_z$')
    # ax0.hist(wp, bins_w) #, label=r'$u*k_x$')
    # ax0.set_xlabel('m/s')
    # # ax0.set_xlim(vmin, vmax)
    # ax0.grid(True)
    
    
    
    ax0 = axs[1,0]
    ax0.set_title(r'Bragg vector')
    ax0.hist(kx, bins,
             density=density,
                     # xmin, xmax,
                     # ymin, ymax,
                     # vmin, vmax,
                     # cmap,
                     # normalization_x,
                     )
    ax0.set_xlabel(r'$k_x$')
    ax0.set_ylabel('Counts')
    ax0.grid(True)
    
    ax0 = axs[1,1]
    
    ax0.set_title(r'Bragg vector')
    ax0.hist(ky, bins,
             density=density,
                     # xmin, xmax,
                     # ymin, ymax,
                     # vmin, vmax,
                     # cmap,
                     # normalization_x,
                     )
    ax0.set_xlabel(r'$k_y$')
    ax0.grid(True)
    
    ax0 = axs[1,2]
    
    ax0.set_title(r'Bragg vector')
    ax0.hist(kz, bins,
             density=density,
                     # xmin, xmax,
                     # ymin, ymax,
                     # vmin, vmax,
                     # cmap,
                     # normalization_x,
                     )
    ax0.set_xlabel(r'$k_z$')
    ax0.grid(True)
    
    # ax0.set_xlim(umin/5, umax/5)
    
    # # best fit of data
    # (mu, sigma) = norm.fit(w)
    #
    # # add a 'best fit' line
    # y = norm.pdf( bins_w, mu, sigma)
    # ax0.plot(bins_w, y, 'r--', linewidth=2)
    #


    ##########
    
    # ax0 = axs[1,1]
    #
    # ax_2dhist_simple(kx, u, ax0, (bins,bins_u),
    #                 xmin=k_min, xmax=k_max,
    #                 ymin=vmin, ymax=vmax,
    #                  # cmap,
    #                  # normalization_x,
    #                  )
    # ax0.set_xlabel('Kx')
    #
    # ax0 = axs[1,2]
    #
    # ax_2dhist_simple(ky, v, ax0, (bins,bins_u),
    #                  xmin=k_min, xmax=k_max,
    #                  ymin=vmin, ymax=vmax,
    #                  # cmap,
    #                  # normalization_x,
    #                  )
    # ax0.set_xlabel('Ky')
    #
    # ax0 = axs[1,3]
    #
    # ax_2dhist_simple(kz, w, ax0, (bins,bins_w),
    #                  xmin=k_min, xmax=0,
    #                  ymin=vmin, ymax=vmax,
    #                  # vmin, vmax,
    #                  # cmap,
    #                  # normalization_x,
    #                  )
    # ax0.set_xlabel('Kz')
        
    ## Vel vs Doppl
    
    # ax0 = axs[2,0]
    #
    # ax_2dhist_simple(u, dops, ax0, (bins_u,bins_w),
    #                  xmin=vmin, xmax=vmax,
    #                 ymin=k_min, ymax=k_max,
    #                  # cmap,
    #                  # normalization_x,
    #                  )
    #
    # ax0.set_ylabel('Doppler')
    # ax0.set_xlabel('u')
    #
    # ax0 = axs[2,1]
    #
    # ax_2dhist_simple(v, dops, ax0, (bins_u,bins_w),
    #                  xmin=vmin, xmax=vmax,
    #                 ymin=k_min, ymax=k_max,
    #                  # cmap,
    #                  # normalization_x,
    #                  )
    #
    # ax0.set_ylabel('Doppler')
    # ax0.set_xlabel('v')
    #
    # ax0 = axs[2,2]
    #
    # ax_2dhist_simple(w, dops, ax0, (bins_w,bins_w),
    #                  xmin=k_min, xmax=k_max,
    #                 ymin=k_min, ymax=k_max,
    #                  # cmap,
    #                  # normalization_x,
    #                  )
    #
    # ax0.set_ylabel('Doppler')
    # ax0.set_xlabel('w')
    
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def plot_hist(df):
    
    import seaborn as sns
    
    # Specify the location you are interested in
    x_location = 14.25
    y_location = 70.35
    z_location = 96
    
    # Define a tolerance for matching the irregular steps
    tolerance = 1
    ztolerance = 1
    
    t0 = df['t'].min() + pd.to_timedelta(21*60*60, unit='s')  
    ttolerance = pd.to_timedelta(30*60, unit='s') 
    
    valid  = (df['lons'].between(x_location - tolerance, x_location + tolerance))
    valid &= (df['lats'].between(y_location - tolerance, y_location + tolerance))
    valid &= (df['heights'].between(z_location - ztolerance, z_location + ztolerance)) 
    valid &= (df['t'].between(t0 - ttolerance, t0 + ttolerance))
    
    # Filter the DataFrame for the specific location with the given tolerance
    filtered_df = df[valid]
    
    # print(filtered_df)
    
    # Extract the temperature values
    temperature_values = filtered_df['dops']
    
    # Check if the filtered DataFrame is empty
    if temperature_values.empty:
        print(f"No data available for location (x={x_location}, y={y_location}) within the tolerance.")
    else:
        # Plot the histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(temperature_values, bins=10, kde=True)
        plt.title(f'Histogram of Temperature Values at Location (x={x_location}, y={y_location})')
        plt.xlabel('Temperature')
        plt.ylabel('Frequency')
        plt.show()

        

def read_doppler_hdf5(fn, latref = 54., lonref = 12.5,
                      tbase_txt='2018-11-05',
                      tmin_hr=0, tmax_hr=24
                      ):
    '''
    Read Koki's hdf5 file containing meteor detections and Doppler shifts, returning a pandas DataFrame.
    tbase_txt  - date in year-month-dom ASCII format
    tmin_hr    - minimum hours of a given day to be considered
    tmax-hr    - maximum hours of a given day to be considered
    Columns: data [Hz] - Doppler shifts of meteor detections, related to wind through forward model:
                         data = coeff_u * u  +  coeff_v * v  +  coeff_w * w
             sigma [Hz]- 1-sigma uncertainty in "data"
             coeff_u   - see above
             coeff_v   - see above
             coeff_w   - see above
             lat [deg] - latitude of detected meteor
             lon [deg] - longitude of detected meteor
             x [km]    - x coordinate relative to latref/lonref
             y [km]    - y coordinate relative to latref/lonref
             z [km]    - altitude of detected meteor
             t         - time of meteor detection
             
    '''

    f = h5py.File(fn,'r')
    f.close()
    
    return

def read_wind_hdf5(fn):
    '''
    Read Koki's hdf5 file containing meteor detections and Doppler shifts, returning a pandas DataFrame.
    tbase_txt  - date in year-month-dom ASCII format
    tmin_hr    - minimum hours of a given day to be considered
    tmax-hr    - maximum hours of a given day to be considered
    Columns: data [Hz] - Doppler shifts of meteor detections, related to wind through forward model:
                         data = coeff_u * u  +  coeff_v * v  +  coeff_w * w
             sigma [Hz]- 1-sigma uncertainty in "data"
             coeff_u   - see above
             coeff_v   - see above
             coeff_w   - see above
             lat [deg] - latitude of detected meteor
             lon [deg] - longitude of detected meteor
             x [km]    - x coordinate relative to latref/lonref
             y [km]    - y coordinate relative to latref/lonref
             z [km]    - altitude of detected meteor
             t         - time of meteor detection
             
    '''

    try:
        f = h5py.File(fn,'r')
    except:
        return(None)
    
    times = f['t'][...]
    
    df = {}
    # Forward model is -2*pi*d = bragg_vector dot wind vector
    df['u'] = f['wind_u'][:] 
    df['v'] = f['wind_v'][:] 
    df['w'] = f['wind_w'][:] 
    
    # Location variables
    df['x'] = f['x'][:] 
    df['y'] = f['y'][:] 
    df['z'] = f['z'][:] 
    df['times'] = times
    
    f.close()
    
    return df

def save_mean_winds(df, filename):
    
    # Open HDF5 file and write in the data_dict structure and info
    with h5py.File(filename,'w') as fp:
        
        for key in df.keys():
        
            fp[key] = df[key]

    
    
def plot_winds(x, y,
               u, v, w,
               min_value=-50,
               max_value=50,
               figpath='./',
               alpha=0.5):
    
    norm = plt.Normalize(min_value, max_value)
    norm2 = plt.Normalize(-10, 10)
    
    _, axs = plt.subplots(3,1, figsize=(10,8), sharex=True, sharey=True)
    
    im = axs[0].scatter(x, y, c=u, cmap='jet', norm=norm, alpha=alpha)
    plt.colorbar(im, ax=axs[0])
    
    im = axs[1].scatter(x, y, c=v, cmap='jet', norm=norm, alpha=alpha)
    plt.colorbar(im, ax=axs[1])
    
    im = axs[2].scatter(x, y, c=w, cmap='jet', norm=norm2, alpha=alpha)
    plt.colorbar(im, ax=axs[2])
    
    # axs[0].set_xlim(64,72)
    # axs[0].set_ylim(70,110)
    plt.tight_layout()
    
    for i in range(3):
        axs[i].grid(True)
    
    figname = os.path.join(figpath, "winds_z_%d.png" %x[0] )
    
    plt.savefig(figname)
    # plt.show()
    
    plt.close()
    
def plot_winds3D(t,x, y,z,
               u, v, w,
               min_value=-50,
               max_value=50,
               figpath='./',
               alpha=0.5):
    
    norm = plt.Normalize(min_value, max_value)
    norm2 = plt.Normalize(-5, 5)
    
    fig = plt.figure(figsize=(10,8))
    
    ax0 = fig.add_subplot(311, projection='3d')
    ax1 = fig.add_subplot(312, projection='3d')
    ax2 = fig.add_subplot(313, projection='3d')
    
    im = ax0.scatter3D(t, x, z, c=u, cmap='jet', norm=norm, alpha=alpha)
    plt.colorbar(im, ax=ax0)
    
    im = ax1.scatter3D(t, x, z, c=v, cmap='jet', norm=norm, alpha=alpha)
    plt.colorbar(im, ax=ax1)
    
    im = ax2.scatter3D(t, x, z, c=w, cmap='jet', norm=norm2, alpha=alpha)
    plt.colorbar(im, ax=ax2)
    
    # ax0.set_ylim(52,72)
    ax0.set_zlim(70,110)
    ax0.set_box_aspect(aspect = (2,1,1))
    
    # ax1.set_ylim(52,72)
    ax1.set_zlim(70,110)
    ax1.set_box_aspect(aspect = (2,1,1))
    
    # ax2.set_ylim(52,72)
    ax2.set_zlim(70,110)
    ax2.set_box_aspect(aspect = (2,1,1))
    
    plt.tight_layout()
    
    figname = os.path.join(figpath, "winds_3D_%d.png" %t[0] )
    
    plt.savefig(figname)
    # plt.show()
    
    plt.close()

def __get_data_mask(df, dx=np.inf, dy=10, dz=4,
                    z0=90):
    
    xmask = (df['x']>-dx/2) & (df['x']<dx/2)
    ymask = (df['y']>-dy/2) & (df['y']<dy/2)
    zmask = (df['z']>z0-dz/2) & (df['z']<z0+dz/2)
    
    return(xmask, ymask, zmask)
        
def test_doppler_files():
    
    path = '/Users/mcordero/Data/IAP/SIMONe/Germany/Simone2018'
    figpath = path
    
    #Read meteor data in LLA coordinates
    meteor_obj = SMRReader(path)
    
    while True:
        
        info = meteor_obj.read_next_file(enu_coordinates=True)
        if info != 1: break
        
        df = meteor_obj.df
        
        xmask, ymask, zmask = __get_data_mask(df)
        
        mask = xmask & ymask & zmask
        
        times = df['times'][mask].values
        lats = df['x'][mask].values
        lons = df['y'][mask].values
        alts = df['z'][mask].values
        
        u = df['u'][mask].values
        v = df['v'][mask].values
        w = df['w'][mask].values
        
        plot_winds(times, lats,
                   u, v, w,
                   figpath=figpath)
        
        plot_winds3D(times,
                     lats, lons, alts,
                     u, v, w,
                     figpath=figpath)

def test_wind_files():
    
    path = '/Users/mcordero/Data/IAP/SIMONe/Winds2018_3D_mask_before'
    figpath = path
    
    files = glob.glob('%s/*.h5' %path)
    
    for file in files:
        
        filename = os.path.join(path, file)
        df = read_wind_hdf5(filename)
        if df is None: break
        
        xmask, ymask, zmask = __get_data_mask(df)
        
        times = df['times']
        lats = df['x'][xmask]
        lons = df['y'][ymask]
        alts = df['z'][zmask]
        
        u = df['u'][:,xmask]
        v = df['v'][:,xmask]
        w = df['w'][:,xmask]
        
        u = u[:,:,ymask,zmask]
        v = v[:,:,ymask,zmask]
        w = w[:,:,ymask,zmask]
        
        T,X,Y,Z = np.meshgrid(times, lats, lons, alts, indexing='ij')
        
        plot_winds(T.flatten(), X.flatten(),
                   u.flatten(), v.flatten(), w.flatten(),
                   figpath=figpath)
        
        plot_winds3D(T.flatten(),
                     X.flatten(), Y.flatten(), Z.flatten(),
                     u.flatten(), v.flatten(), w.flatten(),
                     figpath=figpath)
                
class SMRReader(object):
    
    n_samples = None
    times = None
    latitudes = None
    longitudes = None
    altitudes = None
    
    braggs = None
    dopplers = None
    
    u = None
    v = None
    w = None
    
    x = None
    y = None
    z = None
    
    r = None
    
    file_index = 0
            
    def __init__(self, path, realtime=False, pattern='*'):
        
        files = self.__find_files(path, pattern=pattern)
        
        if len(files) == 0:
            raise ValueError('No files found in %s' %path )
        
        self.path = path
        
        df = self.__read_data(files[0], enu_coordinates=False)
        
        self.df = df
        self.unfiltered_df = df
        self.files = files
        
        if realtime:
            self.file_index = len(files) - 2
        else:
            self.file_index = -1
    
        self.set_spatial_center()
        
        print('Default meteor middle point (LLA): ', self.lon_center, self.lat_center, self.alt_center)
        
    def __find_files(self, path, pattern='*'):
        
        files = glob.glob1(path, "%s.h5" %pattern )
        files = sorted(files)
        
        return(files)
    
    def __read_data(self, filename, enu_coordinates=True):
        
        file = os.path.join(self.path, filename)
        
        with h5py.File(file,'r') as fp:

            times   = fp['t'][()]
            link    = fp['link'][()]
            alt     = fp['heights'][:]*1e-3 #To km
            lat     = fp['lats'][:]
            lon     = fp['lons'][:]
            dopplers    = fp['dops'][:]
            dop_errs    = fp['dop_errs'][:]
            braggs      = fp['braggs'][:]
            
            try:
                dcosx       = fp['dcos'][:,0]
                dcosy       = fp['dcos'][:,1]
            except:
                dcosx   =   alt*0
                dcosy   =   alt*0
            
            empty_data = np.zeros_like(times, dtype=np.float64) + np.nan
            
            if 'u' in fp.keys():
                u = fp['u'][:]
                v = fp['v'][:]
                w = fp['w'][:]
            else:
                u = empty_data
                v = empty_data
                w = empty_data
            
            if 'temp' in fp.keys():
                T = fp['temp'][:]   #Kelvin
                P = fp['pres'][:]   #Pascal <> N/m2
                rho = fp['rho'][:]  #kg/m3
                tke = fp['tke'][:]  #m2/s2
            else:
                T = empty_data
                P = empty_data
                rho = empty_data
                tke = empty_data
                
            
            if 'SMR_like' in fp.keys():
                smr_like = fp['SMR_like'][:]
            else:
                smr_like = np.ones_like(times)
            
            # attrs = None
            # if fp.attrs:
            #     attrs = {}
            #     attrs['lon_center'] = fp.attrs['lon_ref']
            #     attrs['lat_center'] = fp.attrs['lat_ref']
            #     attrs['alt_center'] = fp.attrs['alt_ref']
        
        self.ini_time = times.min()
        self.n_samples = len(times)
        
        t = pd.to_datetime(times, unit='s')
        df = pd.DataFrame(data = t, columns=['t'])
        
        df['times'] = times
        df['link'] = link
        df['lats'] = lat
        df['lons'] = lon
        df['heights'] = alt #km
        
        df['dop_errs'] = dop_errs
        df['braggs_x'] = braggs[:,0]
        df['braggs_y'] = braggs[:,1]
        df['braggs_z'] = braggs[:,2]
        df['dcosx'] = dcosx
        df['dcosy'] = dcosy
        
        #DNS, Scale by 10
        df['dops'] = dopplers
        
        df['u'] = u
        df['v'] = v
        df['w'] = w
        
        df['T'] = T
        df['P'] = P
        df['rho'] = rho
        df['tke'] = tke
        
        df['SMR_like'] = smr_like
        
        df = df[df.dops.notnull()]
        
        ### Meteor geolocation
        # getting the radius at each point, the difference with respect to getting the radius to a reference point is less than 40 meters.
        
        if enu_coordinates:
            #Make calculations with filtered data
            lats = df['lats'].values
            lons = df['lons'].values
            alts = df['heights'].values
            
            # x0,y0,h0 = lla2xyh(lats=lats, lons=lons, alts=alts,
            #                 lat_center=self.lat_center,
            #                 lon_center=self.lon_center,
            #                 alt_center=self.alt_center,
            #                 )
            
            x,y,_ = lla2enu(lats, lons, alts,
                            lat_ref=self.lat_center,
                            lon_ref=self.lon_center,
                            alt_ref=self.alt_center,
                            units='m',
                            )
            
            ##################
            #xyz coordinates in m
            df['x'] = x
            df['y'] = y
            df['z'] = alts*1e3
        
        # mask = (df['SMR_like'] == 1)
        # df = df[mask]
        
        df.sort_values(['times'], inplace=True, ignore_index=True)
        
        # if attrs is not None:
        #     df.attrs = attrs
            
        return(df)
    
    def get_initial_time(self):
        
        return(self.ini_time)
    
    def set_spatial_center(self, lon_center=None, lat_center=None, alt_center=None):
        
        if lat_center is None:
            try:
                lat_center = self.df.attrs['lat_center']
            except:
                lat_center = np.round( np.median(self.df['lats'].values), 1)
            
        if lon_center is None:
            try:
                lon_center = self.df.attrs['lon_center']
            except:
                lon_center = np.round( np.median(self.df['lons'].values), 1)
        
        if alt_center is None:
            try:
                alt_center = self.df.attrs['alt_center']
            except:
                alt_center = np.round( np.median(self.df['heights'].values), 1)
              
        self.lat_center = lat_center
        self.lon_center = lon_center
        self.alt_center = alt_center
        
    def get_spatial_center(self):
        
        return(self.lon_center, self.lat_center, self.alt_center)
    
    def get_rx_tx_lla(self, index):
        
        link = self.df['link'][index].decode('ascii', 'replace')
        link = link.split('-')[0]
        tx, rx = link.split('_')[0:2]
        
        tx_lla = stations[tx]
        rx_lla = stations[rx]
        
        return(tx_lla, rx_lla)
        
    def read_next_file(self, enu_coordinates=True, single_day=False):
        
        self.file_index += 1
        self.central_date = None
        
        if self.file_index >= len(self.files):
            return 0
        
        df = None
        for i in [0,-1,1]:
            
            idx_file = self.file_index + i
            
            if idx_file < 0:
                continue
            
            if idx_file >= len(self.files):
                continue
            
            filename = self.files[idx_file]
            
            df_i = self.__read_data(filename, enu_coordinates=enu_coordinates)
            
            print('\nMeteor file %s [t=%2.1f]' %(filename, self.ini_time) )
            
            if df is None:
                df = df_i
                self.central_date = datetime.utcfromtimestamp(df["times"][0])
            else:
                df = pd.concat([df, df_i], ignore_index=True)
            
            if single_day:
                break
            
        self.df = df
        self.unfiltered_df = df
            
        self.filename = filename
        
        return(1)   
    
    def save(self, rpath, df=None, filename=None, dropnan=True, scale_factor=1.0):
        
        if df is None:
            df = self.df
        
        if filename is None:
            fullfile = self.files[self.file_index]
            _, filename = os.path.split(fullfile)
            filename = os.path.join(rpath, 'filtered_'+filename)
        
        if dropnan:
            df = df.dropna(subset=['dops'])
        
        braggs = np.array( [df['braggs_x'].values,
                            df['braggs_y'].values,
                            df['braggs_z'].values
                            ]
                        )
        
        fp = h5py.File(filename,'w')

        fp['link']      = df['link'].values
        fp['heights']   = df['heights'].values*1e3 #To m
        fp['lats']      = df['lats'].values
        fp['lons']      = df['lons'].values
        fp['dops']      = df['dops'].values*scale_factor
        fp['dop_errs']  = df['dop_errs'].values
        fp['braggs']    = braggs.T
        # fp['dcosx']     = self.dcosx
        # fp['dcosy']     = self.dcosy  
        fp['t']         = df['times'].values
        fp['u']         = df['u'].values*scale_factor
        fp['v']         = df['v'].values*scale_factor
        fp['w']         = df['w'].values*scale_factor
        
        fp['temp']        = df['T'].values  #Kevin
        fp['pres']        = df['P'].values  #Pascal
        fp['rho']         = df['rho'].values    #kg/m3
        fp['tke']         = df['tke'].values    #m2/s2
        
        if 'SMR_like' in df.keys():
            fp['SMR_like']    = df['SMR_like'].values
        
        for key in df.attrs.keys():
            fp.attrs[key] = df.attrs[key]
            
        fp.close()
    
    def filter(self, path=None, outlier_sigma=5, **kwargs):
        
        df = self.unfiltered_df.copy()
        
        df = remove_close_clusters(df)
        
        if df.size == 0:
            self.df = df
            return
        
        #Remove outliers using clustering
        df = filter_data(df,
                         central_date=self.central_date,
                         **kwargs)
        
        if df.size == 0:
            print("File %s has been completely filtered" %self.central_date)
            self.df = df
            return
            
        #Remove outliers based on mean wind
        df_winds, df = mean_wind_grad(df, outlier_sigma=outlier_sigma) 
        
        if path is not None:
        
            ini_date = self.get_initial_date()
        
            figfile = os.path.join(path, "mean_wind_%s.png" %ini_date.strftime('%Y%m%d_%H0000'))
        
            plot_mean_winds(df_winds["times"],
                            df_winds["alts"],
                            df_winds["u0"],
                            df_winds["v0"],
                            df_winds["w0"],
                            vmins=[-100,-100, -10],
                            vmaxs=[ 100, 100,  10],
                            figfile=figfile,
                            histogram=True
                            )
        
        self.df  = df
        
        return
    
    def add_synthetic_noise(self, sigma):
        
        if sigma == 0:
            self.df['dop_errs'] = 1.0
            return
        
        n = self.df.shape[0]
        
        noise = sigma*np.random.randn(n)
        
        self.df['dops'] += noise
        self.df['dop_errs'] = sigma
        
        
    def plot_sampling(self, suffix="", **kwargs):
        
        df = self.df
        
        if df.size == 0:
            return
        
        ini_date = self.get_initial_date()
        
        suffix += ini_date.strftime('_%Y%m%d_%H0000')
        
        try:
            plot_spatial_sampling(df, suffix=suffix, **kwargs)
            plot_Doppler_sampling(self.unfiltered_df, df, suffix=suffix, path=kwargs["path"])
        except:
            pass
        
        return
    
    def plot_hist(self, suffix="", **kwargs):
        
        df = self.df
        df_unfiltered = self.unfiltered_df
        
        ini_date = self.get_initial_date()
        
        suffix += ini_date.strftime('%Y%m%d_%H0000')
        
        plot_Doppler_sampling(df_unfiltered, df, suffix=suffix, **kwargs)
        
    def get_initial_date(self):
        
        ini_date = datetime.utcfromtimestamp(self.df['times'].min()) 
        
        return(ini_date)

              
if __name__ == '__main__':
    
    test_doppler_files()
    # test_wind_files()
    