'''
Created on 4 Sep 2022

@author: mcordero
'''
# A module getting windfield cuts using Brian's wind field  inversion on MMARIA systems

import os
from datetime import datetime

# Import TensorFlow and NumPy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors

# from scipy.stats import norm

from pinn import hyper as pinn

from utils.PhysicalParameters import MSIS
from utils.histograms import ax_2dhist_simple
from utils.plotting import epoch2num
from radar.smr.smr_file import filter_data


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
                        vmin=None, vmax=None,
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
    
    filename = os.path.join(path, 'sampling_%s_%d.png' %(suffix, lla) )
    
    # _, axs = plt.subplots(2, 3, figsize=(12,10))
    
    fig = plt.figure(figsize=(12,10))
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
    plt.close()

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
        y = df['lats'].values.astype(dtype)[:N]
        z = df['heights'].values.astype(dtype)[:N]
        
        xlabel = 'Longitude'
        ylabel = 'Latitude'
        zlabel = 'Altitude (km)'
    else:
        x = df['x'].values.astype(dtype)[:N]*1e-3
        y = df['y'].values.astype(dtype)[:N]*1e-3
        z = df['z'].values.astype(dtype)[:N]*1e-3
        
        xlabel = 'X (m)'
        ylabel = 'Y (m)'
        zlabel = 'Z (m)'
        
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
    
def plot_Doppler_sampling(df_unfiltered, df, path, suffix='',
                          lla=True, cmap='jet',
                          label_colorbar='meteor counts',
                          vmin=-10, vmax=10,
                          k_min=-1.3, k_max=1.3,
                          umin=-50, umax=50,
                          bins=40,
                          density=False,
                          noise=None):
    
    bins_u = np.linspace(umin, umax, bins)
    bins_w = np.linspace(vmin, vmax, bins)
    
    
    # t = df['times'].values
    
    # t_num = epoch2num(t)
    # locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    # formatter = mdates.ConciseDateFormatter(locator)
    
    # tlabel = 'UTC'
    
    kx = df['braggs_x'].values                     #[N]
    ky = df['braggs_y'].values
    kz = df['braggs_z'].values
    
    dops = 2*np.pi*df['dops'].values
    derr = 2*np.pi*df['dop_errs'].values
    
    dops_unf = 2*np.pi*df_unfiltered['dops'].values
    derr_unf = 2*np.pi*df_unfiltered['dop_errs'].values
    
    std = np.std(dops)
    std_unf = np.std(dops_unf)
    
    # u = df['u'].values                              #[N]
    # v = df['v'].values                              #[N]
    # w = df['w'].values 
    
    dxy = np.sqrt( df['dcosx']**2 + df['dcosy']**2 )
    zenith = np.arcsin(dxy)*180/np.pi
    
    filename = os.path.join(path, 'sampling_Dopp_%s.png' %(suffix) )
    
    _, axs = plt.subplots(2, 3, figsize=(9,6))
    
    ###### Doppler ###
    
    ax0 = axs[0,0]
    
    ax0.set_title(r'Original dataset: $2 \pi f$ (std=%3.2f)' %std_unf)
    ax0.hist(dops_unf, bins,
                    label='Doopler'
                     )    
    
    # ax0.hist(derr_unf, bins,
    #             label='Noise',
    #             color='r',
    #             alpha=0.3,
    #              ) 
        
    ax0.set_ylabel('Counts')
    ax0.set_xlabel('Doppler (m/s)')
    # ax0.set_xlim(vmin, vmax)
    ax0.set_yscale('log')
    ax0.grid(True)
    
    
    ax0 = axs[0,1]
    
    ax0.set_title(r'Filtered data: $2 \pi f$ (std=%3.2f)' %std)
    ax0.hist(dops, bins,
                    label='Doopler'
                     )    
    
    # ax0.hist(derr, bins,
    #             label='Noise',
    #             color='r',
    #             alpha=0.3,
    #              ) 
        
    ax0.set_xlabel('Doppler (m/s)')
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
    ax0.set_xlabel('Doppler error (m/s)')
    
    
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
    
def train_hyper(df,
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
                  only_SMR=True,
                  dropout=True,
                  w_pde_update_rate=1e-4,
                  nn_type = 'deeponet',
                 ):
    
    # if dt < 0: tini = -dt
    # else: tini = 0
    # dt = np.abs(dt)
    
    # config_gpu(gpu_flg = 1)
    
    seed = 191
    # Set data type
    np.random.seed(seed)
    
    ini_date = datetime.utcfromtimestamp(df['times'].min())    
    plot_spatial_sampling(df, path=rpath, suffix='prefilter_%s' %ini_date.strftime('%Y%m%d-%H%M%S'),
                        # vmin=0, vmax=500,
                        )
    
    df_filtered = filter_data(df,
                     tini=tini, dt=dt,
                     dlon=dlon, dlat=dlat, dh=dh,
                     lon_center=lon_center,
                      lat_center=lat_center,
                      alt_center=alt_center,
                      only_SMR=only_SMR)
    
    # bgn = -10
    # df.loc[:,'u'] = df['u'].values + bgn
    # df.loc[:,'v'] = df['v'].values + bgn
    # df.loc[:,'w'] = df['w'].values + bgn
    # df.loc[:,'dops'] = df['dops'].values - bgn/(2*np.pi)*(df['braggs_x'].values + df['braggs_y'].values + df['braggs_z'].values)
    
    # df_training = df_filtered
    # df_test     = df_filtered#.sample(frac=0.01, random_state=0) #
    
    df_training = df_filtered.sample(frac=0.95, random_state=0)
    df_test     = df_filtered.drop(df_training.index)
    
    df_training.sort_index(inplace=True)
    
    ini_date = datetime.utcfromtimestamp(df_training['times'].min())
    plot_spatial_sampling(df_training, path=rpath,
                        suffix='postfilter_%s' %ini_date.strftime('%Y%m%d-%H%M%S'),
                        # vmin=0, vmax=100,
                        )
    
    plot_Doppler_sampling(df, df_filtered, rpath,
                          suffix='postfilter_%s' %ini_date.strftime('%Y%m%d-%H%M%S')
                          )
    
    
    # plot_2point_sampling(df_training, path=rpath,
    #                     suffix='delta_sampling_%s' %ini_date.strftime('%Y%m%d-%H%M%S'),)
    
    # plot_delta_sampling(df, rpath)
    
    msis = None
    # try:
    #     msis = MSIS(ini_date,
    #                  glat=df['lats'].median(),
    #                  glon=df['lons'].median(),
    #                  time_range=dt,
    #                  plot_values=True)
    # except:
    #     msis = None
    
    if short_naming:
        suffix = '%s' %ini_date.strftime('%Y%m%d-%H%M%S')
    else:
        suffix = "%s_w%02dn%03.2f[%s]a%sl%02dn%03dd%03db%02dw%2.1elr%2.1em%2.1e_laaf%2.1e_%s_ur=%2.1e" %(
                                                            ini_date.strftime('%Y%m%d'),
                                                            dt,
                                                            noise_sigma,
                                                            NS_type,
                                                            activation[:3],
                                                            num_hidden_layers,
                                                            num_neurons_per_layer,
                                                            n_nodes,
                                                            n_blocks,
                                                            w_pde,
                                                            learning_rate,
                                                            N_pde,
                                                            w_srt,
                                                            # dropout,
                                                            w_init[:2],
                                                            # init_sigma,
                                                            w_pde_update_rate,
                                                            )
    
    ###########################
    #Traiing dataset

    t = df_training['times'].values                          #[N]
    
    x = df_training['lons'].values                             #[N]
    y = df_training['lats'].values                             #[N]
    z = df_training['heights'].values                             #[N]
    
    kx = df_training['braggs_x'].values                     #[N]
    ky = df_training['braggs_y'].values
    kz = df_training['braggs_z'].values
    
    k = np.stack([kx, ky, kz], axis=1)                      #[N,3]
    
    u = df_training['u'].values                              #[N]
    v = df_training['v'].values                              #[N]
    w = df_training['w'].values                              #[N]
    
    # T = df_training['T'].values                            #Kelvin
    # rho = df_training['rho'].values                            #Kg/m3
    # P = df_training['P'].values                              #Pascal <> N/m2
    # rho_msis     = msis.get_rho(z*1e3)
    # N   = msis.get_N(z*1e3)
    
    dops        = 2*np.pi*df_training['dops'].values
    dop_errs    = 2*np.pi*df_training['dop_errs'].values
    
    dop_std     = np.std(dops)
    noise_std   = 10#np.std(dop_errs)
    
    d = dops
    
    if noise_sigma < 0:
        noise_std  = 1.0
    elif noise_sigma > 0:
        print("Adding synthetic noise ...")
        noise = noise_sigma*dop_std*np.random.normal(0, 1, size=dops.shape[0])
        d   = dops + noise    #[N]
        noise_std   = np.std(noise)
    
    d_err = np.ones_like(dops)*noise_std
    
    print('*'*40)
    print("Doppler std=", dop_std )
    print("Noise std=", noise_std)
    print('*'*40)
    
    ###########################
    # df_test = df_training
    
    #Test dataset
    t_test = df_test['times'].values                          #[N]
    
    x_test = df_test['lons'].values                             #[N]
    y_test = df_test['lats'].values                             #[N]
    z_test = df_test['heights'].values                             #[N]
    
    u_test = df_test['u'].values                              #[N]
    v_test = df_test['v'].values                              #[N]
    w_test = df_test['w'].values                              #[N]
    
    # P_test   = df_test['P'].values
    # rho_test = df_test['rho'].values
    
    kx_test = df_test['braggs_x'].values                     #[N]
    ky_test = df_test['braggs_y'].values
    kz_test = df_test['braggs_z'].values
    
    d_test = 2*np.pi*df_test['dops'].values
    
    ###########################
    if filename_model is None:
        filename_model = 'model_%s.h5' %suffix #ini_date.strftime('%Y%m%d')
        
    # filename_mean = os.path.join(rpath, filename_model)
    
    suffix_mean = suffix #'%s_w%02d_n%03.1f' %(ini_date.strftime('%Y%m%d-%H%M%S'), dt, noise_sigma) # #
    filename_mean = os.path.join(rpath, 'mean_model_%s.h5' %suffix_mean)
    
    if os.path.exists(filename_mean):
    
        # print('Loading %s' %filename_mean)
        nn = pinn.restore(filename_mean)
    
    else:
        # Initialize Neural Network model
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
                        lon_ref  = (x.min() + x.max())/2,
                        lat_ref  = (y.min() + y.max())/2,
                        alt_ref  = (z.min() + z.max())/2,
                        NS_type  = NS_type,
                        nn_type = nn_type,
                    )
        
        # with tf.device("/device:GPU:0"):
        nn.train(t, x, y, z, d, k,
                 t_test, x_test, y_test, z_test, u_test, v_test, w_test,
                 # t, x, y, z, u, v, w,
                 d_err      = d_err,
                 epochs     = nepochs,
                 filename   = filename_mean,
                 w_pde_update_rate = w_pde_update_rate,
                 lr      = learning_rate,
                w_data  = w_data,
                w_div   = w_pde,
                w_mom   = w_pde,
                w_temp  = w_pde,
                w_srt   = w_srt,
                ns_pde  = N_pde,
                # NS_type  = NS_type,
                dropout  = dropout,
                laaf=laaf,
                 )
    
        nn.save(filename_mean)
        # nn.restore(filename_mean)
    
    figname01 = os.path.join(rpath, 'loss_hist_%s.png' %suffix_mean)
    
    nn.plot_loss_history(figname=figname01)
    
    ############################################################
    #Traning points
    figname02 = os.path.join(rpath, 'training_winds_%s.png' %suffix_mean)
    figname03 = os.path.join(rpath, 'training_errors_%s.png' %suffix_mean)
    figname04 = os.path.join(rpath, 'training_errors_k_%s.png' %suffix_mean)
    figname05 = os.path.join(rpath, 'training_statistics_%s.png' %suffix_mean)
    figname_P = os.path.join(rpath, 'training_pressure_%s.png' %suffix_mean)
    
    outputs = nn.infer(t, x, y, z)
    
    u_nn = outputs[:,0]
    v_nn = outputs[:,1]
    w_nn = outputs[:,2]
    
    d_nn = -(u_nn*kx + v_nn*ky + w_nn*kz)
    
    nn.plot_statistics(
                    u, v, w,
                    u_nn, v_nn, w_nn,
                    figname=figname05)
    
    nn.plot_solution(t, x, y, z,
                     u, v, w, d,
                     u_nn, v_nn, w_nn, d_nn,
                      k_x=kx, k_y=ky, k_z=kz,
                     figname_winds=figname02,
                     figname_errs=figname03,
                     figname_errs_k=figname04,
                     figname_pressure=figname_P)
    
    ############################################################
    #Validation points
    
    figname02 = os.path.join(rpath, 'testing_winds_%s.png' %suffix)
    figname03 = os.path.join(rpath, 'testing_errors_%s.png' %suffix)
    figname04 = os.path.join(rpath, 'testing_errors_k_%s.png' %suffix)
    figname05 = os.path.join(rpath, 'testing_statistics_%s.png' %suffix)
    figname_P = os.path.join(rpath, 'testing_pressure_%s.png' %suffix)
    
    outputs = nn.infer(t_test, x_test, y_test, z_test)
    
    u_nn = outputs[:,0]
    v_nn = outputs[:,1]
    w_nn = outputs[:,2]
    
    d_nn = -(u_nn*kx_test + v_nn*ky_test + w_nn*kz_test)
    
    nn.plot_statistics(
                    u_test, v_test, w_test,
                    u_nn, v_nn, w_nn,
                    figname=figname05)
    
    nn.plot_solution(t_test, x_test, y_test, z_test,
                     u_test, v_test, w_test, d_test,
                     u_nn, v_nn, w_nn, d_nn,
                    k_x=kx_test, k_y=ky_test, k_z=kz_test,
                     figname_winds=figname02,
                     figname_errs=figname03,
                     figname_errs_k=figname04,
                     figname_pressure=figname_P)
    
    return( filename_mean )