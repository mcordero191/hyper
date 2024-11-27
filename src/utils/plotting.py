import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datetime

from skimage.filters import gaussian

def epoch2num(t):

    dates = [datetime.datetime.utcfromtimestamp(i) for i in t]
    num = mdates.date2num(dates)
    
    return(num)
    
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
                     decimation=2,
                     df_meteor=None,
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
    
    w_masked = np.ma.masked_array(w,mask=np.isnan(w))
    
    if w_masked.count() < 1:
        return None, None

    
    im = None
    im = ax.pcolormesh(x, y, w, cmap=cmap, vmin=wmin, vmax=wmax)#, alpha=0.3)
    # im = ax.contourf(x,y,w_masked, cmap=cmap, vmin=wmin, vmax=wmax)#, alpha=0.3)
    
    # im = ax.imshow(w, interpolation='Gaussian',
    #                cmap=cmap,
    #                vmin=wmin, vmax=wmax,
    #                origin='lower',
    #                extent = (x[0,0], x[0,-1], y[0,0], y[-1,0]),
    #                aspect='auto')
    
    qk = ax.quiver(x[::decimation,::decimation],
                    y[::decimation,::decimation],
                    u[::decimation,::decimation],
                    v[::decimation,::decimation],
                   # w,
                   # cmap=cmap,
                   # clim=[wmin, wmax],
                   scale=scale,
                   color='b'
                   )
    
    # ax.scatter(15.82, 69.45, marker='x', color='r')
    
    alpha = 0.5
    
    # ax.scatter(14.25, 70.35, marker='+', s=15, c='m', alpha=alpha, label='VORTEX: DWNLEG') #MAARSY        
    # ax.scatter(15.75, 69.45, marker='o', s=15, c='m', alpha=alpha, label='VORTEX: UPLEG') #MAARSY
    ax.scatter(16.04, 69.30, marker='*', s=20, c='r', alpha=alpha, label='MAARSY') #MAARSY
    
    ax.set(xlabel='Longitude', ylabel ='Latitude')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    ax.set_title(title)
    
    if df_meteor is not None:
        
        met_lons    = df_meteor['lons'].values
        met_lats    = df_meteor['lats'].values
        # met_alts    = df_meteor['heights'].values
        
        ax.scatter(met_lons, met_lats, marker='X', c='k', alpha=0.2, label='Meteors')
     
    return(qk, im)

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
                 cmap = 'seismic',
                 zdecimator=1,
                 scale=None,
                 df_meteor=None,
                 h_width=0.5, #km
                 ):
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
                       figsize=(5.0*nx_subplot, 5.5*ny_subplot),
                       squeeze=False)
    
    fig.suptitle(title, fontsize=14, weight='bold')
    
    for i, ax in enumerate(axs.flat[::-1]):
        
        if i >= nz: break
        
        df_meteor_h = None
        if df_meteor is not None:
            mask = np.abs(df_meteor['heights'] - z1d[i]) < h_width
            df_meteor_h = df_meteor[mask]
            
        qk, im = _plot_3d_vector(x3d[i,:,:], y3d[i,:,:],
                        u[i,:,:], v[i,:,:], w[i,:,:],
                        ax,
                        title='h=%2.1f km' %z1d[i],
                        xmin=xmin, xmax=xmax,
                        ymin=ymin, ymax=ymax,
                        wmin=wmin, wmax=wmax,
                        scale=scale,
                        cmap=cmap,
                        cmap_label=cmap_label,
                        df_meteor=df_meteor_h,
                        )
        
        if qk is None:
            continue
        
        if i==0:
            qk1 = ax.quiverkey(qk, 0.95, 0.95, 50, r'$50 \frac{m}{s}$',
                               labelpos='E',
                                coordinates='figure',
                               color='b')
            
            ax.legend()
            
            plt.legend(loc='upper left')
    
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

def plot_mean_winds_grads(t, h, ug,
                        figfile='./test.png',
                        vmins=None,
                        vmaxs=None,
                        cmap='seismic',
                        xlabel='UTC (h)',
                        ylabel='Altitude (km)',
                        titles=[
                                r'$u_0$ (m/s)', r'$v_0$ (m/s)', r'$w_0$ (m/s)',
                                r'$u_x$ (m/s/m)', r'$v_x$ (m/s/m)', r'$Div$ (m/s/m)', 
                                r'$u_y$ (m/s/m)', r'$v_y$ (m/s/m)', r'$Vor$ (m/s/m)', 
                                r'$u_z$ (m/s/m)', r'$v_z$ (m/s/m)', r'$w_z$ (m/s/m)',
                                ]
                        ):
    '''
    ug    :    (
    '''
    vmin = None
    vmax = None
    
    num = epoch2num(t)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    formatter = mdates.ConciseDateFormatter(locator)
    
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(16,8))
    
    axs_flat = axs.flat
    
    images = {}
    for i in range(12):
        
        ax = axs_flat[i]
        
        if vmins is not None: vmin = vmins[i]
        if vmaxs is not None: vmax = vmaxs[i]
        
        d = ug[i].T
        
        im = ax.pcolormesh(num, h, d, cmap=cmap, vmin=vmin, vmax=vmax)
        
        images[i] = im    
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle='--')
        
        ax.set_title(titles[i])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    
    for ax in axs_flat:
        ax.label_outer()
    
    for i, im in images.items():
        plt.colorbar(im, ax=axs_flat[i])
    
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
    
def plot_mean_winds(t, h, u, v, w,
                    figfile=None,
                    vmins=None,
                    vmaxs=None,
                    cmap='seismic',
                    ylabel='Altitude (km)',
                    xlabel="UT",
                    histogram=False,
                    df_ref=None,
                    titles = ['Zonal wind', 'Meridional wind', 'Vertical wind'],
                    bins=40
                    ):
    
    vmin = None
    vmax = None
    
    num = epoch2num(t)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    
    ncols = 1
    if histogram:
        ncols = 2
    
    us = [u, v, w]
    
    if df_ref is not None:
        
        alt_ref = df_ref['alt']
        valid_h = (alt_ref >= h.min()) & (alt_ref <= h.max())
        
        us_ref = [df_ref['u0'][valid_h], df_ref['v0'][valid_h], df_ref['w0'][valid_h]]

    # fig, axs = plt.subplots(3, ncols, figsize=(8,6), squeeze=False, sharex=False)
    
    axs = []
    images = {}
    
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    
    _ = plt.figure(figsize=(8,6))
    
    for i in range(3):
        
        data = us[i]
        std = np.nanstd(data)
        
        if vmins is not None: vmin = vmins[i]
        if vmaxs is not None: vmax = vmaxs[i]
        
        # bins = np.linspace(vmin, vmax, 40)
        
        # ax = axs[i,0]
        ax = plt.subplot2grid( (3,2+ncols), (i,0), colspan=3, rowspan=1 )
        
        ax.set_title(titles[i])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xlim(np.min(num), np.max(num))
        
        im = ax.pcolormesh(num, h, data.T, cmap=cmap, vmin=vmin, vmax=vmax)
        
        images[i] = im
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_ticklabels([])
        
        ax.grid(True, linestyle='--')
        
        
        ax.text(0.85, 0.85, "std=%3.2f" %std,
                bbox=bbox,
                transform=ax.transAxes
                )
        
        axs.append(ax)
        
        if histogram:
            # ax0 = axs[i,1]
            ax0 = plt.subplot2grid( (3,2+ncols), (i,3), colspan=1, rowspan=1 )
            ax0.set_xlabel('%s (m/s)' %titles[i])
            
            d = data[np.isfinite(data)]
            
            # if i==2: bins = 40
                
            _, edges, _ = ax0.hist(d, bins=bins,
                                   # density=True,
                                   color='grey',
                                   alpha=0.5,
                                   zorder=1,
                                   label='HYPER')
            ax0.grid(True, linestyle='--')
            
            max_edge = np.max( np.abs(edges) )
            
            ax0.set_xlim(-max_edge, max_edge)
            ax0.set_ylabel('Counts')
            
            # if i==2: bins = np.linspace(vmin, vmax, 30)
            
            if df_ref is not None:
                d = us_ref[i]
                d = d[np.isfinite(d)]
                
                ax0.hist(d, bins=bins,
                         # density=True,
                         color='c',
                         alpha=0.3,
                         zorder=0,
                         label='C.2022')
            
            # if i == 0:
            #     ax0.set_title('Histogram')
            #     ax0.legend(fontsize=10)
            
            ax0.locator_params(tight=True, nbins=4)
            
    
    for ax in axs:
        ax.label_outer()
    
    for i, im in images.items():
        plt.colorbar(im, ax=axs[i], label='m/s')
        
    plt.tight_layout(
                     pad=0.1, 
                     rect=[0.02,0.0,1.0,1.0]
                     )
    
    axs[-1].annotate('@jmu',
                     xy = (1.05, -0.2),
                     xycoords='axes fraction',
                     ha='left',
                     va="top",
                     fontsize=8,
                     alpha=0.5,
                     weight='bold')
    
    if figfile is None:
        plt.show()
    else:
        plt.savefig(figfile)
        
    plt.close("all")
    

def plot_field(lons, lats, alts,
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
               zdecimation=1,
               apply_nan_mask=True,
               fontsize=18,
               labelsize=14,
               coord='xyz',
               alpha_meteors=0.3,
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
        
    # LON, LAT = np.meshgrid(lons, lats)
    
    xlabel = 'EW direction (km)'
    ylabel = 'NS direction (km)'
    
    if coord != 'xyz':
        xlabel = r'Longitude ($^\circ$)'
        ylabel = r'Latitude ($^\circ$)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    meteor_size = 3
    
    for zi in range(0,nz,zdecimation):
        
        if nrows>1:
            ve = fields_est[0][zi]
                
            if np.all(np.isnan(ve)):
                continue
                
        filename = os.path.join(path, "wind_field_%2.1f_%s.png" %(alts[zi], prefix, ) )
        
        prefix = prefix.replace('_', ' ')
        
        fig = plt.figure(figsize=(4*nfields,4*nrows))
        plt.suptitle(r'%s - Altitude=$%2.1f~km$' %(prefix, alts[zi]), fontsize=fontsize)
        
        if df_sampling is not None:
            df_sampling_z = df_sampling[ np.abs(df_sampling['heights'] - alts[zi]) <= 1]#*1e3 ]
        
            if coord == 'xyz':
                samp_lons = df_sampling_z['x'].values*1e-3
                samp_lats = df_sampling_z['y'].values*1e-3
            else:
                samp_lons = df_sampling_z['lons'].values
                samp_lats = df_sampling_z['lats'].values
        
        
        ax = None
        for iplot in range(nfields):
            
            mask = None
            if nrows>1:
                ax = fig.add_subplot(nrows, nfields, nfields+iplot+1,
                                     sharex=ax, sharey=ax) #, projection='3d')
                
                if fields_est[iplot] is None: continue
                
                ve = fields_est[iplot][zi]
                
                if np.all(np.isnan(ve)):
                    break
                
                vmin = vmins[iplot]
                vmax = vmaxs[iplot]
                
                if vmin is None: vmin = np.nanmin(ve)
                if vmax is None: vmax = np.nanmax(ve)
                
                im = ax.pcolormesh(lons, lats, ve,
                                  cmap=cmap,
                                  vmin=vmin,
                                  vmax=vmax)
                
                std = np.nanstd(ve)
                
                # ax.text(0.05, 0.95, r'$std=%.2e$' %std,
                #         transform=ax.transAxes, fontsize=14,
                #         verticalalignment='top',
                #         bbox=props
                #         )
                
                if df_sampling is not None:
                    ax.plot(samp_lons, samp_lats, 'kx', alpha=alpha_meteors, markersize=meteor_size)
                
                # im = ax.plot_surface(LON, LAT, ve,
                #                  cmap=cmap,
                #                  vmin=vmin,
                #                  vmax=vmax,
                #                  alpha=alpha)
                #
                # ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x')
                # ax.set_zlim(vmin,vmax)
                
                # ax.set_xlim(-100, 100)
                # ax.set_ylim(-100, 100)
                
                ax.set_xlabel(xlabel, fontsize=labelsize)
                ax.set_ylabel(ylabel, fontsize=labelsize)
                ax.set_title(r'$%s_{hyper}$' %(titles[iplot]), fontsize=fontsize)
                
                ax.label_outer()
                
                plt.colorbar(im, ax=ax)
                
                if apply_nan_mask:
                    mask = np.isnan(ve)
            
            ax = fig.add_subplot(nrows, nfields, iplot+1,
                                 sharex=ax, sharey=ax) #, projection='3d')
        
            vmin = vmins[iplot]
            vmax = vmaxs[iplot]
            
            if vmin is None: vmin = np.nanmin(fields[iplot])
            if vmax is None: vmax = np.nanmax(fields[iplot])
            
            
            v = fields[iplot][zi]
            
            # v = gaussian(v, 3)
            
            if mask is not None:
                v[mask] = np.nan
            
            im = ax.pcolormesh(lons, lats, v,
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax)
            
            std = np.nanstd(v)
                
            # ax.text(0.05, 0.95, r'$std=%.2e$' %std,
            #         transform=ax.transAxes, fontsize=14,
            #         verticalalignment='top',
            #         bbox=props
            #         )
                
            if df_sampling is not None:
                ax.plot(samp_lons, samp_lats, 'kx', alpha=alpha_meteors, markersize=meteor_size)
            
            # 
            # im = ax.plot_surface(LON, LAT, v,
            #                      cmap=cmap,
            #                      vmin=vmin,
            #                      vmax=vmax,
            #                      alpha=alpha)
            #
            # ax.scatter(samp_lons, samp_lats, samp_lats*0 + vmin, marker='x')
            # ax.set_zlim(vmin, vmax)
            
            ax.set_xlabel(xlabel, fontsize=labelsize)
            ax.set_ylabel(ylabel, fontsize=labelsize)
            ax.set_title(r'$%s_{true}$' %(titles[iplot]), fontsize=fontsize)
            
            # ax.set_xlim(-100, 100)
            # ax.set_ylim(-100, 100)
            
            ax.label_outer()
            
            plt.colorbar(im, ax=ax)
            
            if nrows>2:
                ax = fig.add_subplot(nrows, nfields, 2*nfields+iplot+1,
                                     sharex=ax, sharey=ax) #,projection='3d')
        
                ve_r = ve - np.nanmean(ve)
                v_r  = v - np.nanmean(v)
                
                idx = np.where(np.isfinite(ve))
                
                f1 = v_r[idx]
                f2 = ve_r[idx]
                
                cc = np.dot(f1, f2)/np.sqrt(np.nansum(f1**2)*np.nansum(f2**2))
                
                # tv1 = np.mean(np.abs(f1))
                # tv2 = np.mean(np.abs(f2))
                v_exp = np.sqrt(np.nanmean(v_r**2))
                
                rel_err = np.abs(v_r - ve_r)/v_exp
                
                # rmse     = np.sqrt( np.nanmean(f**2) )
                
                im = ax.pcolormesh(lons, lats, rel_err,
                                  cmap='binary',
                                  vmin=0,
                                  vmax=1)
                
                ax.text(0.05, 0.95, r'$\mathrm{CC}=%.2f$' %cc,
                        transform=ax.transAxes, fontsize=14,
                        verticalalignment='top',
                        bbox=props
                        )
                
                ax.set_xlabel(xlabel, fontsize=labelsize)
                ax.set_ylabel(ylabel, fontsize=labelsize)
                ax.set_title('RAE{%s}' %(titles[iplot]), fontsize=fontsize)
                
                # t = '[CC=%3.2f, TV1=%2.1f, TV2=%2.1f]' %(corr, tv1, tv2)
                
                # ax.set_xlim(-100, 100)
                # ax.set_ylim(-100, 100)
                if df_sampling is not None:
                    ax.plot(samp_lons, samp_lats, 'rx', alpha=alpha_meteors, markersize=meteor_size)
                
                ax.label_outer()
                
                plt.colorbar(im, ax=ax)
        
        ax.set_xlim(np.min(lons), np.max(lons))
        ax.set_ylim(np.min(lats), np.max(lats))
            
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
def plot_rmses(times, y, rmses, labels,
                     figname=None,
                     cmap='seismic',
                     vmins=None,
                     vmaxs=None):
    
    '''
    Inputs:
        times    :    1D ARRAY
        y    :    1D ARRAY
        rmses   :    dimension [ntimes, nfields, nalt]
        
        labels    :    dimension []
                        :
        
    '''
    
    rmses =  np.array(rmses)
    
    nplots = rmses.shape[1]
    
    x = epoch2num(times)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    
    fig, axs = plt.subplots(nplots, 1, sharex=True, sharey=True, figsize=(8,6))
    
    for i in range(nplots):
        
        ax = axs[i]
        rmse = rmses[:,i,:]
        label = labels[i]
        
        # rmse = gaussian(rmse, [3,0])
        
        if vmaxs is not None: vmax = vmaxs[i]
        else: vmax = None
        
        if vmins is not None: vmin = vmins[i]
        else: vmin = None
        
        im = ax.pcolormesh(x, y, rmse.T,
                           cmap=cmap,
                           vmin=vmin, 
                           vmax=vmax)
        
        ax.set_title(label.lower())
        plt.colorbar(im, ax=ax, label='m/s')
        
        ax.grid(True)
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    
    if figname is not None:
        plt.savefig(figname)
    else:
        plt.show()
        
    plt.close()