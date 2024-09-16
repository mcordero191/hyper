'''
Created on 29 Nov 2022

@author: mcordero
'''

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from scipy.stats import spearmanr
from matplotlib.ticker import NullFormatter

def plot_2dhist(X, Y,
                thresholdX=None, thresholdY=None,
                logY=None, logX=None,
                filterX=None, filterY=None,
                bins=100,
                labelX='X',
                labelY='Y',
                figname='./test.png'):

    # vectorize image data.
    x_data = X.ravel()
    y_data = Y.ravel()

    # decide which points to include
    x_in = np.isfinite(x_data)
    y_in = np.isfinite(x_data)

    if thresholdX != None:
        x_thr = x_data>float(thresholdX)
    else:
        x_thr = True
    if thresholdY != None:
        y_thr = y_data>float(thresholdY)
    else:
        y_thr = True

    if filterX != None:
        x_fil = x_data!=filterX
    else:
        x_fil = True
    if filterY != None:
        y_fil = y_data!=filterY
    else:
        y_fil = True

    y = y_data[x_in & y_in & x_thr & y_thr & x_fil & y_fil ]
    x = x_data[x_in & y_in & x_thr & y_thr & x_fil & y_fil ]


    # log scale if you like.
    if logY != None:
        y = np.log(y)
    if logX != None:
        x = np.log(x)
    

    # start with a rectangular Figure
    mainFig = plt.figure(1, figsize=(8,8), facecolor='white')
    
    # define some gridding.
    axHist2d = plt.subplot2grid( (9,9), (1,0), colspan=8, rowspan=8 )
    axHistx  = plt.subplot2grid( (9,9), (0,0), colspan=8 )
    axHisty  = plt.subplot2grid( (9,9), (1,8), rowspan=8 )

    ax_2dhist([axHist2d, axHistx, axHisty],
              X, Y,
              bins=bins,
              labelX=labelX, labelY=labelY)
    
    # set the window title
    mainFig.canvas.set_window_title( (labelX + ' vs. ' + labelY) )
    
    # actually draw the plot.
    plt.savefig(figname)
    
def ax_2dhist(axs,
              X, Y,
                thresholdX=None, thresholdY=None,
                logY=None, logX=None,
                filterX=None, filterY=None,
                bins=100,
                labelX='X',
                labelY='Y',
                norm=False,
                log_scale=False,
                cmap="jet",
                ):

    # vectorize image data.
    x_data = X.ravel()
    y_data = Y.ravel()

    # decide which points to include
    x_in = np.isfinite(x_data)
    y_in = np.isfinite(y_data)

    if thresholdX != None:
        x_thr = x_data>float(thresholdX)
    else:
        x_thr = True
    if thresholdY != None:
        y_thr = y_data>float(thresholdY)
    else:
        y_thr = True

    if filterX != None:
        x_fil = x_data!=filterX
    else:
        x_fil = True
    if filterY != None:
        y_fil = y_data!=filterY
    else:
        y_fil = True

    y = y_data[x_in & y_in & x_thr & y_thr & x_fil & y_fil ]
    x = x_data[x_in & y_in & x_thr & y_thr & x_fil & y_fil ]
    
    # log scale if you like.
    if logY != None:
        y = np.log(y)
    if logX != None:
        x = np.log(x)
    
    # define some gridding.
    # axHist2d = plt.subplot2grid( (9,9), (1,0), colspan=8, rowspan=8 )
    # axHistx  = plt.subplot2grid( (9,9), (0,0), colspan=8 )
    # axHisty  = plt.subplot2grid( (9,9), (1,8), rowspan=8 )
    
    axHist2d = axs[0]
    axHistx  = axs[1]
    axHisty  = axs[2]
    
    # the 2D Histogram, which represents the 'scatter' plot:
    H, xedges, yedges = np.histogram2d( x, y, bins=bins )
    
    if norm:
        # Hx = np.sum(H, axis=0)
        Hy = np.sum(H, axis=1)
        # my = np.nanmean(Hy)
        # Hy = np.where(np.isfinite(Hy), Hy/Hy[:,None], Hy/my)
        H  = H/Hy[:,None] #Rescale again
    
    if log_scale:
        H = np.log10(H)
    
    axHist2d.imshow(H.T, interpolation=None, aspect='auto', cmap=cmap)
    axHist2d.set_ylim( [ axHist2d.get_ylim()[1], axHist2d.get_ylim()[0] ] )
    
    axHist2d.grid(True)
    
    # set titles
    axHist2d.set_xlabel(labelX)#, fontsize=16)
    axHist2d.set_ylabel(labelY)#, fontsize=16)
    
    f_indx = interp1d(xedges, np.arange(xedges.shape[0]), bounds_error=False, fill_value="extrapolate")
    f_indy = interp1d(yedges, np.arange(yedges.shape[0]), bounds_error=False, fill_value="extrapolate")
    
    axHist2d.plot(f_indx(yedges[:]), np.arange(yedges.shape[0]), '--', color='#000000', alpha=0.5, linewidth=3)
    
    min_vx = xedges.min()
    max_vx = xedges.max()
    
    min_vy = yedges.min()
    max_vy = yedges.max()
    
    xmin = f_indx(min_vx)
    xmax = f_indx(max_vx)
    
    ymin = f_indy(min_vy)
    ymax = f_indy(max_vy)
    
    axHist2d.set_xlim( xmin, xmax )
    axHist2d.set_ylim( ymin, ymax )
    
    # label 2d hist axes
    myValues_x = np.arange(1e2*min_vx, 1e2*max_vx, (max_vx-min_vx)*1e2/5, dtype=np.int32)/1e2
    myValues_y = np.arange(1e2*min_vy, 1e2*max_vy, (max_vy-min_vy)*1e2/5, dtype=np.int32)/1e2
    
    myXTicks = f_indx(myValues_x)
    myYTicks = f_indy(myValues_y)
    
    mask_x = np.isfinite(myXTicks)
    mask_y = np.isfinite(myYTicks)
    
    axHist2d.set_xticks(myXTicks[mask_x])
    axHist2d.set_yticks(myYTicks[mask_y])
    axHist2d.set_xticklabels(myValues_x[mask_x])
    axHist2d.set_yticklabels(myValues_y[mask_y])
    
    xi = f_indx( max_vx - (max_vx - min_vx)*0.05)
    yi = f_indy( max_vy - (max_vy - min_vy)*0.05)
    
    # print some correlation coefficients at the top of the image.
    text = r'r=%3.2f, $\rho$=%3.2f' %( np.corrcoef( x, y )[1][0], spearmanr( x, y )[0])
    axHist2d.text(xi, yi, text, style='italic', fontsize=10, color='w',
                  ha='right', va='top',
                  bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    
    nullfmt   = NullFormatter()
    
    if axHistx is not None:
        # make histograms for x and y seperately.
        axHistx.hist(x, bins=xedges, facecolor='#0000AA', alpha=0.5, edgecolor='None' )
        
        # set axes
        axHistx.set_xlim( [xedges.min(), xedges.max()] )
    
        # remove some labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHistx.yaxis.set_major_formatter(nullfmt)
        
        # remove some axes lines
        axHistx.spines['top'].set_visible(False)
        axHistx.spines['right'].set_visible(False)
        axHistx.spines['left'].set_visible(False)
        
        # remove some ticks
        axHistx.set_xticks([])
        axHistx.set_yticks([])
        
        axHistx.set_xlim( min_vx, max_vx )
    
    if axHisty is not None:
        
        axHisty.hist(y, bins=yedges, facecolor='#0000AA', alpha=0.5, orientation='horizontal', edgecolor='None')
        
        axHisty.set_ylim( [yedges.min(), yedges.max()] )
        
        axHisty.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        
        axHisty.spines['top'].set_visible(False)
        axHisty.spines['bottom'].set_visible(False)
        axHisty.spines['right'].set_visible(False)
        
        axHisty.set_xticks([])
        axHisty.set_yticks([])
        
        axHisty.set_ylim( min_vy, max_vy )

def ax_2dhist_simple(X, Y, ax, bins=40,
                     xmin=None, xmax=None,
                     ymin=None, ymax=None,
                     vmin=None, vmax=None,
                     cmap='jet',
                     normalization_x=False,
                     log_scale=False,
                     ):
    
    if xmin is None: xmin = np.nanmin(X)
    if xmax is None: xmax = np.nanmax(X)
    if ymin is None: ymin = np.nanmin(Y)
    if ymax is None: ymax = np.nanmax(Y)
    
    # vectorize image data.
    x_data = X.ravel()
    y_data = Y.ravel()
    
    H, xedges, yedges = np.histogram2d(x_data, y_data,
                   bins=bins,
                   range=[[xmin, xmax],[ymin, ymax]])
    #
    # if normalization_x:
    #     H /= np.nansum(H, axis=0)[None,:]
    
    # H = np.log10(H)
    
    ##the min and max values of all histograms
    if vmin is None: vmin = np.min(H)
    if vmax is None: vmax = np.max(H)
    
    norm = None
    if log_scale:
        
        norm = mpl.colors.LogNorm()
        vmin=None
        vmax=None
        
    im = ax.pcolormesh(xedges, yedges, H.T, vmin=vmin, vmax=vmax, cmap=cmap,
                       norm = norm)
    
    # im = plt.hist2d(x_data, y_data,
    #                bins=bins,
    #                range=[[xmin, xmax],[ymin, ymax]],
    #                cmin=vmin, cmax=vmax,
    #                cmap=cmap)
    
    plt.colorbar(im, ax=ax)
    
    return(im)
    
if __name__ == '__main__':
    
    nx = 100
    ny = 100
    
    X = np.random.randn(nx,ny)
    Y = np.random.randn(nx,ny)
    
    fig, axs = plt.subplots(1, 3)
    ax_2dhist(X=X, Y=Y, axs=axs, bins=10, norm=True)
    
    plt.show()
    