'''
Created on 30 Aug 2022

@author: mcordero
'''

import numpy as np

from skimage.filters import window

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['image.cmap'] = 'jet'

from atmospheric_models.DNS  import DNSReader

def compute_spectra(d, idx=None, win='blackman'):
    
    u = d['u']
    v = d['v']
    w = d['w']
    
    nz, ny, nx = u.shape
    
    if idx is None:
        idx = int(nz/2)
    
    w = window(win, u[idx].shape)
    
    Fu = np.fft.rfft(u[idx]*w, norm='forward', axis=1)
    Fv = np.fft.rfft(v[idx]*w, norm='forward', axis=1)
    
    return(Fu, Fv)

def plot_spectrum(u, v, Fu, Fv, umin=-10, umax=10, vmin=1e-14, vmax=1e2, cmap_u='RdBu_r'):
    
    pow_u = np.power(np.abs(Fu), 2)
    pow_v = np.power(np.abs(Fv), 2)
    pow_uv = pow_u+pow_v
    
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    
    nx, ny = Fu.shape
    
    x = np.arange(nx)
    y = np.arange(ny)+1
    
    # plt.figure(figsize=(9,6))
    
    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(9,6))
    
    axs[0,0].pcolormesh( u, vmin=umin, vmax=umax, cmap=cmap_u )
    
    axs[0,1].pcolormesh( v, vmin=umin, vmax=umax, cmap=cmap_u )
    
    axs[1,0].pcolormesh(y, x, pow_u, norm=norm )
    
    axs[1,1].pcolormesh(y, x, pow_v, norm=norm )
    
    axs[1,2].pcolormesh(y, x, pow_uv, norm=norm  )
    
    axs[2,0].plot(y, np.mean(pow_u, axis=0), 'o--' )
    
    axs[2,1].plot(y, np.mean(pow_v, axis=0), 'o--' )
    
    axs[2,2].plot(y, np.mean(pow_uv, axis=0), 'o--' )
    
    for ax in axs[1,:]:
        ax.set_xscale('log')
        ax.set_xlim(1,500)
        
    for ax in axs[2,:]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(1,500)
        ax.set_ylim(vmin, vmax)
        
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_spectra(Fus, Fvs, umin=-10, umax=10, vmin=1e-14, vmax=1e2, cmap_u='RdBu_r', labels=None):
    
    Fus = np.array(Fus)
    Fvs = np.array(Fvs)
    
    n, nt, nx, ny = Fus.shape
    
    x = np.arange(nx)
    y = np.arange(ny)+1
    
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    # fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12,6))
    
    fig, axs = plt.subplot_mosaic("03;13;23", figsize=(12,6))
    axs = [axs[key] for key in sorted(axs.keys())]
    
    colors = ['r', 'g', 'b']
    for i in range(n):
    
        pow_u = np.power(np.abs(Fus[i]), 2)
        pow_v = np.power(np.abs(Fvs[i]), 2)
        pow_uv = pow_u+pow_v
        
        mean_y = np.mean(pow_uv, axis=1)
        mean_t = np.mean(mean_y, axis=0)
        
        axs[i].pcolormesh(y, x, pow_uv[0], norm=norm  )
        
        label=''
        if labels is not None:
            label = labels[i]
        
        for j in range(nt):
            axs[3].plot(y, mean_y[j], '-', color=colors[i], alpha=0.1) 
               
        axs[3].plot(y, mean_t, 'o--', label=label, color=colors[i])
        
        if i == 0:
            noise_level = 0.3*np.mean(mean_t)
            axs[3].axhline(noise_level, linestyle='--', color='k')
        
        for ax in axs[0:3]:
            ax.set_xscale('log')
            ax.set_xlim(1,500)
        
        
        ax = axs[-1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(1,500)
        ax.set_ylim(vmin, vmax)
        
        ax.legend()
        
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    import argparse
    
    dns_path = '/Users/radar/Data/IAP/Models/DNS/NonStratified'
    hx1_path = r'/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/old/nnIPINN_5.03_specX1/final/None'
    hx10_path = r"/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/old/nnIPINN_15.03_specX10/final/None"
    
    
    # hx1_path  = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnDEEPONET_11.00/final64x64_ur=1.0e-07/None/outs"
    #
    #
    # # hx1_path  = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnDEEPONET_11.02/final128x128_ur=1.0e-05/3500/outs"
    # hx10_path = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnDEEPONET_11.02/final256x256_ur=1.0e-05/3500/outs"
    #
    # # hx1_path   = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnDEEPONET_11.03/final3x64_ur=1.0e-05/None/outs"
    # hx1_path   = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnDEEPONET_11.03/final3x64_ur=1.0e-05/None/outs"
    # hx10_path  = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnDEEPONET_11.03/final4x64_ur=1.0e-05/None/outs"
    #
    # hx1_path  = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnDEEPONET_11.04/final64_ur=1.0e-05/None/outs"
    # # hx1_path = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnDEEPONET_1.04/final_ur=1.0e-07/None/outs"
    # hx1_path = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnRPINN_1.05/final_ur=1.0e-03/None/outs"
    hx10_path = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnRPINN_11.06/final_ur=1.0e-05/None/outs"
    
    
    hx1_path  = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnRESPINN_3.00/dl05n256dur=1.0e-07/None/outs"
    hx10_path = "/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/winds/nnRESPINN_13.00/dl05n256dur=1.0e-07/None/outs"
    
    dns = DNSReader(dns_path, dimZ=100)
    hx1 = DNSReader(hx1_path, dimZ=100)
    hx10 = DNSReader(hx10_path, dimZ=100)
    
    Fus0 = []
    Fvs0 = []
    
    Fus1 = []
    Fvs1 = []
    
    Fus2 = []
    Fvs2 = []
    
    i =0
    while True:
        df = dns.read_next_block()
        if df is None: break
        
        Fu0, Fv0 = compute_spectra(df)
        
        
        df = hx1.read_next_block()
        if df is None: break
        
        Fu1, Fv1 = compute_spectra(df)
        
        
        
        df = hx10.read_next_block()
        if df is None: break
        
        Fu2, Fv2 = compute_spectra(df)
        
        
        Fus0.append(Fu0)
        Fvs0.append(Fv0)
        
        Fus1.append(Fu1)
        Fvs1.append(Fv1)
    
    
        Fus2.append(Fu2)
        Fvs2.append(Fv2)
        
        i += 1
        if i == 5:
            break
    
    # Fu0 = np.mean(Fus0, axis=0)
    # Fv0 = np.mean(Fvs0, axis=0)
    #
    # Fu1 = np.mean(Fus1, axis=0)
    # Fv1 = np.mean(Fvs1, axis=0)
    #
    # Fu2 = np.mean(Fus2, axis=0)
    # Fv2 = np.mean(Fvs2, axis=0)
    
    plot_spectra([Fus0, Fus1, Fus2],
                 [Fvs0, Fvs1, Fvs2],
                 labels=['DNS', 'Hx1', 'Hx10'])