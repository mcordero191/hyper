'''
Created on 30 Aug 2022

@author: mcordero
'''
import os, glob
import time, datetime
import numpy as np

import tensorflow as tf
DTYPE='float32'

import matplotlib.pyplot as plt
# plt.style.use("fivethirtyeight")
# plt.style.use('seaborn')

from pinn import hyper as pinn

def plot_weight_hist(filename, figpath, postfix='', bins=25, log_file=None):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    spec_layer = nn.model.spec_layer0
    R = spec_layer.R
    w = spec_layer.w
    
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    
    axs[0,0].hist(np.ravel(R), bins=bins, label='R', alpha=0.4, density=True)
    axs[0,0].hist(np.ravel(w), bins=bins, label='w', alpha=0.4, density=True)
    
    axs[0,0].legend()
    
    for layer in nn.model.nlb_layers0:
        w = layer.w
        axs[0,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=0.4, density=True)
        axs[0,1].legend()
        
    for layer in nn.model.u_layers:
        w = layer.w
        axs[1,0].hist(np.ravel(w), bins=bins, label=layer.name, alpha=0.4, density=True)
        axs[1,0].legend()
        
    for layer in nn.model.w_layers:
        w = layer.w
        axs[1,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=0.4, density=True)    
        axs[1,1].legend()
    
    plt.tight_layout()
    figfile = os.path.join(figpath, 'weight_%s.png' %postfix)
    
    plt.savefig(figfile)

def plot_weight_hist_v5(filename, figpath,
                        postfix='',
                        bins=20,
                        log_file=None,
                        density=False,
                        alpha=0.3):
    
    from pinn import pinn_v5_DMD as pinn
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    t_emb = nn.model.t_emb
    t_fnn = nn.model.t_fnn
    t_u = nn.model.t_u
    t_v = nn.model.t_v
    t_w = nn.model.t_w
    
    # x_emb = nn.model.x_emb
    # x_fnn = nn.model.x_fnn
    # x_u = nn.model.x_u
    # x_v = nn.model.x_v
    # x_w = nn.model.x_w
    
    fig, axs = plt.subplots(2, 3, figsize=(10,10))
    
    xc = t_emb.get_weights()[0]
    h = t_emb.get_weights()[1]
    
    ##t layers ###
    axs[0,0].hist(np.ravel(xc), bins=bins, label='t0', alpha=alpha, density=density)
    axs[0,0].hist(np.ravel(h), bins=bins, label='h_t', alpha=alpha, density=density)
    
    for layer in t_fnn.layers:
        w = layer.get_weights()[::2]
        axs[0,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=alpha, density=density)
        
    w = t_u.get_weights()[::2]
    axs[0,2].hist(np.ravel(w), bins=bins, label=t_u.name, alpha=alpha, density=density)
    
    w = t_v.get_weights()[::2]
    axs[0,2].hist(np.ravel(w), bins=bins, label=t_v.name, alpha=alpha, density=density)
    
    w = t_w.get_weights()[::2]
    axs[0,2].hist(np.ravel(w), bins=bins, label=t_w.name, alpha=alpha, density=density)
    
    ##X layers ###
    # axs[1,0].hist(np.ravel(x_emb.xc), bins=bins, label='x0', alpha=alpha, density=density)
    # axs[1,0].hist(np.ravel(x_emb.h), bins=bins, label='h_x', alpha=alpha, density=density)
    #
    # for layer in x_fnn.layers:
    #     w = layer.get_weights()[::2]
    #     axs[1,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=alpha, density=density)
    #
    # w = x_u.get_weights()[::2]
    # axs[1,2].hist(np.ravel(w), bins=bins, label=x_u.name, alpha=alpha, density=density)
    #
    # w = x_v.get_weights()[::2]
    # axs[1,2].hist(np.ravel(w), bins=bins, label=x_v.name, alpha=alpha, density=density)
    #
    # w = x_w.get_weights()[::2]
    # axs[1,2].hist(np.ravel(w), bins=bins, label=x_w.name, alpha=alpha, density=density)
    
    for ax in axs.flatten():
        ax.legend()
        ax.set_xlim(-1,1)
    
    axs[0,0].set_xlim(-1.2,2.2)
    axs[1,0].set_xlim(-1.2,2.2)
    
    plt.tight_layout()
    figfile = os.path.join(figpath, 'weight_%s_%s.png' %(postfix, log_file) )
    
    plt.savefig(figfile)
    
def plot_weight_hist_v5_0(filename, figpath,
                        postfix='',
                        bins=20,
                        log_file=None,
                        density=False,
                        alpha=0.3):
    
    from pinn import pinn_v5_DMD as pinn
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    t_emb = nn.model.t_emb
    t_fnn = nn.model.t_fnn
    t_u = nn.model.t_u
    t_v = nn.model.t_v
    t_w = nn.model.t_w
    
    x_emb = nn.model.x_emb
    x_fnn = nn.model.x_fnn
    x_u = nn.model.x_u
    x_v = nn.model.x_v
    x_w = nn.model.x_w
    
    fig, axs = plt.subplots(2, 3, figsize=(10,10))
    
    
    ##t layers ###
    axs[0,0].hist(np.ravel(t_emb.xc), bins=bins, label='t0', alpha=alpha, density=density)
    axs[0,0].hist(np.ravel(t_emb.h), bins=bins, label='h_t', alpha=alpha, density=density)
    
    for layer in t_fnn.layers:
        w = layer.get_weights()[::2]
        axs[0,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=alpha, density=density)
        
    w = t_u.get_weights()[::2]
    axs[0,2].hist(np.ravel(w), bins=bins, label=t_u.name, alpha=alpha, density=density)
    
    w = t_v.get_weights()[::2]
    axs[0,2].hist(np.ravel(w), bins=bins, label=t_v.name, alpha=alpha, density=density)
    
    w = t_w.get_weights()[::2]
    axs[0,2].hist(np.ravel(w), bins=bins, label=t_w.name, alpha=alpha, density=density)
    
    ##X layers ###
    axs[1,0].hist(np.ravel(x_emb.xc), bins=bins, label='x0', alpha=alpha, density=density)
    axs[1,0].hist(np.ravel(x_emb.h), bins=bins, label='h_x', alpha=alpha, density=density)
    
    for layer in x_fnn.layers:
        w = layer.get_weights()[::2]
        axs[1,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=alpha, density=density)
        
    w = x_u.get_weights()[::2]
    axs[1,2].hist(np.ravel(w), bins=bins, label=x_u.name, alpha=alpha, density=density)
    
    w = x_v.get_weights()[::2]
    axs[1,2].hist(np.ravel(w), bins=bins, label=x_v.name, alpha=alpha, density=density)
    
    w = x_w.get_weights()[::2]
    axs[1,2].hist(np.ravel(w), bins=bins, label=x_w.name, alpha=alpha, density=density)
    
    for ax in axs.flatten():
        ax.legend()
        ax.set_xlim(-1,1)
    
    axs[0,0].set_xlim(-1.2,2.2)
    axs[1,0].set_xlim(-1.2,2.2)
    
    plt.tight_layout()
    figfile = os.path.join(figpath, 'weight_%s_%s.png' %(postfix, log_file) )
    
    plt.savefig(figfile)
    
    
def plot_weight_hist_v7(filename, figpath, postfix='',
                        bins=25, log_file=None,
                        xmin=-1.5, xmax=1.5, 
                        alpha=0.3,
                        density=False,
                        residual_layer=False,
                        histtype='barstacked'):
    
    from pinn import pinn_v7_subnets as pinn
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    nn_model = nn.model.net_mean
    
    if residual_layer:
        nn_model = nn.model.net_residuals
    
    kw_layer = nn_model.t_emb
    nn_layer = nn_model.t_fnn
    tu_layer = nn_model.t_u
    
    if residual_layer:
        tv_layer = nn_model.t_v
        tw_layer = nn_model.t_w
        xu_layer = nn_model.x_u
        xv_layer = nn_model.x_v
        xw_layer = nn_model.x_w
    
    w = kw_layer.weights[0]
    
    vmin = np.min(w)
    vmax = np.max(w)
    
    bins0 = np.linspace(vmin, vmax, num=bins)
    
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    
    axs[0,0].hist(np.ravel(w[0]), bins=bins0, label='Emb x0', alpha=alpha, density=density, histtype=histtype)
    axs[0,0].hist(np.ravel(w[1]), bins=bins0, label='Emb x1', alpha=alpha, density=density, histtype=histtype)
    axs[0,0].hist(np.ravel(w[2]), bins=bins0, label='Emb x2', alpha=alpha, density=density, histtype=histtype)
    axs[0,0].hist(np.ravel(w[3]), bins=bins0, label='Emb x3', alpha=alpha, density=density, histtype=histtype)
    
    
    w = nn_layer.weights[0]
    axs[0,1].hist(np.ravel(w), bins=bins, label=nn_layer.name, alpha=alpha, density=density)
    
        
    wu = tu_layer.weights[0]
    axs[1,0].hist(np.ravel(wu), bins=bins, label=tu_layer.name, alpha=alpha, density=density)
    
    if residual_layer:
        wv = tv_layer.weights[0]
        ww = tw_layer.weights[0]
        axs[1,0].hist(np.ravel(wv), bins=bins, label=tv_layer.name, alpha=alpha, density=density)
        axs[1,0].hist(np.ravel(ww), bins=bins, label=tw_layer.name, alpha=alpha, density=density)
        
        
        wu = xu_layer.weights[0]
        wv = xv_layer.weights[0]
        ww = xw_layer.weights[0]
        axs[1,1].hist(np.ravel(wu), bins=bins, label=xu_layer.name, alpha=alpha, density=density)   
        axs[1,1].hist(np.ravel(wv), bins=bins, label=xv_layer.name, alpha=alpha, density=density)   
        axs[1,1].hist(np.ravel(ww), bins=bins, label=xw_layer.name, alpha=alpha, density=density)  
    
      
    axs[0,0].legend()
    # axs[0,0].set_xlim(xmin, xmax)
    axs[0,1].legend()
    axs[0,1].set_xlim(xmin, xmax)
    axs[1,0].legend()
    axs[1,0].set_xlim(xmin, xmax)
    axs[1,1].legend()
    axs[1,1].set_xlim(xmin, xmax)
    
    axs[0,0].grid(True, linestyle='--')
    axs[0,1].grid(True, linestyle='--')
    axs[1,0].grid(True, linestyle='--')
    axs[1,1].grid(True, linestyle='--')
    
    plt.tight_layout()
    figfile = os.path.join(figpath, 'weight_%s_%s.png' %(postfix, log_file) )
    
    plt.savefig(figfile)


def plot_weight_hist_v8(filename, figpath, postfix='',
                        bins=8, log_file=None,
                        xmin=-1.5, xmax=1.5, 
                        alpha=0.3,
                        density=False,
                        residual_layer=False,
                        histtype='bar'):
    
    from pinn import pinn_v7_subnets as pinn
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.PINN()
    nn.restore(filename, log_index=log_file)
    
    nn_model = nn.model.net_mean
    
    if residual_layer:
        nn_model = nn.model.net_residuals
    
    kw_layer = nn_model.emb
    nn_layer = nn_model.kernel
    tu_layer = nn_model.linear
    
    if residual_layer:
        tv_layer = nn_model.t_v
        tw_layer = nn_model.t_w
        xu_layer = nn_model.x_u
        xv_layer = nn_model.x_v
        xw_layer = nn_model.x_w
    
    xc = kw_layer.xc
    h  = kw_layer.h
    
    # vmin = np.min(w)
    # vmax = np.max(w)
    
    # bins0 = np.linspace(vmin, vmax, num=bins)
    
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    
    axs[0,0].hist(xc.numpy().T, bins=bins, label=['Delay x0', 'Delay x1', 'Delay x2', 'Delay x3'], alpha=alpha, density=density, histtype=histtype)
    # axs[0,0].hist(np.ravel(xc[:,1]), bins=bins, label='Delays x1', alpha=alpha, density=density, histtype=histtype)
    # axs[0,0].hist(np.ravel(xc[:,2]), bins=bins, label='Delays x2', alpha=alpha, density=density, histtype=histtype)
    # axs[0,0].hist(np.ravel(xc[:,3]), bins=bins, label='Delays x3', alpha=alpha, density=density, histtype=histtype)
    
    
    axs[0,1].hist(h.numpy().T , bins=bins, label=['Width x0', 'Width x1', 'Width x2', 'Width x3'], alpha=alpha, density=density, histtype=histtype)
    # axs[1,1].hist(np.ravel(h[:,1]) , bins=bins, label='Width x1', alpha=alpha, density=density, histtype=histtype)
    # axs[1,1].hist(np.ravel(h[:,2]) , bins=bins, label='Width x2', alpha=alpha, density=density, histtype=histtype)
    # axs[1,1].hist(np.ravel(h[:,3]) , bins=bins, label='Width x3', alpha=alpha, density=density, histtype=histtype)
    
    
    ws = []
    labels = []
    for layer0 in nn_layer._flatten_layers(include_self=False):
        w0 = layer0.weights[0]
        ws.append(np.ravel(w0))
        labels.append(layer0.name)
        
    axs[1,0].hist(ws, bins=bins, label=labels, alpha=alpha, density=density)
    
    
    wu = tu_layer.w
    
    ws = []
    labels = []
    comp = ['u', 'v', 'w']
    
    for i in range(wu.shape[0]):
        w0 = wu[i]
        ws.append([w0])
        labels.append('Linear %s' %comp[i])
        
    axs[1,1].hist(ws, bins=bins, label=labels, alpha=alpha, density=density)
    
    if residual_layer:
        wv = tv_layer.weights[0]
        ww = tw_layer.weights[0]
        axs[1,0].hist(np.ravel(wv), bins=bins, label=tv_layer.name, alpha=alpha, density=density)
        axs[1,0].hist(np.ravel(ww), bins=bins, label=tw_layer.name, alpha=alpha, density=density)
        
        
        wu = xu_layer.weights[0]
        wv = xv_layer.weights[0]
        ww = xw_layer.weights[0]
        axs[1,1].hist(np.ravel(wu), bins=bins, label=xu_layer.name, alpha=alpha, density=density)   
        axs[1,1].hist(np.ravel(wv), bins=bins, label=xv_layer.name, alpha=alpha, density=density)   
        axs[1,1].hist(np.ravel(ww), bins=bins, label=xw_layer.name, alpha=alpha, density=density)  
    
      
    axs[0,0].legend()
    # axs[0,0].set_xlim(xmin, xmax)
    axs[0,1].legend()
    # axs[0,1].set_xlim(xmin, xmax)
    axs[1,0].legend()
    # axs[1,0].set_xlim(xmin, xmax)
    axs[1,1].legend()
    # axs[1,1].set_xlim(xmin, xmax)
    
    # axs[0,0].grid(True, linestyle='--')
    # axs[0,1].grid(True, linestyle='--')
    # axs[1,0].grid(True, linestyle='--')
    # axs[1,1].grid(True, linestyle='--')
    
    plt.tight_layout()
    figfile = os.path.join(figpath, 'weight_%s_%s.png' %(postfix, log_file) )
    
    plt.savefig(figfile)

def plot_weight_hist_v7_deeponet(filename, figpath,
                        postfix='',
                        bins=20,
                        log_file=None,
                        density=False,
                        alpha=0.3):
    
    from pinn import pinn_v7_spinn as pinn
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.restore(filename, log_index=log_file)
    
    t_emb = nn.model.get_layer('t_emb')
    t_fnn = nn.model.get_layer('t_dense')
    t_u = nn.model.get_layer('t_u')
    t_v = nn.model.get_layer('t_v')
    t_w = nn.model.get_layer('t_w')
    
    x_emb = nn.model.get_layer('x_emb')
    x_fnn = nn.model.get_layer('x_dense')
    # x_u = nn.model.x_u
    # x_v = nn.model.x_v
    # x_w = nn.model.x_w
    
    fig, axs = plt.subplots(2, 3, figsize=(10,10))
    
    layers = t_emb._layers
    
    for layer in layers:
        
        xc = layer.weights[0]
        h = layer.weights[1]
        
        ##t layers ###
        axs[0,0].hist(np.ravel(xc), bins=bins, label='t0', alpha=alpha, density=density)
        axs[0,0].hist(np.ravel(h), bins=bins, label='h_t', alpha=alpha, density=density)
    
    for layer in t_fnn.layers:
        w = layer.get_weights()[0]
        
        axs[0,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=alpha, density=density)
        
        wt = np.mean(np.abs(w), axis=1)
        axs[0,2].plot(wt, 'o-', label=layer.name)
    
    for layer in x_fnn.layers:
        w = layer.get_weights()[0]
        
        axs[1,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=alpha, density=density)
        
        wt = np.mean(np.abs(w), axis=1)
        axs[1,2].plot(wt, 'o-', label=layer.name)
        
    # w = t_u.get_weights()[::2]
    # axs[0,2].hist(np.ravel(w), bins=bins, label=t_u.name, alpha=alpha, density=density)
    #
    # w = t_v.get_weights()[::2]
    # axs[0,2].hist(np.ravel(w), bins=bins, label=t_v.name, alpha=alpha, density=density)
    #
    # w = t_w.get_weights()[::2]
    # axs[0,2].hist(np.ravel(w), bins=bins, label=t_w.name, alpha=alpha, density=density)
    
    ##X layers ###
    # axs[1,0].hist(np.ravel(x_emb.xc), bins=bins, label='x0', alpha=alpha, density=density)
    # axs[1,0].hist(np.ravel(x_emb.h), bins=bins, label='h_x', alpha=alpha, density=density)
    #
    # for layer in x_fnn.layers:
    #     w = layer.get_weights()[::2]
    #     axs[1,1].hist(np.ravel(w), bins=bins, label=layer.name, alpha=alpha, density=density)
    #
    # w = x_u.get_weights()[::2]
    # axs[1,2].hist(np.ravel(w), bins=bins, label=x_u.name, alpha=alpha, density=density)
    # 
    # w = x_v.get_weights()[::2]
    # axs[1,2].hist(np.ravel(w), bins=bins, label=x_v.name, alpha=alpha, density=density)
    #
    # w = x_w.get_weights()[::2]
    # axs[1,2].hist(np.ravel(w), bins=bins, label=x_w.name, alpha=alpha, density=density)
    
    for ax in axs.flatten():
        ax.legend()
        # ax.set_xlim(-1,1)
    
    # axs[0,0].set_xlim(-1.2,2.2)
    # axs[1,0].set_xlim(-1.2,2.2)
    
    plt.tight_layout()
    figfile = os.path.join(figpath, 'weight_%s_%s.png' %(postfix, log_file) )
    
    plt.savefig(figfile)
 
 
def plot_weight_hist_v10_respinn(filename, figpath, postfix='',
                        bins=32, log_file=None,
                        xmin=-1, xmax=1, 
                        alpha=1.0,
                        density=False,
                        residual_layer=False,
                        histtype="step" #'bar'
                        ):
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    nn = pinn.restore(filename, log_index=log_file)
    
    nn_model = nn.model
    
    kw_layer = nn_model.emb.layer0
    nn_layers = nn_model.laaf_layers
    tu_layers = nn_model.linear_layers
    
    xc = kw_layer.kernel
    h  = kw_layer.bias
    
    # vmin = np.min(w)
    # vmax = np.max(w)
    
    # bins0 = np.linspace(vmin, vmax, num=bins)
    
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    
    axs[0,0].hist(h.numpy().T, bins=bins, label=['Delay', 'Delay x1', 'Delay x2', 'Delay x3'], alpha=alpha, density=density, histtype=histtype)
    # axs[0,0].hist(np.ravel(xc[:,1]), bins=bins, label='Delays x1', alpha=alpha, density=density, histtype=histtype)
    # axs[0,0].hist(np.ravel(xc[:,2]), bins=bins, label='Delays x2', alpha=alpha, density=density, histtype=histtype)
    # axs[0,0].hist(np.ravel(xc[:,3]), bins=bins, label='Delays x3', alpha=alpha, density=density, histtype=histtype)
    
    
    axs[0,1].hist(xc.numpy().T , bins=bins, label=['Width x0', 'Width x1', 'Width x2', 'Width x3'], alpha=alpha, density=density, histtype=histtype)
    # axs[1,1].hist(np.ravel(h[:,1]) , bins=bins, label='Width x1', alpha=alpha, density=density, histtype=histtype)
    # axs[1,1].hist(np.ravel(h[:,2]) , bins=bins, label='Width x2', alpha=alpha, density=density, histtype=histtype)
    # axs[1,1].hist(np.ravel(h[:,3]) , bins=bins, label='Width x3', alpha=alpha, density=density, histtype=histtype)
    
    
    ws = []
    labels = []
    for layer0 in nn_layers:
        w0 = layer0.w
        ws.append(np.ravel(w0))
        labels.append(layer0.name)
        
    axs[1,0].hist(ws, bins=bins, label=labels, alpha=alpha, density=density, histtype=histtype)
    
    ws = []
    labels = []
    for layer0 in tu_layers:
        w0 = layer0.w
        ws.append(np.ravel(w0))
        labels.append(layer0.name)
        
    axs[1,1].hist(ws, bins=bins, label=labels, alpha=alpha, density=density, histtype=histtype)
    
    
    # for i in range(wu.shape[0]):
    #     w0 = wu[i]
    #     ws.append([w0])
    #     labels.append('Linear %s' %comp[i])
    #
    # axs[1,1].hist(ws, bins=bins, label=labels, alpha=alpha, density=density)
    
    for ax in axs.flatten():
        ax.legend()
        ax.set_xlim(xmin, xmax)
        ax.set_yscale("log")
      
    # axs[0,0].legend()
    # axs[0,0].set_xlim(xmin, xmax)
    # axs[0,1].legend()
    # axs[0,1].set_xlim(xmin, xmax)
    # axs[1,0].legend()
    # axs[1,0].set_xlim(xmin, xmax)
    # axs[1,1].legend()
    # axs[1,1].set_xlim(xmin, xmax)
    
    
    
    # axs[0,0].grid(True, linestyle='--')
    # axs[0,1].grid(True, linestyle='--')
    # axs[1,0].grid(True, linestyle='--')
    # axs[1,1].grid(True, linestyle='--')
    
    plt.tight_layout()
    figfile = os.path.join(figpath, 'weight_%s_%s.png' %(postfix, log_file) )
    
    plt.savefig(figfile)
       
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Germany", help='Data path')
    parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Argentina/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Norway/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Condor/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/JRO/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Piura/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/NewMexico/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Norway/", help='Data path')
    # parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Virtual/DNS_Simone2018/", help='Data path')
    
    parser.add_argument('-m', '--model', dest='model', default=None, help='neural network model')
    parser.add_argument('-s', '--subfolder', dest='subfolder', default="nnRESPINN_11.41", help='subfolder where the neural network model is stored')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='figures extension')
    parser.add_argument('-t', '--type', dest='type', default='full', help='plot type. Either "residuals" or "full" wind')
    parser.add_argument('-l', '--log-file', dest='log_file', default=0, help='select the i-th weights file from the log folder')
    parser.add_argument('-r', '--meteor-path', dest='mpath', default='/Users/radar/Data/IAP/SIMONe/Germany/Simone2018/', help='Data path')
    
    args = parser.parse_args()
    
    path  = args.dpath
    mpath = args.mpath
    model_name = args.model
    ext = args.ext
    type = args.type
    log_file = args.log_file
    
    xrange = None
    yrange = None
    
    
    path_PINN = os.path.join(path, "winds", args.subfolder)
    
    figpath = os.path.join(path_PINN, 'plot_weights')
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
            
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
        
        figpath_type = os.path.join(figpath, '%s_%s_std' %(type, log_file) )
        
        try:
            plot_weight_hist_v10_respinn(filename, figpath, postfix=model, log_file=log_file)
        except:
            pass
        