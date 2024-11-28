import os
import glob
import subprocess

import numpy as np


def find_png_files(path, suffix='80.0'):
    
    fxlist = glob.glob1(path,'*%s.png' %suffix)
    fxlist = sorted(fxlist)
    
    return(fxlist)

def make_gif(file_pattern, output_file):
    
    cmd = "/opt/local/bin/convert -delay 60 -loop 0 %s %s" %(file_pattern, output_file)
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    os.waitpid(p.pid, 0)

def animation_hcuts(figpath, rpath):
    
    altitudes = np.arange(84,100, 6)
    
    for h in altitudes:
        file_pattern = os.path.join(figpath, 'wind_field_*_*_%2.1f.png' %h)
        output_file  = os.path.join(rpath, 'h%2.0f.gif' %h)
        
        make_gif(file_pattern, output_file)

def animation_wind_vector(figpath, rpath, suffix, file_pattern='wind_vec_*.png'):
    
    file_pattern = os.path.join(figpath, file_pattern)
    output_file  = os.path.join(rpath, 'wind_%s.gif' %suffix)
    
    make_gif(file_pattern, output_file)
             
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to emake video animation from a list of png files')
    
    parser.add_argument('-d', '--dpath', dest='dpath', default="/Users/mcordero/Data/IAP/SIMONe/Norway/", help='Data path')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Data path')
    parser.add_argument('-s', '--subfolder', dest='subfolder', default="fPINN_fc2.1_35k", help='subfolder where the neural network model is stored')
    
    args = parser.parse_args()
    
    dpath = args.dpath
    rpath = args.rpath
    subfolder = args.subfolder
        
    figpath = os.path.join(dpath, subfolder, 'plot_residuals_png_None')
    
    if rpath is None:
        rpath = figpath
        
    day_folders = sorted( os.listdir(figpath) )
    
    for folder in day_folders:
        
        fpath = os.path.join(figpath, folder)
        
        if not os.path.isdir(fpath):
            continue
        
        suffix = folder[-8:]
        
        animation_wind_vector(fpath, rpath, suffix)
        