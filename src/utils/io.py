'''
Created on 21 Feb 2024

@author: mcordero
'''
import h5py
import datetime

import pandas as pd

def save_h5file(alts, lats, lons,
                ue, ve, we,
                u=None, v=None, w=None,
                filename='',
                times=None,
                labels=["u", "v", "w"],
                ):
    
    with h5py.File(filename, 'w') as fp:
    
        g = fp.create_group('metadata')
        
        g['lats'] = lats
        g['lons'] = lons
        g['alts'] = alts
        
        if times is not None:
            g['times'] = times
        
        if u is not None:
            g = fp.create_group('truth')
            
            g['u'] = u
            g['v'] = v
            g['w'] = w
    
        g = fp.create_group('data')
        
        g[labels[0]] = ue
        g[labels[1]] = ve
        g[labels[2]] = we
        
def read_vortex_files(filenames=None):
    
    if filenames is None:
        
        upleg = {
                 "filename" : '/Users/radar/Data/IAP/SIMONe/Norway/VorTex/UPLEG34_sigma0_1Bin_size_1km.txt',
                 "time" : datetime.datetime(2023,3,23,21,5,0, tzinfo=datetime.timezone.utc).timestamp(),
                 "lon" : 15.75,
                 "lat" : 69.45,
                 }
        
        downleg={
                 "filename" : '/Users/radar/Data/IAP/SIMONe/Norway/VorTex/DNLEGAVG_AVG.txt',
                 "time" : datetime.datetime(2023,3,23,21,10,0, tzinfo=datetime.timezone.utc).timestamp(),
                 "lon" : 14.25,
                 "lat" : 70.35,
                 }
        
        
    
    
    df = None
    for d in [upleg, downleg]:
        
        dfi = pd.read_csv(d["filename"],
                         # sep=' ',
                         header=0,
                         skiprows=1,
                         sep='\s+',
                         names=['heights', 'u', 'v','ue', 've'],
                         )
        
        dfi["w"]    = 1e-6
        
        dfi["times"]= d["time"]
        
        dfi["lons"] = d["lon"]
        dfi["lats"] = d["lat"]
        
    
        dfi["braggs_x"] = 0.0
        dfi["braggs_y"] = 0.0
        dfi["braggs_z"] = 0.0
        
        dfi["dops"] = 1e-6
        dfi["dop_errs"] = 1.0
        
        
        if df is None:
            df = dfi
        else:
            df = pd.concat( [df, dfi], ignore_index=True )
        
    df = df[df["heights"]<100.]    
        
    return(df)
        
def read_vortex(filename):
    
    df = pd.read_csv(filename,
                     # sep=' ',
                     header=0,
                     skiprows=1,
                     delim_whitespace=True,
                     names=['alt', 'u', 'v','ue', 've'],
                     )
            
    return(df)