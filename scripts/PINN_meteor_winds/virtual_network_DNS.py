'''
Created on 30 Aug 2022

@author: mcordero
'''
import os, time
import numpy as np

from atmospheric_models.DNS import DNSReader
from radar.smr.smr_file import SMRReader

if __name__ == '__main__':
    
    path_meteor_data    = '/Users/radar/Data/IAP/SIMONe/Norway/Vortex'
    path_model          = '/Users/radar/Data/IAP/Models/DNS/NonStratified'
    rpath               = '/Users/radar/Data/IAP/SIMONe/Virtual/'
    
    if not os.path.exists(rpath):
        os.mkdir(rpath)
        
    #Read model data in LLA coordinates
    modelsys = DNSReader(path_model)
    
    #Read meteor data in LLA coordinates
    meteorsys = SMRReader(path_meteor_data)
    
    lat_center, lon_center, alt_center = meteorsys.get_spatial_center()
    
    modelsys.set_meteor_spatial_center(lat_center, lon_center, alt_center)
    
    subfolder = 'DNSx10_%+03d%+03d%+03d' %(lon_center, lat_center, alt_center)
    
    exp = os.path.basename(os.path.abspath(path_meteor_data))
    
    rpath = os.path.join(rpath, 'DNS_' +exp)
    if not os.path.exists(rpath):
        os.mkdir(rpath)
        
        
    rpath = os.path.join(rpath, subfolder)
    if not os.path.exists(rpath):
        os.mkdir(rpath)
    
    while True:
        info = meteorsys.read_next_file()
        
        if info != 1: break
        
        #Restart each new file (no enough modelsys files)
        ini_time = meteorsys.get_initial_time()
        modelsys.set_meteor_initial_time(ini_time)
        
        df = meteorsys.df
        
        for i in range(meteorsys.n_samples):
            
            SMR_like = 1
            
            time_mtr = df['times'][i]
            lat_mtr = df['lats'][i]
            lon_mtr = df['lons'][i]
            alt_mtr = df['heights'][i]
            
            tx_lla, rx_lla = meteorsys.get_rx_tx_lla(index=i)
            
            #Resample in time
            if time_mtr - ini_time > modelsys.times[-1]:
                time_mtr = ini_time + (time_mtr - ini_time)%modelsys.times[-1]
                    
            #Reads the modelsys data
            sample = modelsys.get_meteor_sample(time_mtr,
                                                 [lat_mtr, lon_mtr, alt_mtr],
                                                 tx_lla,
                                                 rx_lla,
                                                 )
            
            if sample is None:
                    
                #Resample in space
                dlat = np.random.normal()*1.5/2
                dlon = np.random.normal()*3.0/2
                dalt = np.random.normal()*10.0/2
                
                sample = modelsys.get_meteor_sample(time_mtr,
                                                 [lat_center+dlat, lon_center+dlon, alt_center+dalt],
                                                 tx_lla,
                                                 rx_lla,
                                                 )
                SMR_like = 0
                
                if sample is None:
                    df.loc[i,'dops'] = np.nan
                    print('x', end='')
                    continue
            
            print('.', end='')
            
            df.loc[i,'meteor_times']   = time_mtr
            
            df.loc[i,'times']          = sample['time']
            df.loc[i,'u']              = sample['u']
            df.loc[i,'v']              = sample['v']
            df.loc[i,'w']              = sample['w']
            
            df.loc[i,'T']              = sample['T']
            # df.loc[i,'P']              = sample['P']
            # df.loc[i,'rho']            = sample['rho']
            # df.loc[i,'tke']            = sample['tke']
            
            df.loc[i,'lats']           = sample['meteor_lla'][0]
            df.loc[i,'lons']           = sample['meteor_lla'][1]
            df.loc[i,'heights']        = sample['meteor_lla'][2]
            df.loc[i,'braggs_x']       = sample['bragg'][0]
            df.loc[i,'braggs_y']       = sample['bragg'][1]
            df.loc[i,'braggs_z']       = sample['bragg'][2]
            df.loc[i,'dops']           = sample['doppler']
            
            df.loc[i,'SMR_like']       = SMR_like
            
        
        df.attrs['lon_ref'] = modelsys.lon_ref
        df.attrs['lat_ref'] = modelsys.lat_ref
        df.attrs['alt_ref'] = modelsys.alt_ref
        
        meteorsys.save_block(rpath, df=df)