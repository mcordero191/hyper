'''
Created on 30 Aug 2022

@author: mcordero
'''
import os, time
import numpy as np

from atmospheric_models.ICON import ICONReader
from radar.specular_meteor_radars.SMR import SMRReader

if __name__ == '__main__':
    
    path_ICON_model     = '/Users/mcordero/Data/IAP/Models/UA ICON/nest3_20160816'
    path_meteor_data    = '/Users/mcordero/Data/IAP/SIMONe/Germany/Simone2018'
    rpath               = '/Users/mcordero/Data/IAP/SIMONe/Virtual/ICON_20160816'
    
    path_ICON_model     = '/Users/mcordero/Data/IAP/Models/UA ICON/nest3_20160815'
    path_meteor_data    = '/Users/mcordero/Data/IAP/SIMONe/Germany/Simone2018'
    rpath               = '/Users/mcordero/Data/IAP/SIMONe/Virtual/ICON_20160815'
    
    # if not os.path.exists(path):
    #     os.mkdir(path)
    #
    # radar_system = os.path.basename(os.path.realpath(path_meteor_data))
    # rpath = os.path.join(path, radar_system)
    
    if not os.path.exists(rpath):
        os.mkdir(rpath)
        
    #Read model data in LLA coordinates
    modelsys = ICONReader(path_ICON_model,
                           lat_center = 70,
                           lon_center = 0,
                           alt_center = 90,
                           alt_range = 40,
                           lon_range = 16,
                           lat_range = 10,
                           )
    
    #Read meteor data in LLA coordinates
    meteorsys = SMRReader(path_meteor_data)
    lat_center, lon_center, alt_center = meteorsys.get_spatial_center()

    subfolder = 'ICON_%+03d%+03d%+03d' %(modelsys.lon_center, modelsys.lat_center, modelsys.alt_center)
    
    rpath = os.path.join(rpath, subfolder)
    if not os.path.exists(rpath):
        os.mkdir(rpath)
        
    modelsys.set_meteor_spatial_center(lat_center, lon_center, alt_center)
    
    while True:
        info = meteorsys.read_next_file()
        
        if info != 1: break
        
        #Restart each new file (no enough modelsys files)
        ini_time = meteorsys.get_initial_time()
        modelsys.set_meteor_initial_time(ini_time)
        
        df = meteorsys.df
        
        for i in range(meteorsys.n_samples):
            
            time_mtr = df['times'][i]
            lat_mtr = df['lats'][i]
            lon_mtr = df['lons'][i]
            alt_mtr = df['heights'][i]
            
            tx_lla, rx_lla = meteorsys.get_rx_tx_lla(index=i)
            
            #Reads the modelsys data
            sample = modelsys.get_meteor_sample(time_mtr,
                                                 [lat_mtr, lon_mtr, alt_mtr],
                                                 tx_lla,
                                                 rx_lla,
                                                 )
            
            if sample is None:
                df.loc[i,'dops'] = np.nan
                print('x', end='')
                continue
            
            print('.', end='')
            
            df.loc[i,'meteor_times']   = time_mtr
            
            df.loc[i,'times']          = sample['epoch']
            df.loc[i,'u']              = sample['u']
            df.loc[i,'v']              = sample['v']
            df.loc[i,'w']              = sample['w']
            
            df.loc[i,'T']              = sample['T']
            df.loc[i,'P']              = sample['P']
            df.loc[i,'rho']            = sample['rho']
            df.loc[i,'tke']            = sample['tke']
            
            df.loc[i,'lats']           = sample['meteor_lla'][0]
            df.loc[i,'lons']           = sample['meteor_lla'][1]
            df.loc[i,'heights']        = sample['meteor_lla'][2]
            df.loc[i,'braggs_x']       = sample['bragg'][0]
            df.loc[i,'braggs_y']       = sample['bragg'][1]
            df.loc[i,'braggs_z']       = sample['bragg'][2]
            df.loc[i,'dops']           = sample['doppler']
            
        meteorsys.save_block(rpath, df=df)