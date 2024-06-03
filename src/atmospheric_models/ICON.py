'''
Created on 2 Sep 2022

@author: mcordero
'''
import glob
import datetime
import numpy as np

import scipy.constants as const
from netCDF4 import Dataset

from georeference.geo_coordinates import lla2ecef, lla2enu

dt1970 = datetime.datetime(1970,1,1,0,0,0)

def datefmt2epoch(date):
    '''
    Convert date format to epoch in seconds from 1970
    
    Input:
        date        :    date and time information 20160816.f
                        where 2016/08/16 and f indicates date and time, respectively
    Output:
        epoch       :    
    '''
    junk = int(date)
    seconds = (date - junk)*24*60*60 +0.01
    
    yy = junk//10000
    junk = junk % 10000
    
    mm = junk//100
    junk = junk % 100
    
    dd = junk
    
    dt = datetime.datetime(yy,mm,dd,0,0,0) + datetime.timedelta(seconds=seconds)
    
    epoch = (dt - dt1970).total_seconds()
    
    return(np.round(epoch))
 
def derivative(u):
    '''
    Calculate the derivative of "u" in the x, y and z direction
    
    Returns:
        u_x, u_y, u_z with the same dimension as u.
        Invalid derivative points are nans.
    '''
    
    u_x = np.zeros_like(u) + np.nan
    u_y = np.zeros_like(u) + np.nan
    u_z = np.zeros_like(u) + np.nan
    
    u_x[1:, :, :] = u[1:,:,:] - u[:-1,:,:]
    u_y[ :,1:, :] = u[:,1:,:] - u[:,:-1,:]
    u_z[ :, :,1:] = u[:,:,1:] - u[:,:,:-1]
    
    return(u_x, u_y, u_z)

def array_selection_3D(u, x_indexes, y_indexes, z_indexes):
    
    return( u[x_indexes][:,y_indexes][:,:,z_indexes] )
    
class ICONReader(object):
    
    time_grid = None
    lat_grid = None
    long_grid = None
    alt_grid = None
    
    ini_time = 0
    ini_time_meteor = 0
    
    def __init__(self, path,
                 alt_range = 40,
                 lon_range = 10,
                 lat_range = 10,
                 lat_center = None,
                 lon_center = None,
                 alt_center = 90,
                 ):
        
        files = self.__find_files(path)
        
        if len(files) < 1:
            raise ValueError('No ICON files found in %s' %path)
            
        time0, lats, lons, alts = self.read_metadata(files[0])
        time1, _, _, _ = self.read_metadata(files[1])
        
        if lat_center is None: lat_center = np.median(lats)
        if lon_center is None: lon_center = np.median(lons)
        if alt_center is None: alt_center = np.median(alts)
        
        print('ICON middle point (LLA): ', lon_center, lat_center, alt_center)
        
        min_alt = alt_center - alt_range/2.0
        max_alt = alt_center + alt_range/2.0
        
        min_lon = lon_center - lon_range/2.0
        max_lon = lon_center + lon_range/2.0
        
        min_lat = lat_center - lat_range/2.0
        max_lat = lat_center + lat_range/2.0
        
        alt_mask = np.where( (alts >= min_alt) & (alts <= max_alt) )[0]
        lon_mask = np.where( (lons >= min_lon) & (lons <= max_lon) )[0]
        lat_mask = np.where( (lats >= min_lat) & (lats <= max_lat) )[0]
        
        alts = alts[alt_mask]
        lons = lons[lon_mask]
        lats = lats[lat_mask]
        
        time_res = time1 - time0
        lat_res = np.abs(lats[1] - lats[0])
        lon_res = np.abs(lons[1] - lons[0])
        alt_res = np.abs(alts[1] - alts[0])
        
        self.ini_time = time0
        self.ini_datetime = datetime.datetime.utcfromtimestamp(time0)
        
        self.lat_grid = lats
        self.lon_grid = lons
        self.alt_grid = alts
        
        self.alt_mask = alt_mask
        self.lon_mask = lon_mask
        self.lat_mask = lat_mask
        
        self.time_resolution = time_res
        self.lat_resolution = lat_res
        self.lon_resolution = lon_res
        self.alt_resolution = alt_res
        
        self.lat_center = lat_center
        self.lon_center = lon_center
        self.alt_center = alt_center
        
        self.files = files
        self.ifile = -1
        self.epoch = None
        self.u = None
        self.v = None
        self.w = None
        
        self.data = None
        
        self.set_radar_frequency()
    
    def __find_files(self, path):
        
        files = glob.glob("%s/ICON_NEST3_*.nc" %path )
        files = sorted(files)
        
        return(files)
    
    def __read_block(self, time_meteor):
        
        ifile = int( (time_meteor-self.ini_time_meteor)/self.time_resolution )
        
        if ifile >= len(self.files):
            print('O', end='')
            return 0
        
        filename = self.files[ifile]
        
        d = self.read_data(filename)
        
        self.epoch = d['epoch'] #+ self.ini_time_meteor - self.ini_time
        # self.u = d['u']
        # self.v = d['v']
        # self.w = d['w']
        
        self.data = d
        
        self.ifile = ifile
        
        print('\nICON file #%d [t=%2.1f]' %(ifile, self.epoch) )
        return(1)
        
    def __is_in_buffer(self, time_meteor):
        
        ifile = int(time_meteor-self.ini_time_meteor)//self.time_resolution
        
        if ifile == self.ifile:
            return True
    
        return False
    
    def __get_lat_index(self, lat_meteor):
        
        delta = np.abs(self.lat_grid - lat_meteor - self.lat_diff)
        index = np.argmin( delta )
        
        if delta[index] > self.lat_resolution/2:
            # print('\nLatitude %f not found' %lat_meteor)
            return(None)
            
        return(index)
    
    def __get_lon_index(self, lon_meteor):
        
        delta = np.abs(self.lon_grid - lon_meteor - self.lon_diff)
        index = np.argmin( delta )
        
        if delta[index] > self.lon_resolution/2:
            # print('\nLongitude %f not found' %lon_meteor)
            return(None)
        
        return(index)
    
    def __get_alt_index(self, alt_meteor):
        
        delta = np.abs(self.alt_grid - alt_meteor - self.alt_diff)
        index = np.argmin( delta )
        
        if delta[index] > self.alt_resolution/2:
            # print('\nAltitude %f not found' %alt_meteor)
            return(None)
        
        return(index)
    
    def __lla_in_new_coordinate_system(self, lla, match_grid=True):
        
        if not match_grid:
            lat = lla[0] + self.lat_diff
            lon = lla[1] + self.lon_diff
            alt = lla[2] + self.alt_diff
            
            return( [lat, lon, alt] )
        else:
            lat_index = self.__get_lat_index(lla[0])
            lon_index = self.__get_lon_index(lla[1])
            alt_index = self.__get_alt_index(lla[2])
            
            if (lat_index is None) or (lon_index is None) or (alt_index is None):
                return(None, None)
            
            lat = self.lat_grid[lat_index]
            lon = self.lon_grid[lon_index]
            alt = self.alt_grid[alt_index]
        
            return( [lat, lon, alt], [lat_index, lon_index, alt_index] )
        
    def __get_bragg_vector(self, meteor_lla, rx_lla, tx_lla):
        
        # R_met  : ecef coordinate of meteor
        # R_tx : ecef coordinate of transmitter
        # R_rx : ecef coordinate of receiver
        # all the above should be in (x,y,z) form of numpy array
        # lambd: wavelength of radar (in meters)
            
        r_met = lla2ecef(*meteor_lla)
        r_rx = lla2ecef(*rx_lla)
        r_tx = lla2ecef(*tx_lla)
        
        r_met = np.array(r_met)
        r_rx = np.array(r_rx)
        r_tx = np.array(r_tx)
        
        ki = (np.subtract(r_met,r_tx)/ np.linalg.norm(r_met-r_tx)) * (2*np.pi/self.radar_wavelength) # check the order
        ks = (np.subtract(r_rx,r_met)/ np.linalg.norm(r_rx-r_met)) * (2*np.pi/self.radar_wavelength) # check the order
        
        kb = np.subtract(ks, ki)
    
        return(kb)
        
    def read_metadata(self, filename):
        
        fp = Dataset(filename,'r')

        t = fp.variables['time'][0].data
        
        alt = fp.variables['z_ifc'][1:,0,0].data*1e-3 #To km
        lat  = fp.variables['lat'][:].data
        lon  = fp.variables['lon'][:].data
        
        fp.close()
        
        #Date information 20160816.f
        epoch = datefmt2epoch(t)
        
        return(epoch, lat, lon, alt)
    
    def read_data(self, filename, read_all_parms=False):
        '''
        Return u, v, and w with dimension [altitudes, latitudes, longitudes]
        '''
        fp = Dataset(filename,'r')
    
        t = fp.variables['time'][0].data
        
        u = fp.variables['u'][0,:,:,:].data
        v = fp.variables['v'][0,:,:,:].data
        w = fp.variables['w'][0,1:,:,:].data #Discard first row (no data)
        
        if read_all_parms:
            T = fp.variables['temp'][0,:,:,:].data  #Temperature (K)
            P = fp.variables['pres'][0,:,:,:].data  #Pressure (Pa <> N/m^2 <> kg.m/s^2/m^2 <> kg/m/s^2)
            rho = fp.variables['rho'][0,:,:,:].data   #Neutral density (kg/m^3)
            tke = fp.variables['tke'][0,1:,:,:].data  #Turbulent kinetic energy
        
        fp.close()
        
        #Date information 20160816.f
        epoch = datefmt2epoch(t)
        
        #Apply masks
        u = array_selection_3D(u, self.alt_mask, self.lat_mask, self.lon_mask)
        v = array_selection_3D(v, self.alt_mask, self.lat_mask, self.lon_mask)
        w = array_selection_3D(w, self.alt_mask, self.lat_mask, self.lon_mask)
        
        if read_all_parms:
            T = array_selection_3D(T, self.alt_mask, self.lat_mask, self.lon_mask)
            P = array_selection_3D(P, self.alt_mask, self.lat_mask, self.lon_mask)
            rho = array_selection_3D(rho, self.alt_mask, self.lat_mask, self.lon_mask)
            tke = array_selection_3D(tke, self.alt_mask, self.lat_mask, self.lon_mask)
        
        
        d = {}
        d['epoch'] = epoch
        d['u'] = u
        d['v'] = v
        d['w'] = w
        
        if read_all_parms:
            d['T'] = T
            d['P'] = P
            d['rho'] = rho
            d['tke'] = tke
        else:
            d['T'] = None
            d['P'] = None
            d['rho'] = None
            d['tke'] = None
        
        return(d)
    
    def read_next_block(self, derivatives=False, skip_block=False):
        '''
        Return a dictionary containing: epoch, u, v, w
        
        u, v, w are 3D arrays with dimensions (altitude, latitude, longitude)
        '''
        ifile = self.ifile + 1
        
        self.ifile = ifile
        
        if ifile >= len(self.files):
            return None
        
        if skip_block:
            return {}
        
        filename = self.files[ifile]
        
        d = self.read_data(filename)
        
        self.epoch = d['epoch'] #+ self.ini_time_meteor - self.ini_time
        self.u = d['u']
        self.v = d['v']
        self.w = d['w']
        
        self.P = d['P']
        self.T = d['T']
        
        
        
        print('\nFile #%d [t=%2.1f]' %(ifile, self.epoch) )

        
        if derivatives:
            #Since the dimension of u, v and w is [alt, lat, lon]
            # the derivatives _x, _y, _z refer to alt, lat, and lon respectively.
            u_x, u_y, u_z = derivative(d['u'])
            v_x, v_y, v_z = derivative(d['v'])
            w_x, w_y, w_z = derivative(d['w'])
            
            d['u_x'] = u_x
            d['u_y'] = u_y
            d['u_z'] = u_z
            
            d['v_x'] = v_x
            d['v_y'] = v_y
            d['v_z'] = v_z
            
            d['w_x'] = w_x
            d['w_y'] = w_y
            d['w_z'] = w_z
        
        return(d)
    
    def set_radar_frequency(self, radar_frequency=30e6):
        
        radar_wavelength = const.c/radar_frequency
        
        self.radar_wavelength = radar_wavelength
    
    def set_meteor_spatial_center(self, lat_center_meteor, lon_center_meteor, alt_center_meteor):
        
        lat_diff = self.lat_center - lat_center_meteor
        lon_diff = self.lon_center - lon_center_meteor
        alt_diff = self.alt_center - alt_center_meteor
        
        self.lat_diff = lat_diff
        self.lon_diff = lon_diff
        self.alt_diff = 0
        
        self.lat_center_meteor = lat_center_meteor
        self.lon_center_meteor = lon_center_meteor
        self.alt_center_meteor = alt_center_meteor

    def set_meteor_initial_time(self, ini_time_meteor):
        
        self.ini_time_meteor = ini_time_meteor
          
    def get_meteor_sample(self,
                       time_meteor,
                       meteor_lla,
                       tx_lla=None,
                       rx_lla=None,
                       ):
        
        if rx_lla is None: rx_lla = (self.lat_center_meteor, self.lon_center_meteor, 0)
        if tx_lla is None: tx_lla = (self.lat_center_meteor, self.lon_center_meteor, 0)
        
        if not self.__is_in_buffer(time_meteor):
            info = self.__read_block(time_meteor)
            
            if info != 1: return None
        
        rx_lla_new = self.__lla_in_new_coordinate_system(rx_lla, match_grid=False)
        tx_lla_new = self.__lla_in_new_coordinate_system(tx_lla, match_grid=False)
        meteor_lla_new, meteor_lla_new_index = self.__lla_in_new_coordinate_system(meteor_lla)
        
        if (meteor_lla_new is None):
            return(None)
        
        bragg = self.__get_bragg_vector( meteor_lla_new, rx_lla_new, tx_lla_new)
        
        lat_index = meteor_lla_new_index[0]
        lon_index = meteor_lla_new_index[1]
        alt_index = meteor_lla_new_index[2]
        
        u = self.data['u'][alt_index, lat_index, lon_index]
        v = self.data['v'][alt_index, lat_index, lon_index]
        w = self.data['w'][alt_index, lat_index, lon_index]
        
        T = self.data['T'][alt_index, lat_index, lon_index]
        P = self.data['P'][alt_index, lat_index, lon_index]
        rho = self.data['rho'][alt_index, lat_index, lon_index]
        tke = self.data['tke'][alt_index, lat_index, lon_index]
        
        doppler = (-1.0/(2*np.pi))*np.dot( [u,v,w], bragg )
        
        d = {}
        d['epoch'] = self.epoch
        d['u'] = u
        d['v'] = v
        d['w'] = w
        
        d['T'] = T      #Temperature (K)
        d['P'] = P      #Pressure (N/m^2)
        d['rho'] = rho  #Density (kg/m^3)
        d['tke'] = tke  #Turbulent kinetic energy (m2/s2)
        
        d['meteor_lla'] = meteor_lla_new
        d['rx_lla'] = rx_lla_new
        d['tx_lla'] = tx_lla_new
        d['bragg'] = bragg
        d['doppler'] = doppler
        
        return(d)
        
        
if __name__ == '__main__':
    
    path = '/Users/mcordero/Data/IAP/Models/UA ICON/nest3_20160815'
    
    model_obj = ICONReader(path)
    
    model_obj.set_meteor_spatial_center(40, 30, 0)
    
    d = model_obj.get_meteor_sample(10, [35, 35, 90e3] )
    print(d)
    
    d = model_obj.get_meteor_sample(10, [45, 35, 90e3] )
    print(d)
    