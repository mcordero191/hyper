'''
Created on 2 Sep 2022

@author: mcordero
'''
import os, glob
import datetime
import numpy as np

import scipy.constants as const

from georeference.geo_coordinates import lla2ecef, lla2xyh, xyh2lla

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
    seconds = (date - junk)*24*60*60 + 0.01
    
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

def read_data(filename, dimX=1024, dimY=1024, dimZ=1024,
              dimZ_out=100, decX=1, decY=1, decZ=1):
    
    if not dimY: dimY = dimX
    if not dimZ: dimZ = dimX

    print("reading %s ..." %filename)
    
    with open(filename,'r') as f:
        data = np.fromfile(f, dtype=np.float32)
        
    data = data.reshape((dimZ,dimY,dimX))
    
    return(data[:dimZ_out:decZ,::decY,::decX])

def write_data(filename, data):
    '''
    Inputs:
        data    :    3d numpy array [nx, ny, nx]
    '''

    print("saving %s ..." %filename)
    
    data.tofile(filename)
    
    return

class DNSReader(object):
    
    time_grid = None
    lat_grid = None
    lon_grid = None
    alt_grid = None
    
    ini_time = 0
    ini_time_meteor = 0
    
    def __init__(self,
                 path,
                 delta_s=0.292969,       #kilometers
                 delta_t=79.65,         #seconds
                 u_scale=8.8914,
                 dimS=1024,
                 dimZ=None,
                 dimZ_out=100,
                 decS=1,
                 decZ=1,
                 ):
        
        files = self.__find_files(path)
        
        dimT = len(files)
        
        times, x, y, z = self.get_data_grid(delta_t, delta_s, dimT, dimS, dimZ=dimZ_out)
        
        x      = x[::decS]
        y      = y[::decS]
        z      = z[::decZ]

        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        
        self.X = X
        self.Y = Y
        self.Z = Z
        
        self.files = files
        self.ifile = -1
        
        self.time = None
        
        self.u = None
        self.v = None
        self.w = None
        
        self.times  = times
        self.x      = x
        self.y      = y
        self.z      = z
        
        self.data = None
        
        self.delta_t = delta_t
        self.delta_s = delta_s
        
        self.dimT = dimT
        self.dimS = dimS
        
        if dimZ is not None: self.dimZ = dimZ
        else: self.dimZ = dimS

        self.dimZ_out = dimZ_out
        
        self.decS = decS
        self.decZ = decZ
        
        self.u_scale = u_scale
        
        self.set_radar_frequency()
        
    def __find_files(self, path):
        
        files = glob.glob("%s/vx.0[0-1]*.out" %path )
        files = sorted(files)
        
        if len(files) < 1:
            raise ValueError("No files in %s" %path)
        return(files)
    
    def get_data_grid(self,
                      delta_t, delta_s,
                      dimT, dimS, dimZ):
        
        #Date information 20160816.f
        
        times = np.arange(dimT)*delta_t
        z = (np.arange(dimZ) - dimZ//2)*delta_s
        y = (np.arange(dimS) - dimS//2)*delta_s
        x = (np.arange(dimS) - dimS//2)*delta_s
        
        return(times, x, y, z)
    
    def __read_block(self, index):
        
        if index >= len(self.files):
            print('O', end='')
            return 0
        
        d = self.read_data(index)
        
        self.ifile = index
        self.time = d['time']
        self.data = d
        
        print('\nDNS file #%d, %s [t=%2.1f]' %(index, self.files[index], self.time) )
        return(1)
        
    def __is_in_buffer(self, ifile):
        
        # ifile = int(time_meteor-self.ini_time)//self.delta_t
        
        if ifile == self.ifile:
            return True
    
        return False
    
    def __grid_indices(self, lat, lon, alt):
        
        
        x, y, h = lla2xyh(lat, lon, alt,
                          self.lat_ref, self.lon_ref, self.alt_ref,
                          units='km')
        
        ix = int( np.round( (x-self.X[0,0,0])/self.delta_s ) )
        iy = int ( np.round( (y-self.Y[0,0,0])/self.delta_s ) )
        iz = int( np.round( (h-self.alt_ref-self.Z[0,0,0])/self.delta_s ) )

        ind = (iz, iy, ix)

        if ix < 0 or iy < 0 or iz < 0: return(None, None, None)

        if ix >= self.dimS or iy >= self.dimS or iz >= self.dimZ: return(None, None, None)

        return(ind)
        
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
    
    def read_data(self, index):
        '''
        Return u, v, and w with dimension [altitudes, latitudes, longitudes]
        '''
        
        # file_vx = 'vx.%04d.out' %ifile
        # file_vy = 'vy.%04d.out' %ifile
        # file_vz = 'vz.%04d.out' %ifile
        # file_th = 'th.%04d.out' %ifile
        
        filename_vx = self.files[index]
        filename_vy = filename_vx.replace('vx.', 'vy.')
        filename_vz = filename_vx.replace('vx.', 'vz.')
        filename_th = filename_vx.replace('vx.', 'th.')
        
        u  = read_data(filename_vx, self.dimS, self.dimS, self.dimZ, dimZ_out=self.dimZ_out, decX=self.decS, decY=self.decS, decZ=self.decZ)
        v  = read_data(filename_vy, self.dimS, self.dimS, self.dimZ, dimZ_out=self.dimZ_out, decX=self.decS, decY=self.decS, decZ=self.decZ)
        w  = read_data(filename_vz, self.dimS, self.dimS, self.dimZ, dimZ_out=self.dimZ_out, decX=self.decS, decY=self.decS, decZ=self.decZ)
        # th = read_data(filename_th, self.dimS, self.dimS, self.dimZ, dimZ_out=self.dimZ_out, decX=self.decS, decY=self.decS, decZ=self.decZ)
        
        t = self.times[index] + self.ini_time_meteor
        
        #Apply masks
        # u = array_selection_3D(u, self.alt_mask, self.lat_mask, self.lon_mask)
        # v = array_selection_3D(v, self.alt_mask, self.lat_mask, self.lon_mask)
        # w = array_selection_3D(w, self.alt_mask, self.lat_mask, self.lon_mask)
        #
        # th = array_selection_3D(th, self.alt_mask, self.lat_mask, self.lon_mask)
        
        d = {}
        d['time'] = t
        d['u'] = self.u_scale*u
        d['v'] = self.u_scale*v
        d['w'] = self.u_scale*w
        # d['T'] = th
        
        return(d)
    
    def read_next_block(self, derivatives=False, skip_block=False):
        '''
        Return a dictionary containing: time, u, v, w
        
        u, v, w are 3D arrays with dimensions (altitude, latitude, longitude)
        '''
        ifile = self.ifile + 1
        self.ifile = ifile
        
        if ifile >= len(self.files):
            return None
        
        if skip_block:
            d = {}
            return d
        
        d = self.read_data(ifile)
        
        self.time = d['time']
        
        self.u = d['u']
        self.v = d['v']
        self.w = d['w']
        # self.T = d['T']
        
        print('\nFile #%d [t=%2.1f, %s]' %(ifile, self.time, datetime.datetime.utcfromtimestamp(self.time)) )
        
        if derivatives:
            #Since the dimension of u, v and w is [alt, lat, lon]
            # the derivatives _x, _y, _z refer to alt, lat, and lon respectively.
            u_z, u_y, u_x = derivative(d['u'])
            v_z, v_y, v_x = derivative(d['v'])
            w_z, w_y, w_x = derivative(d['w'])
            
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
    
    def write_data(self, u, v, w, path='./'):
        
        u0 = u/self.u_scale
        v0 = v/self.u_scale
        w0 = w/self.u_scale
        
        index = self.get_file_index()
            
        filename = os.path.join(path, 'vx.%04d.out' %index)
        write_data(filename, u0)
        
        filename = os.path.join(path, 'vy.%04d.out' %index)
        write_data(filename, v0)
        
        filename = os.path.join(path, 'vz.%04d.out' %index)
        write_data(filename, w0)
            
            
    def get_file_index(self):
        
        filename = os.path.split(self.files[self.ifile])[1]
        
        index = int(filename[3:7]) #.replace('vx.', '')
        
        return(index)
        
    def set_radar_frequency(self, radar_frequency=30e6):
        
        radar_wavelength = const.c/radar_frequency
        
        self.radar_wavelength = radar_wavelength
    
    def set_meteor_spatial_center(self, lon_ref, lat_ref, alt_ref):
        
        #Convert enu to LLA
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref
        self.alt_ref = alt_ref
        
        LAT, LON, ALT = xyh2lla(self.X, self.Y, self.Z,
                                lat_ref=lat_ref, lon_ref=lon_ref, alt_ref=alt_ref)
        self.lat_grid = LAT
        self.lon_grid = LON
        self.alt_grid = ALT
        
        self.__meteor_grid = True
        
    def set_meteor_initial_time(self, ini_time):
        
        self.ini_time_meteor = ini_time
        
    def get_meteor_sample(self, meteor_time, meteor_lla_loc,
                           tx_lla_loc=None,
                           rx_lla_loc=None,
                           ):
        
        if not self.__meteor_grid:
            raise ValueError('Meteor grid has not been defined ...')
        
        
        ifile = int( (meteor_time - self.ini_time_meteor)//self.delta_t )
        
        if not self.__is_in_buffer(ifile):
            info = self.__read_block(ifile)
            
            if info != 1: return None
        
        iz, iy, ix = self.__grid_indices(*meteor_lla_loc)
        
        if (ix is None):
            return(None)
        
        new_meteor_lla_loc = self.lat_grid[iz, iy, ix], self.lon_grid[iz, iy, ix], self.alt_grid[iz, iy, ix]
        
        bragg = self.__get_bragg_vector( new_meteor_lla_loc, rx_lla_loc, tx_lla_loc)
        
        u = self.data['u'][iz, iy, ix]
        v = self.data['v'][iz, iy, ix]
        w = self.data['w'][iz, iy, ix]
        
        T = self.data['T'][iz, iy, ix]
        # P = self.data['P'][iz, iy, ix]
        # rho = self.data['rho'][iz, iy, ix]
        # tke = self.data['tke'][iz, iy, ix]
        
        doppler = (-1.0/(2*np.pi))*np.dot( [u,v,w], bragg )
        
        d = {}
        d['time'] = self.time
        d['u'] = u
        d['v'] = v
        d['w'] = w
        
        d['T'] = T      #Temperature (K)
        # d['tke'] = tke  #Turbulent kinetic energy (m2/s2)
        
        d['meteor_lla'] = new_meteor_lla_loc
        d['rx_lla'] = rx_lla_loc
        d['tx_lla'] = tx_lla_loc
        d['bragg'] = bragg
        d['doppler'] = doppler
        
        return(d)
        
if __name__ == '__main__':
    
    path = '/Users/mcordero/Data/IAP/Models/UA ICON/nest3_20160815'
    
    model_obj = DNSReader(path)
    
    model_obj.set_meteor_spatial_center(40, 30, 0)
    
    d = model_obj.get_meteor_sample(10, [35, 35, 90e3] )
    print(d)
    
    d = model_obj.get_meteor_sample(10, [45, 35, 90e3] )
    print(d)
    
