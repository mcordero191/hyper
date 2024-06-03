'''
Created on 2 Sep 2022

@author: mcordero
'''
import os, glob
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from georeference.geo_coordinates import lla2xyh, lla2enu
from utils.clustering import hierarchical_cluster

stations = {}
stations['collm']       = [51.31, 13.0, 10e-3]
stations['bornim']      = [52.44, 13.02, 10e-3]
stations['bor']         = [52.44, 13.02, 10e-3]
stations['ntz']         = [53.33, 13.07, 10e-3]
stations['jruh']        = [54.63, 13.37, 10e-3]
stations['kborn']       = [54.10, 11.8, 10e-3]
stations['breege']      = [54.62, 13.37, 10e-3]
stations['guderup']     = [55.00, 9.86, 10e-3]
stations['mechelsdorf'] = [54.12, 11.67, 10e-3]
stations['salzwedel']   = [52.75, 10.90, 10e-3]
stations['andenes']     = [69.27, 16.04, 10e-3]
stations['tromso']      = [69.58, 19.22, 10e-3]
stations['alta']        = [69.96, 23.29, 10e-3]
stations['straumen']    = [67.40, 15.6, 10e-3]

def get_xyz_bounds(x, y, z, n=4):
    
    xsigma = np.std(x)
    ysigma = np.std(y)
    zsigma = np.std(z)
    
    return( np.round( n*xsigma, 1), np.round( n*ysigma, 1), np.round( n*zsigma,1) )

def time_filter(df, tini=0, dt=24):
    
    # Slice to reconstruct
    #tbase = pd.to_datetime(dfraw['t'][0].date())
    de = df['dop_errs']
    
    std = np.sqrt(np.var(de))
    cond1 = (de > 3*std)
    
    dxy = np.sqrt( df['dcosx']**2 + df['dcosy']**2 )
    zenith = np.arcsin(dxy)*180/np.pi
    cond2 = (zenith > 65)
    
    df = df[~(cond1 & cond2)]
    
    if tini >= 0:
        tbase = pd.to_datetime( df['t'].min() )
    else:
        tbase = pd.to_datetime( df['t'].max() )
        
    t = tbase + pd.to_timedelta(tini, unit='h') # Center time for reconstruction
    
    tmin = pd.to_datetime(t) #- pd.to_timedelta(dt/2, unit='h')
    tmax = pd.to_datetime(t) + pd.to_timedelta(dt*60*60, unit='s')
    
    valid  = (df['t'] >= tmin)       & (df['t'] <= tmax)
    
    return(df[valid])

def filter_data(df, tini=0, dt=24,
                dlon=None, dlat=None, dh=None,
                lon_center=None,
                lat_center=None,
                alt_center=None,
                only_SMR=True):
    
    if tini >= 0:
        tbase = pd.to_datetime( df['t'].min() )
    else:
        tbase = pd.to_datetime( df['t'].max() )
        
    t = tbase + pd.to_timedelta(tini, unit='h') # Center time for reconstruction
    
    tmin = pd.to_datetime(t) #- pd.to_timedelta(dt/2, unit='h')
    tmax = pd.to_datetime(t) + pd.to_timedelta(dt*60*60, unit='s')        
    
    valid  = (df['t'] >= tmin)       & (df['t'] <= tmax)
    
    df = df[valid]
    
    #########################
    
    x = df['lons']
    y = df['lats']
    z = df['heights']
    
    # attrs = df.attrs
    # xmid = attrs['lon_center']
    # ymid = attrs['lat_center']
    # zmid = attrs['alt_center']
    
    if lon_center is not None: xmid = lon_center
    else: xmid = np.round( x.median(), 1)
    
    if lat_center is not None: ymid = lat_center
    else: ymid = np.round( y.median(), 1)
    
    if alt_center is not None: zmid = alt_center
    else: zmid = np.round( z.median(), 1)
    
    print('Filter middle point:', xmid, ymid, zmid)
    
    dlon0, dlat0, dh0 = get_xyz_bounds(x, y, z)
    
    if (dlon is None): dlon = dlon0
    if (dlat is None): dlat = dlat0
    if (dh is None)  : dh   = dh0
    
    # Set boundary
    xmin = xmid - dlon/2
    xmax = xmid + dlon/2
    ymin = ymid - dlat/2
    ymax = ymid + dlat/2
    zmin = zmid - dh/2
    zmax = zmid + dh/2
    
    
    valid = (df['lons'] >= xmin)    & (df['lons'] <= xmax)
    valid &= (df['lats'] >= ymin)    & (df['lats'] <= ymax) 
    valid &= (df['heights'] >= zmin) & (df['heights'] <= zmax)
    
    if only_SMR:
        if 'SMR_like' in df.keys():
            valid &= (df['SMR_like'] == 1) 
    
        #tbase = pd.to_datetime(dfraw['t'][0].date())
    
    df = df[valid]
    
    # std = np.std(df['dop_errs'])
    # df = df[ df['dop_errs'] <  2*std]
    
    std = np.std(df['dops'])
    df = df[ np.abs(df['dops']) <=  6*std]
    
    ######
    
    #########################
    ### Clustering ##########
    # dop = df['dops'].values
    # kz = df['braggs_z'].values
    # dxy = np.sqrt( df['dcosx']**2 + df['dcosy']**2 )
    # zenith = np.arcsin(dxy)*180/np.pi
    #
    # X = np.stack([
    #                 dop,
    #                 kz,
    #                 # zenith,
    #                 # coded_links,
    #               ],
    #               axis=1)
    #
    # valid = hierarchical_cluster(X)
    # df = df[valid]
    
    # de = df['dop_errs']
    # std = np.sqrt(np.var(de))
    # cond1 = (de > 3*std)
    #
    # dxy = np.sqrt( df['dcosx']**2 + df['dcosy']**2 )
    # zenith = np.arcsin(dxy)*180/np.pi
    # cond2 = (np.abs(df['dops']) < 30)
    # df = df[cond2]
    
    return(df)

class SMRReader(object):
    
    n_samples = None
    times = None
    latitudes = None
    longitudes = None
    altitudes = None
    
    braggs = None
    dopplers = None
    
    u = None
    v = None
    w = None
    
    x = None
    y = None
    z = None
    
    r = None
    
    file_index = 0
            
    def __init__(self, path, realtime=False, pattern='*'):
        
        files = self.__find_files(path, pattern=pattern)
        
        if len(files) == 0:
            raise ValueError('No files found in %s' %path )
        
        self.path = path
        
        df = self.__read_data(files[0], enu_coordinates=False)
        
        self.df = df
        self.files = files
        
        if realtime:
            self.file_index = len(files) - 2
        else:
            self.file_index = -1
    
        self.set_spatial_center()
        
    def __find_files(self, path, pattern='*'):
        
        files = glob.glob1(path, "%s.h5" %pattern )
        files = sorted(files)
        
        return(files)
    
    def __read_data(self, filename, enu_coordinates=False):
        
        file = os.path.join(self.path, filename)
        
        with h5py.File(file,'r') as fp:

            times   = fp['t'][()]
            link    = fp['link'][()]
            alt     = fp['heights'][:]*1e-3 #To km
            lat     = fp['lats'][:]
            lon     = fp['lons'][:]
            dopplers    = fp['dops'][:]
            dop_errs    = fp['dop_errs'][:]
            braggs      = fp['braggs'][:]
            
            try:
                dcosx       = fp['dcos'][:,0]
                dcosy       = fp['dcos'][:,1]
            except:
                dcosx   =   alt*0
                dcosy   =   alt*0
            
            empty_data = np.zeros_like(times, dtype=np.float64) + np.nan
            
            if 'u' in fp.keys():
                u = fp['u'][:]
                v = fp['v'][:]
                w = fp['w'][:]
            else:
                u = empty_data
                v = empty_data
                w = empty_data
            
            if 'temp' in fp.keys():
                T = fp['temp'][:]   #Kelvin
                P = fp['pres'][:]   #Pascal <> N/m2
                rho = fp['rho'][:]  #kg/m3
                tke = fp['tke'][:]  #m2/s2
            else:
                T = empty_data
                P = empty_data
                rho = empty_data
                tke = empty_data
                
            
            if 'SMR_like' in fp.keys():
                smr_like = fp['SMR_like'][:]
            else:
                smr_like = np.ones_like(times)
        
        self.ini_time = np.nanmin(times)
        self.n_samples = len(times)
        
        t = pd.to_datetime(times, unit='s')
        df = pd.DataFrame(data = t, columns=['t'])
        
        df['times'] = times
        df['link'] = link
        df['lats'] = lat
        df['lons'] = lon
        df['heights'] = alt #km
        
        df['dop_errs'] = dop_errs
        df['braggs_x'] = braggs[:,0]
        df['braggs_y'] = braggs[:,1]
        df['braggs_z'] = braggs[:,2]
        df['dcosx'] = dcosx
        df['dcosy'] = dcosy
        
        #DNS, Scale by 10
        df['dops'] = dopplers
        
        df['u'] = u
        df['v'] = v
        df['w'] = w
        
        df['T'] = T
        df['P'] = P
        df['rho'] = rho
        df['tke'] = tke
        
        df['SMR_like'] = smr_like
        
        df = df[df.dops.notnull()]
        
        ### Meteor geolocation
        # getting the radius at each point, the difference with respect to getting the radius to a reference point is less than 40 meters.
        
        if enu_coordinates:
            #Make calculations with filtered data
            lats = df['lats'].values
            lons = df['lons'].values
            alts = df['heights'].values
            
            # x0,y0,h0 = lla2xyh(lats=lats, lons=lons, alts=alts,
            #                 lat_center=self.lat_center,
            #                 lon_center=self.lon_center,
            #                 alt_center=self.alt_center,
            #                 )
            
            x,y,_ = lla2enu(lats, lons, alts,
                            lat_ref=self.lat_center,
                            lon_ref=self.lon_center,
                            alt_ref=self.alt_center,
                            units='m',
                            )
            
            ##################
            #xyz coordinates in m
            df['x'] = x
            df['y'] = y
            df['z'] = alts*1e3
            
            attrs = {}
            attrs['lon_center'] = self.lon_center
            attrs['lat_center'] = self.lat_center
            attrs['alt_center'] = self.alt_center
            
            df.attrs = attrs
        
        # mask = (df['SMR_like'] == 1)
        # df = df[mask]
        
        df.sort_values(['times'], inplace=True, ignore_index=True)
        
        return(df)
    
    def get_initial_time(self):
        
        return(self.ini_time)
    
    def set_spatial_center(self, lat_center=None, lon_center=None, alt_center=None):
        
        if lat_center is None:
            lat_center = np.round( np.median(self.df['lats'].values), 1)
            
        if lon_center is None:
            lon_center = np.round( np.median(self.df['lons'].values), 1)
        
        if alt_center is None:
            alt_center = np.round( np.median(self.df['heights'].values), 1)
              
        self.lat_center = lat_center
        self.lon_center = lon_center
        self.alt_center = alt_center
        
        print('Meteor middle point (LLA): ', self.lon_center, self.lat_center, self.alt_center)
        
    def get_spatial_center(self):
        
        return(self.lat_center, self.lon_center, self.alt_center)
    
    def get_rx_tx_lla(self, index):
        
        link = self.df['link'][index].decode('ascii', 'replace')
        link = link.split('-')[0]
        tx, rx = link.split('_')[0:2]
        
        tx_lla = stations[tx]
        rx_lla = stations[rx]
        
        return(tx_lla, rx_lla)
        
    def read_next_file(self, enu_coordinates=False):
        
        self.file_index += 1
        
        if self.file_index >= len(self.files):
            return 0
        
        for i in range(1):
            filename = self.files[self.file_index]
            df = self.__read_data(filename, enu_coordinates=enu_coordinates)
            print('\nMeteor file %s [t=%2.1f]' %(filename, self.ini_time) )

            self.df = df
            
            
        self.filename = filename
        
        return(1)   
    
    def save_block(self, rpath, df=None, filename=None, dropnan=True):
        
        if df is None:
            df = self.df
        
        if filename is None:
            fullfile = self.files[self.file_index]
            _, filename = os.path.split(fullfile)
            filename = os.path.join(rpath, 'virt_'+filename)
        
        if dropnan:
            df = df.dropna(subset=['dops'])
        
        braggs = np.array( [df['braggs_x'].values,
                            df['braggs_y'].values,
                            df['braggs_z'].values
                            ]
                        )
        
        fp = h5py.File(filename,'w')

        fp['link']      = df['link'].values
        fp['heights']   = df['heights'].values*1e3 #To m
        fp['lats']      = df['lats'].values
        fp['lons']      = df['lons'].values
        fp['dops']      = df['dops'].values
        fp['dop_errs']  = df['dop_errs'].values
        fp['braggs']    = braggs.T
        # fp['dcosx']     = self.dcosx
        # fp['dcosy']     = self.dcosy  
        fp['t']         = df['times'].values
        fp['u']         = df['u'].values
        fp['v']         = df['v'].values
        fp['w']         = df['w'].values
        
        fp['temp']        = df['T'].values  #Kevin
        fp['pres']        = df['P'].values  #Pascal
        fp['rho']         = df['rho'].values    #kg/m3
        fp['tke']         = df['tke'].values    #m2/s2
        
        if 'SMR_like' in df.keys():
            fp['SMR_like']    = df['SMR_like'].values
        
        for key in df.attrs.keys():
            fp.attrs[key] = df.attrs[key]
            
        fp.close()
        

def read_doppler_hdf5(fn, latref = 54., lonref = 12.5,
                      tbase_txt='2018-11-05',
                      tmin_hr=0, tmax_hr=24
                      ):
    '''
    Read Koki's hdf5 file containing meteor detections and Doppler shifts, returning a pandas DataFrame.
    tbase_txt  - date in year-month-dom ASCII format
    tmin_hr    - minimum hours of a given day to be considered
    tmax-hr    - maximum hours of a given day to be considered
    Columns: data [Hz] - Doppler shifts of meteor detections, related to wind through forward model:
                         data = coeff_u * u  +  coeff_v * v  +  coeff_w * w
             sigma [Hz]- 1-sigma uncertainty in "data"
             coeff_u   - see above
             coeff_v   - see above
             coeff_w   - see above
             lat [deg] - latitude of detected meteor
             lon [deg] - longitude of detected meteor
             x [km]    - x coordinate relative to latref/lonref
             y [km]    - y coordinate relative to latref/lonref
             z [km]    - altitude of detected meteor
             t         - time of meteor detection
             
    '''

    f = h5py.File(fn,'r')
    f.close()
    
    return

def read_wind_hdf5(fn):
    '''
    Read Koki's hdf5 file containing meteor detections and Doppler shifts, returning a pandas DataFrame.
    tbase_txt  - date in year-month-dom ASCII format
    tmin_hr    - minimum hours of a given day to be considered
    tmax-hr    - maximum hours of a given day to be considered
    Columns: data [Hz] - Doppler shifts of meteor detections, related to wind through forward model:
                         data = coeff_u * u  +  coeff_v * v  +  coeff_w * w
             sigma [Hz]- 1-sigma uncertainty in "data"
             coeff_u   - see above
             coeff_v   - see above
             coeff_w   - see above
             lat [deg] - latitude of detected meteor
             lon [deg] - longitude of detected meteor
             x [km]    - x coordinate relative to latref/lonref
             y [km]    - y coordinate relative to latref/lonref
             z [km]    - altitude of detected meteor
             t         - time of meteor detection
             
    '''

    try:
        f = h5py.File(fn,'r')
    except:
        return(None)
    
    times = f['t'][...]
    
    df = {}
    # Forward model is -2*pi*d = bragg_vector dot wind vector
    df['u'] = f['wind_u'][:] 
    df['v'] = f['wind_v'][:] 
    df['w'] = f['wind_w'][:] 
    
    # Location variables
    df['x'] = f['x'][:] 
    df['y'] = f['y'][:] 
    df['z'] = f['z'][:] 
    df['times'] = times
    
    f.close()
    
    return df

def plot_winds(x, y,
               u, v, w,
               min_value=-50,
               max_value=50,
               figpath='./',
               alpha=0.5):
    
    norm = plt.Normalize(min_value, max_value)
    norm2 = plt.Normalize(-10, 10)
    
    _, axs = plt.subplots(3,1, figsize=(10,8), sharex=True, sharey=True)
    
    im = axs[0].scatter(x, y, c=u, cmap='jet', norm=norm, alpha=alpha)
    plt.colorbar(im, ax=axs[0])
    
    im = axs[1].scatter(x, y, c=v, cmap='jet', norm=norm, alpha=alpha)
    plt.colorbar(im, ax=axs[1])
    
    im = axs[2].scatter(x, y, c=w, cmap='jet', norm=norm2, alpha=alpha)
    plt.colorbar(im, ax=axs[2])
    
    # axs[0].set_xlim(64,72)
    # axs[0].set_ylim(70,110)
    plt.tight_layout()
    
    for i in range(3):
        axs[i].grid(True)
    
    figname = os.path.join(figpath, "winds_z_%d.png" %x[0] )
    
    plt.savefig(figname)
    # plt.show()
    
    plt.close()
    
def plot_winds3D(t,x, y,z,
               u, v, w,
               min_value=-50,
               max_value=50,
               figpath='./',
               alpha=0.5):
    
    norm = plt.Normalize(min_value, max_value)
    norm2 = plt.Normalize(-5, 5)
    
    fig = plt.figure(figsize=(10,8))
    
    ax0 = fig.add_subplot(311, projection='3d')
    ax1 = fig.add_subplot(312, projection='3d')
    ax2 = fig.add_subplot(313, projection='3d')
    
    im = ax0.scatter3D(t, x, z, c=u, cmap='jet', norm=norm, alpha=alpha)
    plt.colorbar(im, ax=ax0)
    
    im = ax1.scatter3D(t, x, z, c=v, cmap='jet', norm=norm, alpha=alpha)
    plt.colorbar(im, ax=ax1)
    
    im = ax2.scatter3D(t, x, z, c=w, cmap='jet', norm=norm2, alpha=alpha)
    plt.colorbar(im, ax=ax2)
    
    # ax0.set_ylim(52,72)
    ax0.set_zlim(70,110)
    ax0.set_box_aspect(aspect = (2,1,1))
    
    # ax1.set_ylim(52,72)
    ax1.set_zlim(70,110)
    ax1.set_box_aspect(aspect = (2,1,1))
    
    # ax2.set_ylim(52,72)
    ax2.set_zlim(70,110)
    ax2.set_box_aspect(aspect = (2,1,1))
    
    plt.tight_layout()
    
    figname = os.path.join(figpath, "winds_3D_%d.png" %t[0] )
    
    plt.savefig(figname)
    # plt.show()
    
    plt.close()

def __get_data_mask(df, dx=np.inf, dy=10, dz=4,
                    z0=90):
    
    xmask = (df['x']>-dx/2) & (df['x']<dx/2)
    ymask = (df['y']>-dy/2) & (df['y']<dy/2)
    zmask = (df['z']>z0-dz/2) & (df['z']<z0+dz/2)
    
    return(xmask, ymask, zmask)
        
def test_doppler_files():
    
    path = '/Users/mcordero/Data/IAP/SIMONe/Germany/Simone2018'
    figpath = path
    
    #Read meteor data in LLA coordinates
    meteor_obj = SMRReader(path)
    
    while True:
        
        info = meteor_obj.read_next_file(enu_coordinates=True)
        if info != 1: break
        
        df = meteor_obj.df
        
        xmask, ymask, zmask = __get_data_mask(df)
        
        mask = xmask & ymask & zmask
        
        times = df['times'][mask].values
        lats = df['x'][mask].values
        lons = df['y'][mask].values
        alts = df['z'][mask].values
        
        u = df['u'][mask].values
        v = df['v'][mask].values
        w = df['w'][mask].values
        
        plot_winds(times, lats,
                   u, v, w,
                   figpath=figpath)
        
        plot_winds3D(times,
                     lats, lons, alts,
                     u, v, w,
                     figpath=figpath)

def test_wind_files():
    
    path = '/Users/mcordero/Data/IAP/SIMONe/Winds2018_3D_mask_before'
    figpath = path
    
    files = glob.glob('%s/*.h5' %path)
    
    for file in files:
        
        filename = os.path.join(path, file)
        df = read_wind_hdf5(filename)
        if df is None: break
        
        xmask, ymask, zmask = __get_data_mask(df)
        
        times = df['times']
        lats = df['x'][xmask]
        lons = df['y'][ymask]
        alts = df['z'][zmask]
        
        u = df['u'][:,xmask]
        v = df['v'][:,xmask]
        w = df['w'][:,xmask]
        
        u = u[:,:,ymask,zmask]
        v = v[:,:,ymask,zmask]
        w = w[:,:,ymask,zmask]
        
        T,X,Y,Z = np.meshgrid(times, lats, lons, alts, indexing='ij')
        
        plot_winds(T.flatten(), X.flatten(),
                   u.flatten(), v.flatten(), w.flatten(),
                   figpath=figpath)
        
        plot_winds3D(T.flatten(),
                     X.flatten(), Y.flatten(), Z.flatten(),
                     u.flatten(), v.flatten(), w.flatten(),
                     figpath=figpath)
              
if __name__ == '__main__':
    test_doppler_files()
    # test_wind_files()
    