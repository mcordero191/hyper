'''
Created on 21 Nov 2022

@author: mcordero
'''
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy import io
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression

from functools import partial

import datetime
import msise00

file_viscocity = '/Users/mcordero/Data/IAP/SIMONe/viscosity_20181102.mat'
file_density = '/Users/mcordero/Data/IAP/SIMONe/msis_20181102.txt'

def run_msis(times, alt, glat, glon):
    """
    Return total density (Kg/m3) and temperature (K) using MSIS-00 model.
    
    The density and temperature dimensions are [ntimes, nalts, nlats, nlons]
    
    Inputs:
        times    :    datetime object with the specified date and time
        alt      :    altitude in km
        glat     :    geographic latitude in degrees
        glon     :    geographic longitude in degrees
        
    Libs:
        atmos    :    xarray-like
        
            Dimensions:     (time, alt, lat, lon)
            Coordinates:
              * time        (time) datetime64[ns] 2013-03-31T12:00:00
              * alt         (alt_km) int64
              * lat         (lat) float64
              * lon         (lon) float64
            Data variables:
                He          (time, alt_km, lat, lon) float64 
                O           (time, alt_km, lat, lon) float64 
                N2          (time, alt_km, lat, lon) float64 
                O2          (time, alt_km, lat, lon) float64
                Ar          (time, alt_km, lat, lon) float64 
                Total       (time, alt_km, lat, lon) float64 
                H           (time, alt_km, lat, lon) float64 
                N           (time, alt_km, lat, lon) float64 
                AnomalousO  (time, alt_km, lat, lon) float64 
                Tn          (time, alt_km, lat, lon) float64 
                Texo        (time, alt_km, lat, lon) float64 
    
    Output:
        rho    :    total neutral density (time, alt_km, lat, lon)
        T      :    neutral temperature (time, alt_km, lat, lon)
    """
    atmos = msise00.run(times, alt, glat, glon)
    atmos = atmos.squeeze()
    
    T = atmos['Tn'].values
    rho = atmos['Total'].values
    
    return(rho, T)
    
def get_msis_parms(times, alt, glat, glon, derivative=False, altitude_mean=False):
    """
    Returns the neutral density, neutral temperature, and the kinematic viscosity
    as MSIS-00 predicts.
    
    Inputs:
    
        times    :    range of datetimes
        alt      :    range of altitudes in km
        glat     :    range of latitudes in degrees
        glon     :    range of longitudes in degrees
        
    Outputs:
        rho      :    density (kg/m3) [ntimes, nalts, nlats, nlots]
        T        :    temperature (K) [ntimes, nalts, nlats, nlots]
        nu       :    viscosity (m2/s) [ntimes, nalts, nlats, nlots]
        
        rho_z    :    if derivative is True, returns the derivative of density respect to altitude.
                      (kg/m3/m)
    """
    
    rho, T = run_msis(times, alt, glat, glon)
    nu = viscosity(T, rho)
    
    if not derivative:
        return(rho, nu)
    
    rho_z = np.empty_like(rho) + np.nan
    rho_z[:,1:] = 1e-3*(rho[:,1:] - rho[:,:-1])/(alt[1:] - alt[:-1])[None,:]
    
    if not altitude_mean:
        return(rho, rho_z, nu)
    
    rho = np.mean(rho, axis=0)
    rho_z = np.nanmean(rho_z, axis=0)
    nu = np.mean(nu, axis=0)
    
    return(rho, rho_z, nu)

def plot_and_save(times, alt, rho, rho_z, nu, T,
                  figfile="./test.png", cmap='jet'):
    
    _, axs = plt.subplots(1, 5, sharey="row", figsize=(12,3))
    
    ax = axs[0]
    ax.set_title('Density')
    # im = ax.pcolormesh(times, alt, np.log10(rho).T, cmap=cmap)
    # plt.colorbar(im, ax=ax, label='log10 (kg/m3)')
    ax.plot(rho, alt, "o--")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel(r"$\rho$ (kg/m3)")
    ax.set_xscale('log')
    
    ax = axs[1]
    ax.set_title('Change of density in z')
    # im = ax.pcolormesh(times, alt, np.log10(rho_z).T, cmap=cmap)
    # plt.colorbar(im, ax=ax, label='log10 (kg/m3)')
    ax.plot(-rho_z, alt, "o--")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel(r"$-\rho_z$ (kg/m3/m)")
    ax.set_xscale('log')
    
    ax = axs[2]
    ax.set_title(r'Ratio of $\rho$ and $\rho_z$')
    # im = ax.pcolormesh(times, alt, np.log10(rho_z).T, cmap=cmap)
    # plt.colorbar(im, ax=ax, label='log10 (kg/m3)')
    ax.plot(-rho_z/rho, alt, "o--")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel(r"$-\rho_z/\rho$ (1/m)")
    ax.set_xscale('log')
    
    ax = axs[3]
    ax.set_title('Kinematic viscosity')
    # im = ax.pcolormesh(times, alt, np.log10(nu).T, cmap=cmap)
    # plt.colorbar(im, ax=ax, label='log10 (m2/s)')
    ax.plot(nu, alt, "o--")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel(r"$\nu$ (m2/s)")
    ax.set_xscale('log')
    
    ax = axs[4]
    ax.set_title('Temperature')
    # im = ax.pcolormesh(times, alt, T.T, cmap=cmap)
    # plt.colorbar(im, ax=ax, label='K')
    ax.plot(T, alt, "o--")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel(r"T ($ˆ\circ$K)")
    
    for ax in axs:
        ax.grid(True)
        ax.label_outer()
    
    plt.tight_layout()
    plt.savefig(figfile)
    plt.close()
    
    
def get_msis_mean_values(dt, alt, glat, glon,
                         derivative=True,
                         time_average=True,
                         time_range=24,
                         plot_values=False,
                         figfile='./msis_values.png'):
    """
    Returns the neutral density, neutral temperature, and the kinematic viscosity
    of the next 24h as MSIS-00 predicts. 
    
    Inputs:
    
        dt       :    datetime object
        alt      :    range of altitudes in km
        glat     :    range of latitudes in degrees
        glon     :    range of longitudes in degrees
        
    Outputs:
        rho      :    density (kg/m3) [ntimes, nalts, nlats, nlots]
        rho_z    :    if derivative is True, returns the derivative of density respect to altitud.
                      (kg/m3/m)
        nu       :    viscosity (m2/s) [ntimes, nalts, nlats, nlots]
        
        
    """
    
    hours = np.arange(0, time_range, 1.0)
    times = [dt + datetime.timedelta(hours=x) for x in hours]
    
    rho, T = run_msis(times, alt, glat, glon)
    nu = viscosity(T, rho)
    
    if not derivative:
        return(rho, nu)
    
    rho_z = np.empty_like(rho) + np.nan
    
    dz = 1e3*(alt[1:] - alt[:-1])
    drho = rho[:,1:] - rho[:,:-1]
    
    rho_z[:,1:] = drho/dz[None,:]
        
    if not time_average:
        return(rho, rho_z, nu)
    
    rho     = np.mean(rho, axis=0)
    rho_z   = np.nanmean(rho_z, axis=0)
    nu      = np.mean(nu, axis=0)
    T       = np.mean(T, axis=0)
    
    if plot_values:
        plot_and_save(times, alt, rho, rho_z, nu, T,
                      figfile=figfile)
        
        
    return(rho, rho_z, nu)

def read_msis(filename=file_density):
    
    alts = []
    rhos = []
    temps = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        
        for line in lines:
            
            # if not line.strip():
            #     break
                
            parms = line.split()
            
            if len(parms) != 8:
                continue
            
            #rho = g/cm^3 <> 1e-3 kg/1e-6 m^3 <> 1e3 kg/m^3
            #T = K
            yy, mm, dd, doy, hh, z, rho, T = parms
            
            try:
                yy = int(yy)
            except:
                continue
            
            if yy<1950:
                continue
            
            alts.append(float(z))
            rhos.append(float(rho)*1e3)
            temps.append(float(T))
        
    return(alts, rhos, temps)
    
def read_viscocity(filename=None):
    
    if filename is None:
        filename = file_viscocity
        
    fp = io.loadmat(filename)
    
    h = fp['z'].ravel()
    nu = fp['nu_final'].ravel()
    
    return(h, nu)

def viscosity(t, rho=None, p=None):
    """
    Calculate kinematic viscosity from air pressure and temperature
    Inputs:
        t    :    temperature in Kelvins
        rho  :    density  in kg/m3
        p    :    pressure in Pascals (N/m2)
        
    Output:
        nu    :    viscosity (m2/s)
    """
    
    beta        = 0.000001458;
    R_gas       = 287.04;
    
    if rho is None:
        rho = p/(R_gas*t);
    
    nu_kmcm = beta*np.sqrt(t)/(1+110.4/t)/rho;

    return(nu_kmcm)

def Brunt_Vaisala_freq(g, rho, rho_z):
    
    N = np.sqrt(-g/rho*rho_z)
    
    return(N)

def gravity(alt, R=6378.137, g=9.8):
    '''
    Inputs:
        alt    :    altitude in km
    '''
    gh = g*(1.0 - 2*alt/R)
    
    return(gh)
    
    
def exp_fit(x, y):
    
    xp = x.reshape((-1,1))
    yp = np.log(y)
    
    model = LinearRegression()
    model.fit(xp, yp)
    
    r_sq = model.score(xp, yp)
    
    print(f"Exp function fit: coefficient of determination: {r_sq}")
    
    a = model.intercept_
    b = model.coef_[0]
    
    scale = np.exp(a)
    
    return(scale, b)
    
def exp_function(t, a, b, c=0):
    
    return (a * np.exp(b * t) + c)

def derivative_exp_function(t, a, b):
    
    return (a * b* np.exp(b * t))
 
def exp_function_ln(t, a, b):
    
    return (np.log(a) - b * t)


def get_func_rho_temp(filename=None, derivative=False):
    
    if filename is None: filename = file_density
        
    h, rho, T = read_msis(filename)
    
    f_rho = UnivariateSpline(h, rho, k=4, s=0)
    f_T = UnivariateSpline(h, T, k=4, s=0)

    if derivative:
        #Derivatives
        f_rho_h = f_rho.derivative()
        f_T_h = f_T.derivative()
        
        return(f_rho, f_T, f_rho_h, f_T_h)
        
    return(f_rho, f_T)

def get_func_rho_temp_z(h, new_h, filename=None, derivative=False,
                        rho=None, T=None):
    
    h_unique, indices, indices_inv = np.unique(h, return_index=True, return_inverse=True)
    f_rho, f_T = get_func_rho_temp(filename)
    
    if rho is not None:
        popt, _ = curve_fit(exp_function, new_h, rho, p0=[1e0, 1e-4])
        f_new_rho = partial(exp_function, a=popt[0], b=popt[1], c=popt[2])
        
        if derivative:
            f_new_rho_p = partial(derivative_exp_function, a=popt[0], b=popt[1])
    else:
        rho = f_rho(h[indices])
        f_new_rho = UnivariateSpline(new_h[indices], rho, k=1, s=0)
    
        if derivative:
            #Derivatives
            f_new_rho_p = f_new_rho.derivative()
        
    
    if True: #T is None:
        T = f_T(h[indices])
        f_new_T = UnivariateSpline(new_h[indices], T, k=1, s=0)
        
        if derivative:
            f_new_T_p = f_new_T.derivative()
            return(f_new_rho, f_new_T, f_new_rho_p, f_new_T_p)
    else:
        # np.unique()
        raise ValueError
    
    return(f_new_rho, f_new_T)

def get_func_nu(filename=None, derivative=False):
        
    h, nu = read_viscocity(filename)
    
    f_nu = UnivariateSpline(h, nu, k=4, s=0)

    if derivative:
        #Derivatives
        f_nu_h = f_nu.derivative()
        
        return(f_nu, f_nu_h)
    
    return(f_nu)

def get_func_nu_z(h, new_h, filename=None, derivative=False, rho=None, T=None):
    
    _, indices = np.unique(h, return_index=True, return_inverse=False)
    
    if (rho is None) or (T is None):
        f = get_func_nu(filename)
        nu = f(h)
    else:
        nu = viscosity(T, rho=rho)
    
    f_new_nu = UnivariateSpline(new_h[indices], nu[indices], k=4, s=0)
    
    if derivative:
        #Derivatives
        f_new_nu_z = f_new_nu.derivative()
        
        return(f_new_nu, f_new_nu_z)
    
    return(f_new_nu)

class MSIS():
    
    def __init__(self,
                 ini_date,
                 glat,
                 glon,
                 time_range=24,
                 min_alt=60,
                 max_alt=120,
                 plot_values=False,
                 units='m'):
        '''
        Altitude range in m or Km
        '''
        # ini_date = datetime.datetime.utcfromtimestamp(epoch)
        
        self.ini_date = ini_date
        self.time_range = time_range
        self.alts = np.arange(min_alt, max_alt, 5, dtype=np.float64)
        self.glat = glat
        self.glon = glon
        self.units = units
        
        self.get_values(plot_values)
        self.interpolate_values()
    
    def get_values(self, plot_values=False):
        
        rho, rho_z, nu = get_msis_mean_values(self.ini_date,
                                             self.alts,
                                             self.glat, 
                                             self.glon,
                                             derivative=True,
                                             time_average=True,
                                             time_range=self.time_range,
                                             plot_values=plot_values)
        
        self.rho = rho
        self.rho_z = rho_z
        self.nu = nu
        
    def interpolate_values(self):
        
        if self.units == 'm':
            scaling = 1e3
        elif self.units == 'km':
            scaling = 1
        else:
            raise ValueError('units can take only m or km, units=%s' %self.units)
        
        indices = np.where(np.isfinite(self.rho))
        # f_rho = UnivariateSpline(self.alts[indices], self.rho[indices], k=4, s=0)
        
        # popt, _ = curve_fit(exp_function, self.alts[indices]*scaling, self.rho[indices],
        #                     sigma=0.01*self.rho[indices],
        #                     p0=[1e0, 1e-4],
        #                     bounds = (0, 1e3)
        #                     )
        
        popt   = exp_fit(self.alts[indices]*scaling, self.rho[indices])
        
        f_rho   = partial(exp_function, a=popt[0], b=popt[1])
        f_rho_z = partial(derivative_exp_function, a=popt[0], b=popt[1])
        
        # indices = np.where(np.isfinite(self.rho_z))
        # f_rho_z = UnivariateSpline(self.alts[indices], self.rho_z[indices], k=4, s=0)
        
        indices = np.where(np.isfinite(self.nu))
        self.f_nu = UnivariateSpline(self.alts[indices]*scaling, self.nu[indices], k=4, s=0)
        
        self.f_rho = f_rho
        self.f_rho_z = f_rho_z
        
    def get_rho(self, alts):
        """
        Inputs:
            alts    :    m or km
        Outputs:
            rho      :    density (kg/m3) [ntimes, nalts, nlats, nlots]
        """
        # alts = np.asarray(alts, dtype=np.float64)
        return( self.f_rho(alts) )
    
    def get_rho_z(self, alts):
        """
        Inputs:
            alts    :    m  or km
        Outputs:
            rho_z    :    derivative of density respect to altitud.
                          (kg/m3/m)
        """
        # alts = np.asarray(alts, dtype=np.float64)
        return( self.f_rho_z(alts) )
    
    def get_nu(self, alts):
        """
        Inputs:
            alts    :    m or km
        Outputs:
            nu       :    viscosity (m2/s) [ntimes, nalts, nlats, nlots]
        """
        
        return( self.f_nu(alts) )
    
    def get_N(self, alts):
        """
        Inputs:
            alts    :    m  or km
        Outputs:
            N        :    Brunt Vaisala frequency
        """
        # alts = np.asarray(alts, dtype=np.float64)
        
        if self.units == 'm':
            altKm = alts*1e-3
        else:
            altKm = alts
        
        g = gravity(altKm)
        rho = self.get_rho(alts)
        rho_z = self.get_rho_z(alts)
        
        N = Brunt_Vaisala_freq(g, rho, rho_z)
        
        return(N)
    

if __name__ == '__main__':
    
    import datetime
    import matplotlib.pyplot as pl
    
    hours = np.arange(0,24,1.)
    
    base = datetime.datetime(2018, 11, 3, 0, 0, 0)
    times = [base + datetime.timedelta(hours=x) for x in hours]
    
    # t0 = datetime(2015, 12, 13, 10, 0, 0)
    # t1 = datetime(2015, 12, 14, 10, 0, 0)
    #
    # times = np.arange(t0, t1, datetime.timedelta(hours=1))
    
    alt = np.arange(70,110,2.)#*1e3
    glat = 53
    glon = 11
    
    rho, rho_z, nu = get_msis_parms(times, alt, glat, glon, derivative=True, altitude_mean=False)
    rho_z = -1*rho_z
    
    # pl.plot( rho[0,:,0,0], alt, 'o--')
    pl.figure(figsize=(10,6))
    pl.subplot(411)
    pl.pcolormesh(hours, alt, np.log10(rho).T, cmap='jet')
    pl.title('Density (log kg/m3)')
    pl.colorbar()
    
    pl.subplot(412)
    pl.pcolormesh(hours, alt, np.log10(rho_z).T, cmap='jet',
                  # vmin = np.log10(np.nanmin(rho_z)),
                  # vmax = np.log10(np.nanmax(rho_z)),
                )
    pl.title('Derivative of density (log kg/m4)')
    pl.colorbar()
    
    pl.subplot(413)
    pl.pcolormesh(hours, alt, np.log10(nu).T, cmap='jet')
    pl.title('Viscosity (log m2/s)')
    pl.colorbar()
    
    # pl.subplot(414)
    # pl.pcolormesh(hours, alt, T.T, cmap='jet')
    # pl.title('Temperature (K)')
    # pl.colorbar()
    
    pl.tight_layout()
    pl.show()
    
    # f = get_func_nu()
    #
    # h = np.arange(80,110)
    # nu = f(h)
    #
    # pl.plot( nu, h, 'o--')
    # pl.show()
    
    # h, rho, T = read_msis()
    #
    # pl.figure()
    #
    # pl.subplot(121)
    # pl.plot(rho, h, 'o-', label='Density')
    # pl.xscale('log')
    # pl.grid()
    # pl.legend()
    #
    # pl.subplot(122)
    # pl.plot(T, h, 'x-', label='Temperature')
    # pl.grid()
    # pl.legend()
    #
    # pl.tight_layout()
    # pl.show()
    
    # x = np.linspace(0, 10, 70)
    # y = np.exp(2*x) +1
    # spl = UnivariateSpline(x, y, k=4, s=0)
    # f = spl.derivative()
    #
    # yp = f(x)
    #
    # pl.plot(x,y, label='y')
    # pl.plot(x,yp, label = 'yp')
    # pl.legend()
    # pl.show()

    