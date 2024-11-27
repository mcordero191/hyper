import numpy as np

"""
Reference:

bowring b., 1976. transformation from spatial to geographical coordinates, 
survey review 23, pp. 323-327. 

bowring, b.r. (1985) the accuracy of geodetic latitude and height equations, 
survey review, vol. 28, no. 218, pp. 202-206. 
"""
import sys

"""
------------------------------------------------------------------------------- 
wgs-84 defining parameters. 
------------------------------------------------------------------------------- 
"""
a = 6378.137 #Km
f = 1.0 / 298.257223563
"""
------------------------------------------------------------------------------- 
wgs-84 derived parameters. 
------------------------------------------------------------------------------- 
"""
one_f = 1.0 - f;

b = a * one_f; # semi-minor axis 
e2 = f * (2.0 - f); # first eccentricity squared 
epsilon = e2 / (1.0 - e2); # second eccentricity squared 
b_a = one_f;

"""
-------------------------------------------------------------------------------
"""
rad2deg = 180.0/np.pi
deg2rad = np.pi/180.0


def lat2km(lat, lat_ref=0):
    
    d = (lat-lat_ref)*np.sin(np.pi/180.0)*a
    
    return(d)

def lon2km(lon, lat, lon_ref=0):

    d = (lon-lon_ref)*np.pi/180.0*np.cos(np.pi*lat/180.0)*a
    
    return(d)

def getMatrixRotation(latitude, longitude):
    """
    Inputs:
        latitude, longitude      :    Geodetic coordinates. Latitude, longitude in degrees
    """
    
    phi = latitude*deg2rad
    lambd = longitude*deg2rad
    
    R = [
         [-np.sin(lambd), -np.sin(phi)*np.cos(lambd), np.cos(phi)*np.cos(lambd)],
         [ np.cos(lambd), -np.sin(phi)*np.sin(lambd), np.cos(phi)*np.sin(lambd)],
         [       0*lambd,                np.cos(phi),               np.sin(phi)],
        ]
    
    return np.array(R, dtype=float)

def lla2ecef(latitude, longitude, h):
    """
    Inputs:
        latitude, longitude, h  :    Geodetic coordinates.
                                     Latitude, longitude in degrees and height in km
        
    Outputs:
        X, Y, Z        :    ECEF coordinates (km)
    """
    
    phi = latitude*deg2rad
    lambd = longitude*deg2rad
    
    N = a/( np.sqrt(1 - e2*np.sin(phi)**2) )
    
    X = (N + h)*np.cos(phi)*np.cos(lambd)
    Y = (N + h)*np.cos(phi)*np.sin(lambd)
    Z = (N*(1 - e2) + h)*np.sin(phi)
    
    return(X, Y, Z)

def lla2ecef_radius(latitude, longitude, h):
    """
    Convert geodetic coordinates to earth-centered earth-fixed
    (ECEF) coordinates...
    
    Parameters:
    -----------
    lat : array_like with shape(N,) or scalar, float,
    geodetic latitudes (dezimal, northern latitudes are positve!)
    lon : array_like with shape(N,) or scalar, float,
    geodetic longitudes (dezimal, eastern longitudes are positve!)
    height : array_like with shape(N,) or scalar, float,
    geodetic heights (in km)
    
    Returns:
    --------
    radius : np.array shape(N,), float,
    array for each ECEF position component
    """
    
    x,y,z = lla2ecef(latitude, longitude, h)
    radius = np.sqrt(x**2+y**2+z**2)
    
    return radius

def lla2xyh(lats, lons, alts, lat_center, lon_center, alt_center, units='m'):
    '''
    '''
    # radius = lla2ecef_radius(lats, lons, alts)*1e3 #km to m
    #
    # x = radius * np.cos( np.deg2rad(lats) ) * np.deg2rad(lons-lon_center)
    # y = radius * np.deg2rad(lats-lat_center)
    
    x, y, _ = lla2enu(lats, lons, alts, lat_center, lon_center, alt_center, units=units)
    
    h = alts
    if units == 'm': h = h*1e3
    
    return(x,y,h)    

def xyh2lla(x, y, h, lat_ref, lon_ref, alt_ref):
    """
    Inputs:
        x    :    Local coordinate (km)
        y    :    Local coordinate (km)
        h    :    Local coordinate (km)
        
        latitude  :    latitude in degrees
        longitude :    longitude in degrees
        alt       :    altitude in kilometers
    """
    
    # radius = lla2ecef_radius(lat_ref, lon_ref, alt_ref) #km to m
    # radius -= h
    #
    # x = radius * np.cos( np.deg2rad(lats) ) * np.deg2rad(lons-lon_center)
    # y = radius * np.deg2rad(lats-lat_center)
    
    h0 = h - alt_ref
    
    R = lla2ecef_radius(lat_ref, lon_ref, alt_ref)
    
    d = np.sqrt(x**2 + y**2)
    angle = d/(R+h0)
    z = (R+h0)*np.cos(angle) - R
    
    lat, lon, alt = enu2lla(x, y, z, lat_ref, lon_ref, alt_ref)
    # lat, lon, alt = ecef2lla(X, Y, Z)
    
    return lat, lon, alt

def ecef2lla(X, Y, Z):
    
    """
    main algorithm. in bowring (1985), u is the parametric latitude. it is crucial 
    to maintain the appropriate signs for the sin(u) and sin(latitude) in the equations 
    below.
    
    Input:
        X, Y, Z        :    ECEF coordinates (km)
        
    Output:
        latitude, longitude, h  :    Geodetic coordinates.
                                     Latitude, longitude in degrees and height in km
    """ 
    
    p2 = X**2 + Y**2
    r2 = p2 + Z**2
    
    p = np.sqrt(p2)
    r = np.sqrt(r2)
    
    """
    equation (17) from bowring (1985), shown to improve numerical accuracy in latitude 
    """
    tanu = b_a * (Z / p) * (1 + epsilon * b / r); 
    tan2u = tanu * tanu;
    
    """
    avoid trigonometric functions for determining cos3u and sin3u
    """
    
    cos2u = 1.0 / (1.0 + tan2u); 
    cosu = np.sqrt(cos2u); 
    cos3u = cos2u * cosu;
    
    sinu = tanu * cosu; 
    sin2u = 1.0 - cos2u; 
    sin3u = sin2u * sinu;
    
    """
    equation (18) from bowring (1985) 
    """
    
    tanlat = (Z + epsilon * b * sin3u) / (p - e2 * a * cos3u);

    tan2lat = tanlat * tanlat; 
    cos2lat = 1.0 / (1.0 + tan2lat); 
    sin2lat = 1.0 - cos2lat;
    
    coslat = np.sqrt(cos2lat); 
    sinlat = tanlat * coslat;
    
    lambd = np.arctan2(Y, X); 
    phi = np.arctan(tanlat);
    
    """
    equation (7) from bowring (1985), shown to be numerically superior to other 
    height equations. note that equation (7) from bowring (1985) writes the last 
    term as a^2 / nu, but this reduces to a * sqrt(1 - e^2 * sin(latitude)^2), because 
    nu = a / sqrt(1 - e^2 * sin(latitude)^2).
    """
    
    h = p * coslat + Z * sinlat - a * np.sqrt(1.0 - e2 * sin2lat);
    
    latitude = phi*rad2deg
    longitude = lambd*rad2deg
    
    return latitude, longitude, h

def _ecef2lla(X, Y, Z):
    
    p = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Z*a, p*b)
    
    lambd = np.arctan2(Y, X)
    phi = np.arctan2(Z + epsilon*b*np.sin(theta)**3, p - e2*a*np.cos(theta)**3)
    
    
    N = a/( np.sqrt(1 - e2*np.sin(phi)**2) )
    
    h = p/np.cos(phi) - N
    
    latitude = phi*rad2deg
    longitude = lambd*rad2deg
    
    return latitude, longitude, h

def enu2ecef(x, y, z, latitude, longitude, h=0):
    """
    Inputs:
        x    :    Local coordinate (km)
        y    :    Local coordinate (km)
        z    :    Local coordinate (km)
        
        latitude  :    latitude in degrees
        longitude :    longitude in degrees
        h         :    altitude in kilometers
        
        #R    :    Matrix rotation. Use getMatrixRotation to get R using phi and lambda.
        
    Outputs:
        X    :    ECEF coordinate
        Y    :    ECEF coordinate
        Z    :    ECEF coordinate
    """
    R = getMatrixRotation(latitude, longitude)
    
    Xr, Yr, Zr = lla2ecef(latitude, longitude, h)
    
    X = R[0,0]*x + R[0,1]*y + R[0,2]*z + Xr
    Y = R[1,0]*x + R[1,1]*y + R[1,2]*z + Yr
    Z = R[2,0]*x + R[2,1]*y + R[2,2]*z + Zr
    
    return X, Y, Z

def ecef2enu(X, Y, Z, latitude, longitude, h=0):
    
    """
    Inputs:
        X    :    ECEF coordinate
        Y    :    ECEF coordinate
        Z    :    ECEF coordinate
        
        latitude  :    latitude in degrees
        longitude :    longitude in degrees
        h         :    altitude in km
        
        #R    :    Matrix rotation. Use getMatrixRotation to get R using phi and lambda.
        
    Outputs:
        
        x    :    Local coordinate
        y    :    Local coordinate
        z    :    Local coordinate
    """
    
    R = getMatrixRotation(latitude, longitude)
    
    Xr, Yr, Zr = lla2ecef(latitude, longitude, h)
    
    Xp = X - Xr
    Yp = Y - Yr
    Zp = Z - Zr
    
    x = R[0,0]*Xp + R[1,0]*Yp + R[2,0]*Zp
    y = R[0,1]*Xp + R[1,1]*Yp + R[2,1]*Zp
    z = R[0,2]*Xp + R[1,2]*Yp + R[2,2]*Zp
    
    return x, y, z

def lla2enu(lat, lon, alt, lat_ref, lon_ref, alt_ref, units='km'):
    """
    Inputs:
        latitude, longitude, h  :    Geodetic coordinates.
                                     Latitude, longitude in degrees and height in km
        
    Outputs:
        x, y, z        :    ENU coordinates (km or m)
    """
    
    #LLA to Geocentric Coordinates
    X, Y, Z = lla2ecef(lat, lon, alt)
    x, y, z = ecef2enu(X, Y, Z, lat_ref, lon_ref, alt_ref)
    
    if units == 'm':
        return 1e3*x, 1e3*y, 1e3*z
    
    return x, y, z

def enu2lla(x, y, z, lat_ref, lon_ref, alt_ref):
    """
    Inputs:
        x    :    Local coordinate (km)
        y    :    Local coordinate (km)
        z    :    Local coordinate (km)
        
        latitude  :    latitude in degrees
        longitude :    longitude in degrees
        h         :    altitude in kilometers
    """
    X, Y, Z = enu2ecef(x, y, z, lat_ref, lon_ref, alt_ref)
    lat, lon, alt = ecef2lla(X, Y, Z)
    
    return lat, lon, alt

def get_vector(rx, tx, unit_vector=True):
    '''
    Get the vector between tx and rx. The origin is rx.
    
    Inputs:
        tx    :    lat, lon and alt of the transmiter location. alt in km
        rx    :    lat, lon and alt of the receiver location. alt in km
    '''
    #LLA to Geocentric Coordinates
    Xt, Yt, Zt = lla2ecef(tx[0], tx[1], h=tx[2])
    #Xr, Yr, Zr = lla2ecef(rx[0], rx[1], h=rx[2])
    
    #Geocentric to Local coordinates
    x,y,z = ecef2enu(Xt, Yt, Zt, rx[0], rx[1], rx[2])
    
    d = np.sqrt(x**2 + y**2 + z**2)
    
    if unit_vector:
        return d, (x/d, y/d, z/d)
    
    return x, y, z

def dcos_to_lla(k, ranges,
                rx = [69.3, 16.04, 0.010],
                tx = [69.3, 16.04, 0.010],
                range_offset=0):
    
    """
    Inputs:
        k         :    array-like. Beam direction (dcosx, dcosy, dcosz).
                       Where dcosx, dcosy and dcosz are 2D arrays with dimension (nx, ny) 
        ranges    :    vector-like. Total range
        rx        :    Rx location in latitude, longitude and altitude
        tx        :    Tx location in latitude, longitude and altitude
        
            Example:
                rx = [54.630211, 13.373216, 0.010],
                tx = [54.118309, 11.769558, 0.070],
    """
    nx, ny = k[0].shape
    nranges = len(ranges)
    
    ##real_range = np.empty((nx, ny, nranges))
    
    x = np.empty((nx, ny, nranges))
    y = np.empty((nx, ny, nranges))
    z = np.empty((nx, ny, nranges))
    
    d, uv = get_vector(rx, tx)
    
    cos_alfa = uv[0]*k[0] + uv[1]*k[1] + uv[2]*k[2]
    
    for iz in range(nranges):
        r= ranges[iz] - range_offset
        r = (r**2 - d**2)/(2*(r - d*cos_alfa))
        
        #real_range[:,:,iz] = Rnew
        
        x[:,:,iz] = r*k[0]
        y[:,:,iz] = r*k[1]
        z[:,:,iz] = r*k[2]
    
    #return x, y, z, real_range
        
    X, Y, Z = enu2ecef(x, y, z, latitude=rx[0], longitude=rx[1], h=rx[2])
    
    latitude, longitude, h = ecef2lla(X, Y, Z)
     
    return latitude, longitude, h

if __name__ == '__main__':
    
    import time
    
    lat0 = 46.017
    lon0 = 7.750
    h0 = 1.673
    
    lat = 45.976
    lon = 7.658
    h = 4.531

    x, y, z = lla2enu(lat, lon, h, lat0, lon0, h0)
    print(x, y, z)
    
    lla = enu2lla(x, y, z, lat0, lon0, h0)
    print(lla)
    
    X, Y, Z = -41756.45163681,   -662.87945936, -19569.52383146
    x, y, z = 19022.1, -279.577, -40867.2
    latitude, longitude, h = 50.0, 0.606060606, 100.0
    
    print(ecef2lla(X, Y, Z))
    
    print(ecef2enu(X, Y, Z, latitude, longitude, h))
    
    #Xp, Yp, Zp = enu2ecef(x, y, z, latitude, longitude, h=h)
    
    #print Xp, Yp, Zp
    
    sys.exit(0)
    
    
    lat_juh = 54.62
    long_juh = 13.37
     
    lat_Ereg = 57.39
    long_Ereg = 13.37
     
    x = 0.0
    y = 350
    z = 90
     
    Xr, Yr, Zr = lla2ecef(lat_juh, long_juh, h=0.01)
    print("Radar coordinates: ", Xr, Yr, Zr)
    
    ini = time.time()
    latitude, longitude, h = ecef2lla(Xr, Yr, Zr)
    print(time.time() - ini)
    print("Radar coordinates: ", latitude, longitude, h)
    
    ini = time.time()
    latitude, longitude, h = _ecef2lla(Xr, Yr, Zr)
    print(time.time() - ini)
    print("Radar coordinates: ", latitude, longitude, h)
     
    Xp, Yp, Zp = enu2ecef(x, y, z, lat_juh, long_juh, h=0.01)
    print("Target coordinates: ", Xp, Yp, Zp)
     
    Xr, Yr, Zr = lla2ecef(lat_Ereg, long_Ereg, h=0.01)
    print("E region coordinates: ", Xr, Yr, Zr)
     
    x,y,z = ecef2enu(Xp, Yp, Zp, lat_Ereg, long_Ereg, h=0.07)
    print("Local coordinates related to E region coordinates: ", x, y, z)
    
    X, Y, Z = enu2ecef(x, y, z, lat_Ereg, long_Ereg, h=0.07)
    print(X, Y, Z)
#     tx_gps = [54.118309, 11.769558, 0.070]
#     rx_gps = [54.630211, 13.373216, 0.010]
#     tlink = 'Kborn-Jruh'
#     
#     Xt, Yt, Zt = lla2ecef(tx_gps[0], tx_gps[1], h=tx_gps[2])
#     Xr, Yr, Zr = lla2ecef(rx_gps[0], rx_gps[1], h=rx_gps[2])
#     
#     x,y,z = ecef2enu(Xt, Yt, Zt, rx_gps[0], rx_gps[1], Xr, Yr, Zr)
#     
#     print "Tx coordinates: ", Xt, Yt, Zt
#     print "Rx coordinates: ", Xr, Yr, Zr
#     
#     m = np.sqrt(x**2 + y**2 + z**2)
#     
#     print "Distance from Rx to Tx: %f = (%f, %f, %f)" %(m, x, y, z)
#     
#     print "Unit vector from Rx to Tx: ", x/m, y/m, z/m
    