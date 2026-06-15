from __future__ import annotations

import numpy as np


a = 6378.137
f = 1.0 / 298.257223563
one_f = 1.0 - f
b = a * one_f
e2 = f * (2.0 - f)
epsilon = e2 / (1.0 - e2)
rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0


def lat2km(lat, lat_ref=0.0):

    return (lat - lat_ref) * deg2rad * a


def lon2km(lon, lat, lon_ref=0.0):

    return (lon - lon_ref) * deg2rad * np.cos(np.pi * np.asarray(lat) / 180.0) * a


def get_matrix_rotation(latitude, longitude):

    phi = np.asarray(latitude) * deg2rad
    lambd = np.asarray(longitude) * deg2rad

    return np.array(
        [
            [-np.sin(lambd), -np.sin(phi) * np.cos(lambd), np.cos(phi) * np.cos(lambd)],
            [np.cos(lambd), -np.sin(phi) * np.sin(lambd), np.cos(phi) * np.sin(lambd)],
            [0 * lambd, np.cos(phi), np.sin(phi)],
        ],
        dtype=float,
    )


def lla2ecef(latitude, longitude, h):

    phi = np.asarray(latitude) * deg2rad
    lambd = np.asarray(longitude) * deg2rad

    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)

    X = (N + h) * np.cos(phi) * np.cos(lambd)
    Y = (N + h) * np.cos(phi) * np.sin(lambd)
    Z = (N * (1 - e2) + h) * np.sin(phi)

    return X, Y, Z


def lla2ecef_radius(latitude, longitude, h):

    x, y, z = lla2ecef(latitude, longitude, h)

    return np.sqrt(x**2 + y**2 + z**2)


def ecef2lla(X, Y, Z):

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    p2 = X**2 + Y**2
    r2 = p2 + Z**2
    p = np.sqrt(p2)
    r = np.sqrt(r2)

    tanu = (b / a) * (Z / p) * (1 + epsilon * b / r)
    tan2u = tanu * tanu

    cos2u = 1.0 / (1.0 + tan2u)
    cosu = np.sqrt(cos2u)
    cos3u = cos2u * cosu

    sinu = tanu * cosu
    sin2u = 1.0 - cos2u
    sin3u = sin2u * sinu

    tanlat = (Z + epsilon * b * sin3u) / (p - e2 * a * cos3u)
    tan2lat = tanlat * tanlat
    cos2lat = 1.0 / (1.0 + tan2lat)
    sin2lat = 1.0 - cos2lat

    coslat = np.sqrt(cos2lat)
    sinlat = tanlat * coslat

    lambd = np.arctan2(Y, X)
    phi = np.arctan(tanlat)

    h = p * coslat + Z * sinlat - a * np.sqrt(1.0 - e2 * sin2lat)

    return phi * rad2deg, lambd * rad2deg, h


def enu2ecef(x, y, z, latitude, longitude, h=0.0):

    R = get_matrix_rotation(latitude, longitude)
    Xr, Yr, Zr = lla2ecef(latitude, longitude, h)

    X = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + Xr
    Y = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + Yr
    Z = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + Zr

    return X, Y, Z


def ecef2enu(X, Y, Z, latitude, longitude, h=0.0):

    R = get_matrix_rotation(latitude, longitude)
    Xr, Yr, Zr = lla2ecef(latitude, longitude, h)

    Xp = X - Xr
    Yp = Y - Yr
    Zp = Z - Zr

    x = R[0, 0] * Xp + R[1, 0] * Yp + R[2, 0] * Zp
    y = R[0, 1] * Xp + R[1, 1] * Yp + R[2, 1] * Zp
    z = R[0, 2] * Xp + R[1, 2] * Yp + R[2, 2] * Zp

    return x, y, z


def lla2enu(lat, lon, alt, lat_ref, lon_ref, alt_ref, units="km"):

    X, Y, Z = lla2ecef(lat, lon, alt)
    x, y, z = ecef2enu(X, Y, Z, lat_ref, lon_ref, alt_ref)

    if units == "m":
        return 1e3 * x, 1e3 * y, 1e3 * z

    return x, y, z


def enu2lla(x, y, z, lat_ref, lon_ref, alt_ref):

    X, Y, Z = enu2ecef(x, y, z, lat_ref, lon_ref, alt_ref)

    return ecef2lla(X, Y, Z)


def lla2xyh(lats, lons, alts, lat_center, lon_center, alt_center, units="m"):

    x, y, _ = lla2enu(lats, lons, alts, lat_center, lon_center, alt_center, units=units)
    h = np.asarray(alts)

    if units == "m":
        h = h * 1e3

    return x, y, h


def xyh2lla(x, y, h, lat_ref, lon_ref, alt_ref):

    h0 = np.asarray(h) - alt_ref
    R = lla2ecef_radius(lat_ref, lon_ref, alt_ref)

    d = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
    angle = d / (R + h0)
    z = (R + h0) * np.cos(angle) - R

    return enu2lla(x, y, z, lat_ref, lon_ref, alt_ref)
