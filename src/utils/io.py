'''
Created on 21 Feb 2024

@author: mcordero
'''
import h5py

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