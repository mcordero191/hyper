import h5py
import pandas as pd

def read_H5File(filename, duration=24*60*60, filter=True):
    
    with h5py.File(filename) as fp:
        
        g = fp['Winds']
        
        u0 = g['U'][0]
        u1 = g['U_SMOOTH'][0]
        u2 = g['U_DIFF'][0]
        u3 = g['U_LOWPASS'][0]
        u4 = g['U_FILTERED'][0]
        
        epoch   = g.attrs['PARAMETERS']['TIME'][0]
        alt     = g.attrs['PARAMETERS']['HEI'][0]
        
    if duration is not None:
        t_valid = epoch < (epoch[0] + duration)
    else:
        t_valid = epoch < epoch[-1]+1
    
    # t = pd.to_datetime(epoch, unit='s')
    # df = pd.DataFrame(data = t, columns=['time'])
    df = {}
    
    df['epoch'] = epoch[t_valid]
    df['alt'] = alt
    df['u0'] = u0[0][:,t_valid]
    df['v0'] = u0[1][:,t_valid]
    df['w0'] = u0[2][:,t_valid]
    
    df['u4h'] = u1[0][:,t_valid]
    df['v4h'] = u1[1][:,t_valid]
    df['w4h'] = u1[2][:,t_valid]
        
    return(df)