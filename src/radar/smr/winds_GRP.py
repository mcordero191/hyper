'''
Created on 19 Jan 2023

@author: mcordero
'''

import numpy as np
import xarray

class GRPData():
    
    def __init__(self, path):
        
        files = self.__find_files(path)
    
    
        self.files = files
        self.ifile = 0
        
    def __find_files(self):
        
        pass
    
    def read_data(self, filename):
        
        xarray.
    
    def read_next_block(self):
        
        self.ifile += 1
        self.read_data(self.files[self.ifile])
    
if __name__ == '__main__':
    
    path = ''
    
    fp = GRPData(path)
    
    winds = fp.read_next_block()