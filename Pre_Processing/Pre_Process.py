import sys
sys.path.append('../')

import math
import numpy as np
import pandas as pd
from numba import cuda
import sklearn.preprocessing as skpre

from Pre_Processing.cuda_functions import *



class Pre_Process_Inputs(object):
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.normalized_train_x = np.zeros( train_x.shape, dtype=np.float32 )
        self.train_y = train_y
        self.normalized_train_y = np.zeros( train_y.shape, dtype=np.float32 )
        
        self.test_x = test_x
        self.normalized_test_x = np.zeros( test_x.shape, dtype=np.float32 )
        self.test_y = test_y
        self.normalized_test_y = np.zeros( test_y.shape, dtype=np.float32 )
        
        self.normalize_scale = [-1,1]
        
    
    
    
    def normalize_inputs(self, train_or_test, min_value, max_value):
        if ( (train_or_test == "train") & (len(self.train_x.shape) < 3) ):
            threads_per_block = (64, 16)
            blocks_per_grid_x = math.ceil( self.train_x.shape[0] / threads_per_block[0] )
            blocks_per_grid_y = math.ceil( self.train_x.shape[1] / threads_per_block[1] )
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            dev_values = cuda.to_device( np.array(self.normalized_train_x, dtype=np.float32) )
            
            normalize_2D[blocks_per_grid, threads_per_block](self.train_x, min_value, max_value, 
                                                             self.normalize_scale[0], self.normalize_scale[1], dev_values)
            
            self.normalized_train_x = dev_values.copy_to_host()
            
        else:
            threads_per_block = (10, 10, 10)
            blocks_per_grid_x = math.ceil( self.train_x.shape[0] / threads_per_block[0] )
            blocks_per_grid_y = math.ceil( self.train_x.shape[1] / threads_per_block[1] )
            blocks_per_grid_z = math.ceil( self.train_x.shape[2] / threads_per_block[2] )
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
            
            dev_values = cuda.to_device( np.array(self.normalized_train_x, dtype=np.float32) )
            
            normalize_3D[blocks_per_grid, threads_per_block](self.train_x, min_value, max_value, 
                                                             self.normalize_scale[0], self.normalize_scale[1], dev_values)
            
            self.normalized_train_x = dev_values.copy_to_host()











