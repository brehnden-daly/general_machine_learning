from numba import cuda
import math




# Accelerated cuda functions


@cuda.jit
def normalize_1D(data, min_val, max_val, scale_min, scale_max, normalized):
    x = cuda.grid(1)
    
    if (x < data.shape[0]):
        norm = ( data[x] - min_val ) / (max_val - min_val)
        normalized[x] = scale_min + ( norm*(scale_max-scale_min) )
        
        
        
        
        
@cuda.jit
def normalize_2D(data, min_val, max_val, scale_min, scale_max, normalized):
    x, y = cuda.grid(2)
    
    if (x < data.shape[0]) & (y < data.shape[1]):
        norm = ( data[x,y] - min_val ) / (max_val - min_val)
        normalized[x,y] = scale_min + ( norm*(scale_max-scale_min) )




@cuda.jit
def normalize_3D(data, min_val, max_val, scale_min, scale_max, normalized):
    x, y, z = cuda.grid(3)
    
    if (x < data.shape[0]) & (y < data.shape[1]) & (z < data.shape[2]):
        norm = ( data[x,y,z] - min_val ) / (max_val - min_val)
        normalized[x,y,z] = scale_min + ( norm*(scale_max-scale_min) )













