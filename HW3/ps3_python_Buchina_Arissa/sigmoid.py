import numpy as np

#1d sigmoid functions 
def sigmoid( z ):
    #e^-z
    ex = np.exp(np.negative(z))
    
    #computing sigmoid
    g = 1 / (1 + ex)
    
    return g
    