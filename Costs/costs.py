from enum import Enum
import numpy as np




class Costs(Enum):
    def mse(y_true, y_pred):
        mse = np.mean(np.power(y_true-y_pred, 2))
        return mse
    
    def mse_prime(y_true, y_pred):
        mse_prime = 2*np.mean(y_pred-y_true)
        return mse_prime
        # return 2*(y_pred-y_true)/y_true.size