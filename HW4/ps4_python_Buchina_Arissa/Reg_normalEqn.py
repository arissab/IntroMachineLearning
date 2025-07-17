import numpy as np

def Reg_normalEqn(X_train, y_train, lam):
    
    #getting size of X_train 
    sets, features = X_train.shape
    
    #transpose X
    X_t = np.transpose(X_train)
    
    #create identity matrix
    D = np.identity(features)
    #set bias iden to 0
    D[0, 0] = 0
    
    if lam == 0:
        theta = np.linalg.pinv(X_t @ X_train) @ (X_t @ y_train)
        print('lam zeroa')
    else:
        theta = np.linalg.pinv( (X_t @ X_train) + (lam * D) ) @ (X_t @ y_train)

    
    return theta