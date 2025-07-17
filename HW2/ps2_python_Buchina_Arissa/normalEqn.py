import numpy as np

#3- Normal equation 
def normalEqn(X_train, y_train):
    
    #transpose X
    X_t = np.transpose(X_train)
    
    #inverse (or equiv)
    inverse = np.linalg.pinv(np.dot(X_t, X_train))
    
    #calculate theta
    theta = np.dot(np.dot(inverse, X_t), y_train)
    
    return theta

#X_train = np.array([ [1, 0, 1], [1, 1, 1.5], [1, 2, 4], [1, 3, 2] ])
#y_train = np.array([ [1.5], [4], [8.5], [8.5] ])

#theta = normalEqn(X_train, y_train)
#print('theta :', theta)