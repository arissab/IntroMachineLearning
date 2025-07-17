import numpy as np
from sigmoid import sigmoid

#1e) grad of cost function 
def gradFunction(theta, X_train, y_train):
    
    #sets of training examples
    m = len(y_train)
    #input z for sigmoid function 
    theta = theta.reshape(-1, 1)
    
    x_theta = np.dot(X_train, theta)
    #compute h(x)/sigmoid 
    g = sigmoid(x_theta) 
    
    #compute grad of cost 
    grad = (1/m) * ( np.dot(X_train.T, (g - y_train)) ) 
    

    return grad.flatten()
    
"""X_train = np.array( ([1, 1, 0], [1, 1, 3], [1, 3, 1], [1, 3, 4]) )
y_train = np.array( ([0], [1], [0], [1]) )  
theta = np.array( ([1], [.5], [.2]) )
grad = gradFunction(theta, X_train, y_train)
print('grad = ', grad)"""