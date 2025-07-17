import numpy as np
from sigmoid import sigmoid

#1e) cost function 
def costFunction(theta, X_train, y_train):
    
    theta = theta.reshape(-1, 1)
    #number of sets
    m = len(y_train)
    
    #dot product x and theta
    theta_x = np.dot(X_train, theta)
    
    #calcultaing each h_theta(x)
    #sigmoid with theta^T * X
    g = sigmoid(theta_x)
    
    #splitting calcs in sum to two parts for easier reading 
    part_1 = y_train * np.log(g)
    part_2 = (1 - y_train) * np.log(1 - g)
    
    #summation of adding part1 and part2
    sum = np.sum(part_1 + part_2)
    
    #final cost calc with multiplying sum by (-1/m), m = # of sets 
    cost = -(1/m)  * sum
    
    return cost  

"""X_train = np.array( ([1, 1, 0], [1, 1, 3], [1, 3, 1], [1, 3, 4]) )
y_train = np.array( ([0], [1], [0], [1]) )  
theta = np.array( ([1], [.5], [.2]) )
cost = costFunction(theta, X_train, y_train)
print('cost = ', cost)"""
    