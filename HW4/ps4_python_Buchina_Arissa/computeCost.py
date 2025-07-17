import numpy as np

# Cost Function (from previous assignment)
def computeCost(X, y, theta):
    
    #taking in X, y, theta. Need m. 
    m = len(y)
    
    #ht(x) = dot product of X and theta 
    ht_x = np.dot(X, theta)
    
    #ht(x)-y
    diff = ht_x - y
    
    #(ht(x) - y)^2
    sqr = np.square(diff)
    
    #summation 
    sum = np.sum(sqr)
    
    #cost
    J = ((1/(2*m)) * sum)
    return J

#test data from previous assignment 
#1 Cost computation 
#X = [ [1, 0, 1], [1, 1, 1.5], [1, 2, 4], [1, 3, 2] ]
#y = [ [1.5], [4], [8.5], [8.5] ]
#theta = [ [.5], [2], [1] ]
#theta2 = [ [10], [-1], [-2] ]

#J = computeCost(X, y, theta)
# print('Cost with (i) theta is J = ', J)
#J = computeCost(X, y, theta2)
# print('Cost with (ii) theta is J = ', J)
    
    