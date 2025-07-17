from computeCost import computeCost
import numpy as np

#2- Gradient descent 
def gradientDescent(X_train, y_train, alpha, iters):
       
    #find size of X_train to know how many features 
    size = X_train.shape
    sets, features = size
    
    #creating the first instance of theta randomly, one for each feature 
    theta = np.random.normal(size= features).reshape(-1, 1)
    
    #empty array to hold cost values of each iteration 
    cost = np.array([0.0] * iters)
    
    #for loop for number of iterations
    for i in range(iters):
        
        #setting up summation  
        ht_x = np.dot(X_train, theta)
        diff = ht_x - y_train
        
        for j in range(features):
            #calculating and replacing 
            theta[j] = theta[j] - (alpha * (1/sets) * np.sum(diff * X_train[:,j].reshape(-1, 1)))
        
        #computing cost for new theta values     
        cost[i] = computeCost(X_train, y_train, theta)
        
    #returning theta array and cost     
    return [theta, cost]

#toy example data                            
#X_train = np.array([ [1, 0, 1], [1, 1, 1.5], [1, 2, 4], [1, 3, 2] ])
#y_train = np.array([ [1.5], [4], [8.5], [8.5] ])
#alpha = 0.001
#iters = 15

#theta, cost = gradientDescent(X_train, y_train, alpha, iters)
#print('theta: ', theta)
#print('cost: ', cost)
