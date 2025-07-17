from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn
import numpy as np
import matplotlib.pyplot as plt


#4- Linear Regression with 1 variable 

#4a) load in data from data1
#get data from .csv file
data1 = np.genfromtxt("input/hw2_data1.csv", delimiter=',')

#split data into appropriate columns/variables 
power = np.array(data1[:, 0]).reshape(-1, 1)
prices = np.array(data1[:, 1]).reshape(-1, 1)




#4b plot the data
plt.clf()
plt.scatter(power, prices, marker='x' )
plt.xlabel("Horse power of a car in 100s")
plt.ylabel("Price in $1,000s")
plt.show()




#4c defining matrix X and y, getting sizes 
#creating bias/x0 column
sets = len(power)
x0 = np.ones((sets, 1))

#adding x0 to features from data to make X matrix
X = np.append(x0, power, axis=1)
y = prices

#finding size of X and y 
size_X = X.shape
size_y = y.shape
print('size of matrix X = ', size_X, ' and size of y = ', size_y)




#4d) randomly splitting X and y to be X_train and y_train 
#shuffling indices
rows = np.arange(len(X))
np.random.shuffle(rows)

#finding where ~90% would be to split between train and test 
split = round(len(X) * .9)

#picking the first 90% of the shuffled indices for train, and the remaining 2 for test 
train_rows = rows[:split]
test_rows = rows[split:]

#pulling X_train and X_test
X_train = X[train_rows]
X_test = X[test_rows]

#pulling y_train and y test
y_train = y[train_rows]
y_test = y[test_rows]




#4e) Compute gradient descent, plot iterations vs cost  
alpha = .3
iters = 500
theta, cost = gradientDescent(X_train, y_train, alpha, iters)

print('Gradient Descent theta = ', theta)

#array to plot iterations
iter_array = np.arange(1, 501)

#clear previous plots 
plt.clf()
#plotting iteration vs cost 
plt.plot(iter_array, cost)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()




#4f) create line of learned model to plot against the scatter plot  

#finding min and max values to use for predictions
x_min = min(power)
x_max = max(power)

#creating 2 points of prediction to plot the line 
y_min = theta[0] + (theta[1] * x_min)
y_max = theta[0] + (theta[1] * x_max)

#clear previous plots
plt.clf()
#plot test data
plt.scatter(power, prices, marker='x' )
#plot prediction line 
plt.plot([x_min, x_max], [y_min, y_max], color='red')
plt.xlabel("Horse power of a car in 100s")
plt.ylabel("Price in $1,000s")
plt.show()





#4g) use X_test to get y_pred, find cost 

pred_cost = computeCost(X_test, y_test, theta)
print('Gradient Descent prediction error: ', pred_cost)





#4h) use Normal Equation with training 
theta_normal = normalEqn(X_train, y_train)
print('Normal Eq theta: ', theta_normal)

normal_cost = computeCost(X_test, y_test, theta_normal)
print('Normal Eq prediction error:', normal_cost)




#4i) using different alpha values in gradient descent, plot and compare
alpha = [0.001, 0.003, 0.03, 3]
iters = 300
iter_array = np.arange(1, 301)

#make a matrix to hold the different all 4 theta vectors to make and plot in a for loop 
for a in range(len(alpha)):
    theta_temp, cost = gradientDescent(X_train, y_train, alpha[a], iters)
    print(f'theta with alpha = {alpha[a]} : ', theta_temp)
    
    #clear previous plots 
    plt.clf()
    #plotting iteration vs cost 
    plt.plot(iter_array, cost, label=f'alpha = {alpha[a]}')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()




#5 - Linear Regression with multiple variables 

#5a) load in data from data3, standardize features, size of X and y   
#load in data3 
#get data from .csv file
data3 = np.genfromtxt("input/hw2_data3.csv", delimiter=',')

#split data into appropriate columns/variables 
engine_size = np.array(data3[:, 0]).reshape(-1, 1)
weight = np.array(data3[:, 1]).reshape(-1, 1)
    #did not need to load engine and weight into different vectors
emissions = np.array(data3[:, 2]).reshape(-1, 1)

#creating X matrix with feature vectors
X_d3 = np.append(engine_size, weight, axis=1)

#standardizing X 
X_mean = np.mean(X_d3, axis=0)
X_std = np.std(X_d3, axis=0)
X_standardized = (X_d3 - X_mean) / X_std
print('mean: ', X_mean)
print('standard deviation: ', X_std)

#creating bias/x0 column
sets = len(engine_size)
x03 = np.ones((sets, 1))

#appending bias column
X_d3 = np.append(x03, X_standardized, axis=1)

#setting y 
y_d3 = emissions

#size of X and y
size_X_d3 = X_d3.shape
size_y_d3 = y_d3.shape
print('size of X is: ', size_X_d3)
print('size of y is: ', size_y_d3)




#5b) gradient descent, plot iters vs cost 
alpha_d3 = 0.01
iters_d3 = 750

theta_d3, cost_d3 = gradientDescent(X_d3, y_d3, alpha_d3, iters_d3)

print('theta =', theta_d3)
print('cost value: ', cost_d3[-1])

#array to plot iters
iter_array_d3 = np.arange(1, 751)

#clear previous plots 
plt.clf()
#plotting iteration vs cost 
plt.plot(iter_array_d3, cost_d3)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()



#5c) Predict C02 emissions/y 
engine_size_test = 2100
weight_test = 1200

#standardizing features
engine_size_test = (engine_size_test - X_mean[0]) / X_std[0]
weight_test = (weight_test - X_mean[1]) / X_std[1]

#calculating prediction with theta values found from gradient descent
emissions_predict = theta_d3[0] + (theta_d3[1] * engine_size_test) + (theta_d3[2] * weight_test)
print('Co2 emission prediction: ', emissions_predict)

