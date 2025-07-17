import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

from sigmoid import sigmoid
from costFunction import costFunction
from gradFunction import gradFunction
from normalEqn import normalEqn


##1a) Load in data from hw3 file 
#loading in data from input file
data = np.loadtxt("input/hw3_data1.txt", delimiter = ',')

#splitting data into feature/label columns
exam1 = np.array(data[:, 0]).reshape(-1, 1)
exam2 = np.array(data[:, 1]).reshape(-1, 1)
admit = np.array(data[:, 2]).reshape(-1, 1)

#appending exam 1 and exam 2 to make feature matrix X
X = np.append(exam1, exam2, axis=1)

#creating bias column length of data
sets = len(exam1)
x0 = np.ones((sets, 1))

#adding bias to feature matrix
X = np.append(x0, X, axis = 1)

#getting the size of X and y 
size_X = X.shape
size_y = admit.shape

print('size of feature matrix X', size_X)
print('size of label vector y', size_y)




##1b) Plot the data 
#creating masks for admit 1/not admit 0 to label scatter plot with the different markers
admitted_mask = admit == 1  
not_admitted_mask = admit == 0  

#creating scatter plot in two differnt lines to use different colors/markers
plt.scatter(exam1[admitted_mask], exam2[admitted_mask], color="hotpink", marker="o", label="Admitted")
plt.scatter(exam1[not_admitted_mask], exam2[not_admitted_mask], color="red", marker="x", label="Not Admitted")
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
#legend to show the markers 
plt.legend()
plt.show()




##1c) divide data into training data and test data
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
y_train = admit[train_rows]
y_test = admit[test_rows]




##1d) sigmoid function 
# creating z to test 
z = np.arange(-15, 15.01, 0.01)
print('z', z)

gz = sigmoid(z)
print('gz: ', gz)

#clear plot 
plt.clf()
plt.plot(z, gz, color = 'hotpink')
plt.xlabel('z')
plt.ylabel('gz')
plt.show()




##1f) optimize
sets, features = X_train.shape
theta = np.zeros(features)

theta = fmin_bfgs(costFunction, theta, fprime=gradFunction, args=(X_train, y_train))
print('optimal theta: ', theta)

cost_conv = costFunction(theta, X_train, y_train)
print('conv cost: ', cost_conv)



##1g) plot decision boundary 
#possible exam scores 
x1 = np.linspace(0, 100, 100)

#finding values for x2 with range of exam scores and optimized thetas 
x2 = - (theta[0] + theta[1] * x1) / theta[2] 

#clear plot
plt.clf()
#setting limits on the plot so the line is not going to long 
plt.xlim([X_train[:, 1].min() - 5, X_train[:, 1].max() + 5])
plt.ylim([X_train[:, 2].min() - 5, X_train[:, 2].max() + 5])
#training data
plt.scatter(exam1[admitted_mask], exam2[admitted_mask], color="hotpink", marker="o", label="Admitted")
plt.scatter(exam1[not_admitted_mask], exam2[not_admitted_mask], color="red", marker="x", label="Not Admitted")
#decision boundary 
plt.plot(x1, x2, color = "green")
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
#legend to show the markers 
plt.legend()
plt.show()



##1h) use test data to find accuracy 
test_sets, test_features = X_test.shape
y_pred = np.zeros(test_sets)

#calculate g for each test point and setting label 
for x in range(test_sets):
       
    g = sigmoid(np.dot(theta.T, X_test[x,:] ))
    #admitted if > .5
    if g > .5:
        y_pred[x] = 1
    else:
        y_pred[x] = 0

#var to track correct labels
correct = 0

#checking each pred against test y
for y in range(test_sets):
    if y_pred[y] == y_test[y]:
        correct += 1

#finding accuracy  
accuracy = (correct/test_sets) * 100
print(accuracy, '%')


##1i) compute admission prob
#test point
xi = [1, 60, 65]
yi = 0

#compute g
gi = sigmoid(np.dot(theta.T, xi))

probi = gi * 100
print('admission probability: ', probi)




##2a) normal equation with data 2
data = np.loadtxt("input/hw3_data2.csv", delimiter = ',')

#splitting data into feature/label columns
n = np.array(data[:, 0]).reshape(-1, 1)
profit = np.array(data[:, 1]).reshape(-1, 1)

#size of data
sets, features = n.shape

#bias column
n0 = np.ones((sets, 1))

n2 = np.square(n)

#adding bias to feature matrix
n = np.append(n0, n, axis = 1)
#adding feature for non-linear, np^2
n = np.append(n, n2, axis = 1)

theta = normalEqn(n, profit)

print('non-linear regression theta: ', theta)




##2b) plot the data
#range of x values to find y_pred curve 
x_range = np.linspace(n[:,1].min(), n[:,1].max(), 100).reshape(-1, 1) 
#x^2 
x2_range = x_range ** 2  # Compute x2 for quadratic term

# calculating non-linear y_pred line 
y_pred = theta[0] + theta[1] * x_range + theta[2] * x2_range

#clear plot
plt.clf()
#fit curve
plt.plot(x_range, y_pred, color='blue')
#training data scatter 
plt.scatter(n[:,1], profit,color="hotpink", marker="x")
plt.xlabel('population in thousands')
plt.ylabel('profit')
plt.show()

