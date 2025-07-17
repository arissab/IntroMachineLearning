import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from Reg_normalEqn import Reg_normalEqn
from computeCost import computeCost
from logReg_multi import logReg_multi

#reading in all of the matlab files, into their own column arrays

#data1 file read in 
data1 = scipy.io.loadmat('input/hw4_data1.mat')
X_data1 = np.array(data1["X_data"])
y_data1 = np.array(data1["y"])

#data2 file read in
data2 = scipy.io.loadmat('input/hw4_data2.mat')
X1_data2 = np.array(data2["X1"])
X2_data2 = np.array(data2["X2"])
X3_data2 = np.array(data2["X3"])
X4_data2 = np.array(data2["X4"])
X5_data2 = np.array(data2["X5"])
y1_data2 = np.array(data2["y1"])
y2_data2 = np.array(data2["y2"])
y3_data2 = np.array(data2["y3"])
y4_data2 = np.array(data2["y4"])
y5_data2 = np.array(data2["y5"])

#data3 file read in
data3 = scipy.io.loadmat('input/hw4_data3.mat')
X_test = np.array(data3["X_test"])
X_train = np.array(data3["X_train"])
y_test = np.array(data3["y_test"])
y_train = np.array(data3["y_train"])



### 1) Regularization 

##1b) adding bias feature
sets = len(X_data1)
x0 = np.ones((sets, 1))
#adding bias to feature matrix
X_data1 = np.append(x0, X_data1, axis = 1)

#getting the size of X and y 
size_X = X_data1.shape
print('size of the feature matrix X_data: ', size_X)



##1c) Compute avg error from 20 different models
#creating empty arrays to hold the errors for each iteration 
training_error = np.zeros((20, 7))
testing_error = np.zeros((20, 7))
# num of iterations 
models = 20

#array of lambda values
    #i removed 0 it since reg equation calls for l > 0, and the graph output was hard to see close up including it. testing error was to large for 0 for be a choice for lambda 
lam = [0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017]

#running 20 times 
for m in range(models):
    #randomly shuffling indices to split data
    rows = np.arange(len(X_data1))
    np.random.shuffle(rows)

    #finding where ~85% would be to split between train and test 
    split = round(len(X_data1) * .85)

    #picking the first ~85% of the shuffled indices for train, and the remaining 2 for test 
    train_rows = rows[:split]
    test_rows = rows[split:]

    #pulling X_train and X_test
    X_train_1 = X_data1[train_rows]
    X_test_1 = X_data1[test_rows]

    #pulling y_train and y test
    y_train_1 = y_data1[train_rows]
    y_test_1 = y_data1[test_rows]
    
    #running with different lambda values 
    for l, lam_value in enumerate(lam):
        theta = Reg_normalEqn(X_train_1, y_train_1, lam[l])
        
        #computing cost for both training and testing data
        training_error[m, l] = computeCost(X_train_1, y_train_1, theta)
        testing_error[m, l] = computeCost(X_test_1, y_test_1, theta)

#finding the average error for both data sets
avg_training_error = np.mean(training_error, axis=0)
avg_testing_error = np.mean(testing_error, axis=0)

#plotting the lambda values vs errors for both data sets 
plt.clf()
plt.plot(lam, avg_training_error, label="training Error", marker='s', color='hotpink')
plt.plot(lam, avg_testing_error, label="testing Error", marker='o', color='purple')
plt.xlabel("Lambda")
plt.ylabel("Average Error")
plt.legend()
plt.show() 

   
        

### 2) KNN - The Effect of K

#first classifier
X_train_cl1 = np.concatenate([X1_data2, X2_data2, X3_data2, X4_data2], axis=0)
y_train_cl1 = np.concatenate([y1_data2, y2_data2, y3_data2, y4_data2], axis=0)
X_test_cl1 = X5_data2
y_test_cl1 = y5_data2

#second classifier
X_train_cl2 = np.concatenate([X1_data2, X2_data2, X3_data2, X5_data2], axis=0)
y_train_cl2 = np.concatenate([y1_data2, y2_data2, y3_data2, y5_data2], axis=0)
X_test_cl2 = X4_data2
y_test_cl2 = y4_data2

#third classifier
X_train_cl3 = np.concatenate([X1_data2, X2_data2, X4_data2, X5_data2], axis=0)
y_train_cl3 = np.concatenate([y1_data2, y2_data2, y4_data2, y5_data2], axis=0)
X_test_cl3 = X3_data2
y_test_cl3 = y3_data2

#fourth classifier
X_train_cl4 = np.concatenate([X1_data2, X3_data2, X4_data2, X5_data2], axis=0)
y_train_cl4 = np.concatenate([y1_data2, y3_data2, y4_data2, y5_data2], axis=0)
X_test_cl4 = X2_data2
y_test_cl4 = y2_data2

#fifth classifier
X_train_cl5 = np.concatenate([X2_data2, X3_data2, X4_data2, X5_data2], axis=0)
y_train_cl5 = np.concatenate([y2_data2, y3_data2, y4_data2, y5_data2], axis=0)
X_test_cl5 = X1_data2
y_test_cl5 = y1_data2

#creating an array of K value and saving the length 
K_values = np.arange(1, 16, 2)
k_len = len(K_values)

#hold the accuracy for each K value for each classifier 
accuracy = np.zeros((5, k_len))

#classifier 1 accuracy 
for k in range(len(K_values)):
    knn = KNeighborsClassifier(n_neighbors=K_values[k])
    knn.fit(X_train_cl1, y_train_cl1.reshape(-1))
    y_pred_cl1 = knn.predict(X_test_cl1)

    correct = 0
    samples = len(y_test_cl1)
    #counting how many labels are accurate
    for i in range(samples):
        if y_pred_cl1[i] == y_test_cl1[i]:
            correct +=1
    #saving the accuracy for the K value 
    accuracy[0, k] = (correct / samples) * 100

#classifier 2 accuracy 
for k in range(len(K_values)):
    knn = KNeighborsClassifier(n_neighbors=K_values[k])
    knn.fit(X_train_cl2, y_train_cl2.reshape(-1))
    y_pred_cl2 = knn.predict(X_test_cl2)

    correct = 0
    samples = len(y_test_cl2)
    #counting how many labels are accurate
    for i in range(samples):
        if y_pred_cl2[i] == y_test_cl2[i]:
            correct +=1
    #saving the accuracy for the K value 
    accuracy[1, k] = (correct / samples) * 100
    
#classifier 3 accuracy 
for k in range(len(K_values)):
    knn = KNeighborsClassifier(n_neighbors=K_values[k])
    knn.fit(X_train_cl3, y_train_cl3.reshape(-1))
    y_pred_cl3 = knn.predict(X_test_cl3)

    correct = 0
    samples = len(y_test_cl3)
    #counting how many labels are accurate
    for i in range(samples):
        if y_pred_cl3[i] == y_test_cl3[i]:
            correct +=1
    #saving the accuracy for the K value 
    accuracy[2, k] = (correct / samples) * 100
     
#classifier 4 accuracy 
for k in range(len(K_values)):
    knn = KNeighborsClassifier(n_neighbors=K_values[k])
    knn.fit(X_train_cl4, y_train_cl4.reshape(-1))
    y_pred_cl4 = knn.predict(X_test_cl4)

    correct = 0
    samples = len(y_test_cl4)
    #counting how many labels are accurate
    for i in range(samples):
        if y_pred_cl4[i] == y_test_cl4[i]:
            correct +=1
    #saving the accuracy for the K value 
    accuracy[3, k] = (correct / samples) * 100
    
#classifier 5 accuracy 
for k in range(len(K_values)):
    knn = KNeighborsClassifier(n_neighbors=K_values[k])
    knn.fit(X_train_cl5, y_train_cl5.reshape(-1))
    y_pred_cl5 = knn.predict(X_test_cl5)

    correct = 0
    samples = len(y_test_cl5)
    #counting how many labels are accurate
    for i in range(samples):
        if y_pred_cl5[i] == y_test_cl5[i]:
            correct +=1
    #saving the accuracy for the K value 
    accuracy[4, k] = (correct / samples) * 100

#calc the avg accurcay for each value of k
avg_accuracy = (np.sum(accuracy, axis=0)) / 5

#plotting the accuracy vs the K value
plt.clf()
plt.plot(K_values, avg_accuracy, color='purple')
plt.xlabel('K values')
plt.ylabel('Avg. Accuracy (%)')
plt.show()



### 3) one-vs-all

#running with training data to see accuracy 
#getting y_pred
y_predict_train = logReg_multi(X_train, y_train, X_train)
#var to track correct labels
correct = 0 
len_y_train = len(y_train)
for i in range(len_y_train):
    #if prediction matches truth, mark correct
    if y_train[i] == y_predict_train[i]:
        correct += 1
#calc accuracy, # correct / total number of labeled
acc = (correct / len_y_train) * 100
print('one-vs-all y_train accuracy: ', round(acc, 2), '%')

#running with testing data to see accuracy 
#getting y_pred
y_predict_test = logReg_multi(X_train, y_train, X_test)
#var to track correct labels
correct = 0 
len_y_test = len(y_test)
for i in range(len_y_test):
    #if prediction matches truth, mark correct
    if y_test[i] == y_predict_test[i]:
        correct += 1

#calc accuracy, # correct / total number of labeled
acc = (correct / len_y_test) * 100
print('one-vs-all y_test accuracy: ', round(acc, 2), '%') 