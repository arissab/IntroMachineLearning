import numpy as np
from sklearn.linear_model import LogisticRegression

def logReg_multi(X_train, y_train, X_test):
    
    #getting size of X_train 
    r, c = X_test.shape 
    y_predict = np.zeros((r,1))
    
    #classifier 1 
    #get binary y_c for 1 vs not 1
    y_c1 = (y_train == 1).astype(int).ravel()
    #train model for classifier 1
    mdl_c1 = LogisticRegression(random_state=0).fit(X_train, y_c1)
    #prob of class for X_test to be class 1 
    proba_c1 = mdl_c1.predict_proba(X_test)
    prob_class1 = (proba_c1[:, 1] * 100).reshape(-1, 1)
    
    
    #classifier 2 
    #get binary y_c for 2 vs not 2
    y_c2 = (y_train == 2).astype(int).ravel()
    #train model for classifier 2
    mdl_c2 = LogisticRegression(random_state=0).fit(X_train, y_c2)
    #prob of class for X_test to be class 2
    proba_c2 = mdl_c2.predict_proba(X_test)
    prob_class2 = (proba_c2[:, 1] * 100).reshape(-1, 1)
    
    
    #classifier 3 
    #get binary y_c for 3 vs not 3
    y_c3 = (y_train == 3).astype(int).ravel()
    #train model for classifier 3
    mdl_c3 = LogisticRegression(random_state=0).fit(X_train, y_c3)
    #prob of class for X_test to be class 3
    proba_c3 = mdl_c3.predict_proba(X_test)
    prob_class3 = (proba_c3[:, 1] * 100).reshape(-1, 1)
    
    #make matrix of all three classification prob
    prob = np.append(prob_class1, prob_class2, axis=1)
    prob = np.append(prob, prob_class3, axis=1)
    
    #y_pred is labeled as the class with the highest prob (+1 to fix index)
    y_predict = np.argmax(prob, axis=1) + 1    
         
    return y_predict