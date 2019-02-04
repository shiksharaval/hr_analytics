# -*- coding: utf-8 -*-


import pandas as pd
from sklearn import metrics
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.naive_bayes as NB
from sklearn.metrics import roc_auc_score

'''
dataSet2 = pd.read_csv("HR_comma_sep.csv")
dataSet2["salary"] = dataSet2["salary"].astype('category').cat.codes
X = dataSet2.drop('left', axis=1)    #data with only features
Y=dataSet2['left']
dt = tree.DecisionTreeClassifier()                                                              
#//replace this for different machine learning models
dt = dt.fit(X, Y)                                                                                            
#//replace this for different machine learning models
kfold = ms.StratifiedKFold(n_splits=10)
predCV = ms.cross_val_predict(dt, X, Y, cv=kfold)                                
#//replace 'dt' with corresponding machine learning models

precisionVal = metrics.precision_score(Y,predCV)
recallVal = metrics.recall_score(Y,predCV)
f1Val = metrics.f1_score(Y,predCV)
KappaVal = metrics.cohen_kappa_score(Y, predCV)
Accuracy= metrics.accuracy_score(Y,predCV )
            #print all the above values
'''

#read the dataset
datSet = pd.read_csv("HR_comma_sep.csv")

df = datSet.copy()
#change string values to numerical
df["salary"] = df["salary"].astype('category').cat.codes
df['sales'] = df['sales'].astype('category').cat.codes

#X and y values
X = df.drop('left', axis=1)
y = df['left']

print(X.shape)
print(y.shape)


#Decision tree classifier
print("Decision Tree Classifier")
model = DecisionTreeClassifier()
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X,y)                                      #change to X,y
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("Decision Tree Classifier = ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))    #change to X,y
print("\n\n")




#random forest
print("Random forest Classifier")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("Random Forest = ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")


#knn
print("KNN")
model = KNeighborsClassifier(n_neighbors=3)
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("KNN= ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")


#Logistic Regression
print("Logistic Regression")
model = LogisticRegression()
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("Logistic Regression = ", rf_roc_auc)
#print(metrics.classification_report(y_test,model.predict(X_test)))
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")


#PLease repeat the fit for these classifiers
#ada = AdaBoostClassifier(n_estimators=400)
print("ADA classifier")
model = ada = AdaBoostClassifier(n_estimators=400)
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("ADA Classifier = ", rf_roc_auc)
#print(metrics.classification_report(y_test,model.predict(X_test)))
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")




#naive bayes classifier
print("Naive bayes")
model = NB.GaussianNB()
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
#print(metrics.classification_report(y_test,model.predict(X_test)))
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("Naive Bayes Classifier = ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")
