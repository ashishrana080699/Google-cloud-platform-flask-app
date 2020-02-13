import numpy as np
import pandas as pd
from sklearn.model_selection import  RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("C:/Users/Dell/Documents/diabetes.csv", header=None, names=col_names)
pima #print dataset

# change datatype to nd array float
data = pima.tail(768).astype(np.float)
#correlation
data.corr()

#split dataset in features and target variable
feature_cols = [ 'bmi', 'bp','glucose','insulin']
X = pima[feature_cols][1:] # Features
y = pima.label[1:]# Target variable


skf = RepeatedKFold(n_splits=100,n_repeats=7,random_state=10)
for train_index, test_index in skf.split(X,y):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]
print(X_test.shape,y_test.shape)  # (7,4) (7,)
print(X_train.shape,y_train.shape) # (761,4) (761,)

# instantiate the model (using the default parameters)
kn = KNeighborsClassifier()
# fit the model with data
kn.fit(X_train,y_train)

#Prediction
y_pred = kn.predict(X_test)
y_pred # print y_pred

#calculate Accuracy
#print("Accuracy:",metrics.accuracy_score(test, test_predict))
#calculate Precision
#print("Precision:",metrics.precision_score(test.astype(int), test_predict.astype(int)))
#calculate Recall
#print("Recall:",metrics.recall_score(test.astype(int), test_predict.astype(int)))


#import joblib
from sklearn.externals import joblib 
#save model as diabetes.pkl
joblib.dump(kn,'C:/Users/Dell/Documents/diabetes.pkl',protocol=2) 
#loading the model
model1 = joblib.load('C:/Users/Dell/Documents/diabetes.pkl')
#predicting
model1.predict(X_test)
