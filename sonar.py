#importing the dependancies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#loading dataset
sonar_data=pd.read_csv('Copy of sonar data.csv', header=None)

print(sonar_data[60].value_counts())

#we get mean values of each column for both mine & rock
print(sonar_data.groupby(60).mean())  

#separating data and label

X=sonar_data.drop(columns=60,axis=1)
y=sonar_data[60]

print(X)
print(y)

#training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y, random_state=1) 

print(X.shape, X_train.shape, X_test.shape)

#model training
model=LogisticRegression()

#training model with training data
model.fit(X_train,y_train)

#model accuracy
X_test_predictions=model.predict(X_test)
acc=accuracy_score(X_test_predictions,y_test)

print('Accuracy on test data: ', acc)

#making a predictive system
input_data=(0.0519,0.0548,0.0842,0.0319,0.1158,0.0922,0.1027,0.0613,0.1465,0.2838,0.2802,0.3086,0.2657,0.3801,0.5626,0.4376,0.2617,0.1199,0.6676,0.9402,0.7832,0.5352,0.6809,0.9174,0.7613,0.8220,0.8872,0.6091,0.2967,0.1103,0.1318,0.0624,0.0990,0.4006,0.3666,0.1050,0.1915,0.3930,0.4288,0.2546,0.1151,0.2196,0.1879,0.1437,0.2146,0.2360,0.1125,0.0254,0.0285,0.0178,0.0052,0.0081,0.0120,0.0045,0.0121,0.0097,0.0085,0.0047,0.0048,0.0053)

#changing input data to numpy array
data=np.asarray(input_data)

#reshape numpy array as we are predicting one instance
data_reshape=data.reshape(1,-1)

prediction=model.predict(data_reshape) #returns r or m

print(prediction)

if(prediction[0]=='R'):
	print("The object is a Rock")
else:
	print("The object is a Mine")
