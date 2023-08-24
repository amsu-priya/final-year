import numpy as np
import matplotlib.pyplot as mtp  
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
  
#importing datasets  
data_set= pd.read_csv('heart.csv')  

#Extracting Independent and dependent Variable  
data_set.DEATH_EVENT.value_counts()

X=data_set.drop(["DEATH_EVENT"],axis=1)
y=data_set["DEATH_EVENT"]

 # Splitting the dataset into training and test set.

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)

print(y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean squared error: ", mse)
print("R-squared: ", r2)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)
print("Accuracy",acc*100)


