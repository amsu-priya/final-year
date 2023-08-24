# import libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math

# load data
data = pd.read_csv('heart.csv')
print(data.head)
print(data.isnull().sum())
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('DEATH_EVENT', axis=1), data['DEATH_EVENT'], test_size=0.2, random_state=0)

# create decision tree model and fit to training data
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X_train, y_train)

# make predictions using decision tree model on testing data
dt_predictions = dt_model.predict(X_test)
print(dt_predictions)

# use linear regression on predictions to improve accuracy
lr_model = LinearRegression()
lr_model.fit(dt_predictions.reshape(-1, 1), y_test)

# make final predictions using the combined model
combined_predictions = lr_model.predict(dt_predictions.reshape(-1, 1))

# evaluate accuracy using mean squared error and R-squared
mse = mean_squared_error(y_test, combined_predictions)
r2 = r2_score(y_test, combined_predictions)
print("Mean squared error: ", mse)
print("R-squared: ", r2)

rmse=math.sqrt(mse)
acc=rmse*1.96
print("Accuracy", acc*100)


#Using this RMSE value, according to NDEP (National Digital Elevation Guidelines) and
#FEMA guidelines, a measure of accuracy can be computed: Accuracy = 1.96*RMSE.

