import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import  mean_squared_error,r2_score

#load data
diabetes = datasets.load_diabetes()

#use only one feature
diabetes_x = diabetes.data[:,np.newaxis,2]

#split the data into two training/test sets
diabetes_x_train = diabetes_x[:-20]
diabetes_x_test = diabetes_x[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_x_test)
#the conefficients
print(regr.coef_)
#the mean square error
print("Mean square error: %.2f"% mean_squared_error(diabetes_y_test,diabetes_y_pred))
print("Variance score:"% r2_score(diabetes_y_test,diabetes_y_pred))
plt.scatter(diabetes_x_test,diabetes_y_test,color='black')
plt.plot(diabetes_x_test,diabetes_y_pred,color='blue',linewidth=3)
plt.show()


