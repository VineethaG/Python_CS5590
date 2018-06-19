import matplotlib.pyplot as p
import numpy as np
from sklearn import datasets, linear_model

def call_model(x,y):
    lin_regr=linear_model.LinearRegression()
    lin_regr.fit(x,y)
    p.scatter(x, y, color='black')
    p.plot(x,lin_regr.predict(x),color='blue',linewidth=3)
    p.show()

def predict_value(x,y,z):
    lin_regr=linear_model.LinearRegression()
    lin_regr.fit(x,y)
    predicted_value = lin_regr.predict(z)
    return predicted_value,lin_regr.coef_,lin_regr.intercept_

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X
diabetes_y_train = diabetes.target
call_model(diabetes_X_train,diabetes_y_train)

print("predicted values are:")
(predicted_value, coeff, cons) = predict_value(diabetes_X_train, diabetes_y_train, -0.02265432)
print("If The diabetes for -0.02265432,then BP will be : $", str(predicted_value))
print("The regression coefficient is ", str(coeff), ", and the constant is ", str(cons))
print("the relationship equation between diabetes and BP is:  = ", str(coeff), "* date + ", str(cons))
