# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ARCHANA S
RegisterNumber: 212223040019 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv("C:/Users/ANANDAN S/Documents/ML labs/student_scores.csv")
df.head()

#splitting training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color = "orange")
plt.plot(x_train,regressor.predict(x_train),color= "red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_train,y_train,color = "purple")
plt.plot(x_test,regressor.predict(x_test),color= "yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(y_test,y_pred)
print('MSE= ',mse)
mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
```
y_pred: array([17.04289179, 33.51695377, 74.21757747, 26.73351648, 59.68164043,
       39.33132858, 20.91914167, 78.09382734, 69.37226512])
y_test: array([20, 27, 69, 30, 62, 35, 24, 86, 76], dtype=int64)
```
![Screenshot 2024-08-29 094713](https://github.com/user-attachments/assets/69cbf5d4-6616-4ca5-9455-c407f1f40593)
```
MSE=  25.463280738222593
MAE =  4.6913974413974415
RMSE =  5.046115410711748

```


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
