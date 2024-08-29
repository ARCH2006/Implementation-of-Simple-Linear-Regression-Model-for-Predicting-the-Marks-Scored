# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. import the necessary libraries and read the file student scores
2.print the x and y values
3.separate the independent values and dependent values
4.split the data
5.create a regression model
6.find mse,mae and rmse and predicted value,then print the values
```
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
df.tail()
x=df.iloc[:,:-1].values
x

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
head:
	Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
tail:
	Hours	Scores
20	2.7	30
21	4.8	54
22	3.8	35
23	6.9	76
24	7.8	86
iloc:
array([[2.5],
       [5.1],
       [3.2],
       [8.5],
       [3.5],
       [1.5],
       [9.2],
       [5.5],
       [8.3],
       [2.7],
       [7.7],
       [5.9],
       [4.5],
       [3.3],
       [1.1],
       [8.9],
       [2.5],
       [1.9],
       [6.1],
       [7.4],
       [2.7],
       [4.8],
       [3.8],
       [6.9],
       [7.8]])
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
