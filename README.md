# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries. \n
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MARXIN LIJO M 
RegisterNumber: 212223240085


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```
## Output:
```
df.head()
```
![image](https://github.com/user-attachments/assets/575a15a2-a08d-41d9-9eb2-afa2c26cba01)
```
df.tail()
```
![image](https://github.com/user-attachments/assets/3466e5df-d03c-4ba1-ba08-b6ed9131089b)

Array value of X

![image](https://github.com/user-attachments/assets/18d5c572-a822-4a4d-bf3e-92f40c3e3b46)

Array value of Y

![image](https://github.com/user-attachments/assets/573d1290-71c8-4dc1-99f0-d5cd62acf903)

Values of Y prediction
![image](https://github.com/user-attachments/assets/ed9ac81c-9c4b-43e8-b86b-f5e4be911a7d)
Training Set Graph
![image](https://github.com/user-attachments/assets/762546c0-619a-4f5e-8801-bf3236a6a833)
Test Set Graph
![image](https://github.com/user-attachments/assets/2970370f-2503-4892-87f1-8b2ff01c1d3a)

Values of MSE, MAE and RMSE

![image](https://github.com/user-attachments/assets/7679dd2b-c216-4294-9928-59edb6a6e34b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
