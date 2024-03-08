## Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: POZHILAN V D
RegisterNumber:  212223240118
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")

```
## Output:
![1](https://github.com/POZHILANVD/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870498/4dd0e997-6f45-43c7-bcf9-6dd9f97aec8a)

![2](https://github.com/POZHILANVD/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870498/bc40bc45-454f-4fbd-908f-0337cb95beb2)
![3](https://github.com/POZHILANVD/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870498/20a1d598-778c-4ea8-96a0-78a46f6b654e)
![4](https://github.com/POZHILANVD/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870498/25d3eccd-e27c-407c-9895-f366418cfb24)
![5](https://github.com/POZHILANVD/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870498/a7bf3690-87d3-4382-beba-a65456ed35aa)
![last](https://github.com/POZHILANVD/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870498/59aa7d0c-69f7-45c5-9dd9-d62aec3f310c)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
