import numpy as np
import pandas as pd


alpha = 0.01
MaxIt = 1500


data = pd.read_csv('data.csv')
X = data['population'].values
y = data['profit'].values


Theta = np.zeros(2)

#gradient descent
def fit(X, y, theta, alpha, MaxIt):
    m = len(y) 
    for i in range(MaxIt):
        h0 = X.dot(theta)  
        Err = h0 - y
        grad = (alpha / m) * (X.T.dot(Err)) 
        theta -= grad

    return theta


X = np.column_stack((np.ones(X.shape[0]), X))

Theta = fit(X, y, Theta, alpha, MaxIt)

print("Theta:", Theta)
ypred = X.dot(Theta)
MeanSqEr = np.mean((ypred - y) ** 2)
print("Mean Squared Error:", MeanSqEr)
