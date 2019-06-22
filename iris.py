import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset=load_iris()
X=dataset.data
y=dataset.target

plt.scatter(X[y==0,0],X[y==0,1],c='r',label='setosa')
plt.scatter(X[y==1,0],X[y==1,1],c='g',label='Versicolour')
plt.scatter(X[y==2,0],X[y==2,1],c='b',label='Virginica')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.grid()
plt.title('analysis on the dataset')
plt.show()

plt.scatter(X[y==0,2],X[y==0,3],c='r',label='setosa')
plt.scatter(X[y==1,2],X[y==1,3],c='g',label='Versicolour')
plt.scatter(X[y==2,2],X[y==2,3],c='b',label='Virginica')

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.grid()
plt.title('analysis on the dataset')
plt.show()

plt.scatter(X[y==0,0],X[y==0,2],c='r',label='setosa')
plt.scatter(X[y==1,0],X[y==1,2],c='g',label='Versicolour')
plt.scatter(X[y==2,0],X[y==2,2],c='b',label='Virginica')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend()
plt.grid()
plt.title('analysis on the dataset')
plt.show()

plt.scatter(X[y==0,1],X[y==0,2],c='r',label='setosa')
plt.scatter(X[y==1,1],X[y==1,2],c='g',label='Versicolour')
plt.scatter(X[y==2,1],X[y==2,2],c='b',label='Virginica')

plt.xlabel('sepal width')
plt.ylabel('petal length')
plt.legend()
plt.grid()
plt.title('analysis on the dataset')
plt.show()