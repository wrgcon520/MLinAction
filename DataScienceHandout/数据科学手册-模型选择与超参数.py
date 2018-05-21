
# coding: utf-8

# In[14]:


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
from sklearn.cross_validation import cross_val_score
cross_val_score(model,X,y,cv=5)


# In[15]:


from sklearn.cross_validation import LeaveOneOut
cross_val_score(model,X,y,cv=LeaveOneOut(len(X))).mean()


# In[16]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
def PolyNomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))
import numpy as np
def make_data(N,err=1.0,rseed=1):
    rng=np.random.RandomState(rseed)
    X = rng.rand(N,1)**2
    y = 10 - 1./(X.ravel()+0.1)
    if err>0:
        y += err * rng.randn(N)
    return X,y
X,y = make_data(40)


# In[33]:


import matplotlib.pyplot as plt
import seaborn
seaborn.set()
X_test = np.linspace(-0.1, 1.1, 500)[:,np.newaxis]
plt.scatter(X.ravel(),y,color='black')
axis = plt.axis()
for degree in [1,3,5]:
    y_test = PolyNomialRegression(degree).fit(X,y).predict(X_test)
    plt.plot(X_test,y_test,label='degree={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')
plt.show()


# In[38]:


from sklearn.learning_curve import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolyNomialRegression(),X,y,'polynomialfeatures__degree',degree,cv=7)
plt.plot(degree,np.median(train_score,1),color='blue',label='training score')
plt.plot(degree,np.median(val_score,1),color='red',label='validation score')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()


# In[43]:


plt.scatter(X.ravel(),y)
lim = plt.axis()
y_test = PolyNomialRegression(3).fit(X,y).predict(X_test)
plt.plot(X_test.ravel(),y_test)
plt.axis(lim)
plt.show()


# In[48]:


get_ipython().magic('matplotlib inline')
x2,y2=make_data(200)
plt.scatter(x2.ravel(),y2)


# In[52]:


degree = np.arange(0, 21)
train_score2, val_score2 = validation_curve(PolyNomialRegression(),x2,y2,'polynomialfeatures__degree',degree,cv=7)
plt.plot(degree,np.median(train_score2,1),color='blue',label='training2 score')
plt.plot(degree,np.median(train_score,1),color='blue',label='training score')
plt.plot(degree,np.median(val_score2,1),color='red',alpha=0.3,label='validation2 score')
plt.plot(degree,np.median(val_score,1),color='red',alpha=0.3,label='validation score')
plt.legend(loc='lower center')
plt.ylim(0,1)
plt.xlabel('degree')
plt.ylabel('score')


# In[54]:


from sklearn.grid_search import GridSearchCV
param_grid = {'polynomialfeatures__degree':np.arange(21),
              'linearregression__fit_intercept':[True,False],
              'linearregression__normalize':[True,False]}
grid = GridSearchCV(PolyNomialRegression(),param_grid,cv=7)
grid.fit(X,y)


# In[58]:


model = grid.best_estimator_
plt.scatter(X,y)
lim = plt.axis()
y_test = model.fit(X,y).predict(X_test)
plt.plot(X_test,y_test,hold=True)
plt.axis(lim)

