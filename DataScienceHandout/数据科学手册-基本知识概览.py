
# coding: utf-8

# In[59]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = sns.load_dataset('iris')
iris.head()
sns.set()
sns.pairplot(iris, hue='species', size=1.5)
plt.show()


# In[60]:


X_iris = iris.drop('species', axis=1)
X_iris.shape


# In[61]:


Y_iris = iris['species']
Y_iris.shape


# In[62]:


rng = np.random.RandomState(0)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x,y)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model


# In[63]:


X = x[:,np.newaxis]
X.shape


# In[64]:


model.fit(X,y)


# In[65]:


model.intercept_


# In[66]:


Xfit = np.linspace(-1,11)
Xfit = Xfit[:,np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(X, y)
plt.plot(Xfit,yfit)


# In[67]:


from sklearn.cross_validation import train_test_split

Xtrain,Xtest,ytrain,ytest = train_test_split(X_iris,Y_iris,random_state=1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain,ytrain)
y_predict = model.predict(Xtest)


# In[68]:


from sklearn.metrics import accuracy_score
accuracy_score(y_predict, ytest)


# In[69]:


from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X_iris)
y_2d = model.transform(X_iris)
iris['PCA1']=y_2d[:,0]
iris['PCA2']=y_2d[:,1]
sns.lmplot('PCA1','PCA2',hue='species',data=iris,fit_reg=False)


# In[70]:


from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=3,covariance_type='full')
model.fit(X_iris)
y_gmm=model.predict(X_iris)
iris['cluster']=y_gmm
sns.lmplot('PCA1','PCA2',data=iris,hue='species',col='cluster',fit_reg=False)


# In[71]:


from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
print(X.shape,y.shape)

from sklearn.manifold import Isomap
model = Isomap(n_components=2)
model.fit(digits.data)
data_pro = model.transform(digits.data)
print(data_pro.shape)

plt.scatter(data_pro[:,0], data_pro[:,1],c=digits.target,edgecolor='none',alpha=0.5,cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label',ticks=range(10))
plt.clim(-0.5,9.5)


# In[72]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain,ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


# In[73]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)

