#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


cancer=pd.read_csv('breastcancer.csv')


# In[31]:


cancer.head(2)


# In[32]:


cancer.describe().T


# In[33]:


cancer.isnull().sum()


# In[34]:


cancer.info()


# In[35]:


cancer=(cancer.drop('id',axis=1))


# In[36]:


cancer.hist(figsize=(20,20))


# In[37]:


cancer.columns


# In[38]:


#convert categorical to numerical data sets
cancer['diagnosis']=cancer.replace(['B','M'],[0,1])


# In[39]:


cancer.groupby('diagnosis').size()


# In[40]:


sns.countplot(x='diagnosis',data=cancer)


# In[41]:


plt.figure(figsize=(30,30))
sns.heatmap(cancer.corr(),cmap='RdYlGn',annot=True)


# In[84]:


cancer.shape


# In[94]:


#cross validation dividing data into label
x=cancer.iloc[:,1:31]
y=cancer.iloc[:,-31]
x.head(2)


# In[95]:


from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
X=pd.DataFrame(scx.fit_transform(x))


# In[96]:


from sklearn.model_selection import train_test_split


# In[97]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.33,random_state=42)


# In[98]:


from sklearn.neighbors import KNeighborsClassifier


# In[99]:


knn=KNeighborsClassifier(n_neighbors=7)


# In[100]:


knn.fit(X_train,y_train)


# In[118]:


#find max trai score and test score
train_score=[]
test_score=[]
train_score.append(knn.score(X_train,y_train))
test_score.append(knn.score(X_test,y_test))


# In[119]:


max_train_score=max(train_score)


# In[120]:


max_train_score


# In[121]:


max_test_score=max(test_score)


# In[122]:


max_test_score


# In[128]:


#fidn the nearst neighbor
knn.score(X_test,y_test)


# In[139]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import cross_val_score,GridSearchCV
ypredict=knn.predict(X_test)


# In[151]:


cnf=confusion_matrix(y_test,ypredict)
sns.heatmap(cnf,annot=True,fmt='g')
plt.title('confusion matrix',y=1.1)
plt.ylabel('actual label')
plt.xlabel('predicted label')


# In[148]:


pd.crosstab(y_test,ypredict,rownames=['True'],colnames=['predicted'])


# In[152]:


accuracy_score(y_test,ypredict)*100


# In[155]:


print(classification_report(y_test,ypredict))


# In[176]:


scores=(cross_val_score(knn,X_train,y_train,cv=10))


# In[177]:


scores.mean()


# In[178]:


scores.max()


# In[179]:


scores.min()


# In[180]:


parameter_grid={'n_neighbors':np.arange(1,50)}


# In[181]:


knncv=GridSearchCV(knn,parameter_grid,cv=5)


# In[183]:


knncv.fit(X_train,y_train)


# In[185]:


print('best score'+str(knncv.best_score_))


# In[190]:


print('best parameter'+str(knncv.best_params_))


# In[ ]:




