#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[69]:


train = pd.read_csv(r'C:\Users\HP\DATASETS/train.csv')
test = pd.read_csv(r'C:\Users\HP\DATASETS/test.csv')


# In[70]:


train.head()


# In[71]:


# Indice de las filas en forma de lista
train.columns


# In[72]:


# numero de filas y de columnas 
train.shape


# In[73]:


# se revisa la condicion de los datos y si existen algunos nulos
train.info()


# In[74]:


# aqui revisamos como estan distribuidas las variables 
#numericas al igual que el conteo nuevamente de los datos y metricas como promedios,DE , quantiles etc.
train.describe()


# In[75]:


# comportamiento de las variables categóricas
train.describe(include=['O'])


# In[76]:


train.groupby(['Survived']).count()['PassengerId']


# In[77]:


train.groupby(['Survived','Sex']).count()['PassengerId']


# In[78]:


grouped_sex = train.groupby(['Survived','Sex']).count()['PassengerId']
print(grouped_sex)
(grouped_sex.unstack(level=0).plot.bar())
plt.show()


# In[79]:


# embarked vs pclass
print(train.groupby(['Pclass', 'Embarked'])
        .count()['PassengerId']
        .unstack(level=0)
        .plot.bar())


# In[80]:


train[['Survived', 'Sex', 'Age', 'Pclass']].head(3)


# In[81]:


train[['Survived', 'Sex', 'Age', 'Pclass']].info()


# In[82]:


(train[train['Age'].isna()]
      .groupby(['Sex', 'Pclass'])
      .count()['PassengerId']
      .unstack(level=0))


# In[83]:


(train[train['Age'].isna()]
      .groupby(['SibSp', 'Parch'])
      .count()['PassengerId']
      .unstack(level=0))


# In[84]:


train['Age'].median()


# In[85]:


train['Age'] = train['Age'].fillna(28.0)
train[['Survived', 'Sex', 'Age', 'Pclass']].info()


# In[86]:


train['Sex'] = train['Sex'].map({'female': 1, 'male': 0}).astype(int)


# In[87]:


train[['Survived', 'Sex', 'Age', 'Pclass']].head(3)


# In[88]:


train['FlagSolo'] = np.where(
    (train['SibSp'] == 0) & (train['Parch'] == 0), 1, 0)


# In[89]:


grouped_flag = train.groupby(['Survived','FlagSolo']).count()['PassengerId']
print(grouped_flag)
(grouped_flag.unstack(level=0).plot.bar())
plt.show()


# In[90]:


train[['Survived', 'Sex', 'Age', 'Pclass', 'FlagSolo']].head(3)


# In[91]:


Y_train = train['Survived']

features = ['Sex', 'Age', 'Pclass', 'FlagSolo']
X_train = train[features]

print(Y_train.shape, X_train.shape)


# In[92]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


# In[93]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)


# In[94]:


from sklearn.metrics import plot_confusion_matrix

def conf_mat_acc(modelo):
  disp = plot_confusion_matrix(modelo, X_train, Y_train,
                        cmap=plt.cm.Blues, values_format="d")
  true_pred = disp.confusion_matrix[0,0]+disp.confusion_matrix[1,1]
  total_data = np.sum(disp.confusion_matrix)
  accuracy = true_pred/total_data
  print('accuracy: ', np.round(accuracy, 2))
  plt.show()


# In[95]:


conf_mat_acc(logreg)


# In[96]:


conf_mat_acc(decision_tree)


# In[97]:


print(test.head(3))
test.info()


# In[98]:


test['Sex'] = test['Sex'].map({'female': 1, 'male': 0}).astype(int)

test['Age'] = test['Age'].fillna(28.0)

test['FlagSolo'] = np.where(
    (test['SibSp'] == 0) & (test['Parch'] == 0), 1, 0)


# In[99]:


print(test.info())
test[features].head(3)


# In[100]:


X_test = test[features]

print(X_test.shape)


# In[101]:


Y_pred_log = logreg.predict(X_test)
Y_pred_tree = decision_tree.predict(X_test)
print(Y_pred_log[0:10])


# In[102]:


print(Y_pred_log[0:20])
print(Y_pred_tree[0:20])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




