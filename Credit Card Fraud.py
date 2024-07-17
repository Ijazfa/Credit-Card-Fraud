#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[2]:


chunk_size = 100000


# In[3]:


for chunk in pd.read_csv("C:/Users/Ijaz khan/Downloads/creditcard.csv", chunksize=chunk_size):
    chunks = chunk
    


# In[32]:


df = chunks
df


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.nunique()


# In[9]:


plt.figure(figsize=(8,6))
sns.countplot(x=df['Class'])
plt.title('Class')
plt.show()


# In[10]:


df.describe()


# In[11]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr())
plt.show()


# In[12]:


df.corr()


# In[13]:


x = df.drop(columns = 'Class')
y = df['Class']


# In[ ]:





# In[14]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
xs = scaler.fit_transform(x)
xs.shape


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
select = RandomForestClassifier(n_estimators=100, random_state= 34)


# In[ ]:





# In[16]:


select.fit(xs,y)


# In[17]:


smf = SelectFromModel(select, threshold='median')
smf.fit(xs,y)


# In[18]:


x_selected = smf.transform(xs)



# In[19]:


selected_features = x.columns[smf.get_support()]

print(f'original number of features : {x.shape[1]}')
print(f'selected number of features : {x_selected.shape[1]}')
print(f'selected features : {selected_features}')


# In[20]:


from sklearn.model_selection import StratifiedGroupKFold


# In[21]:


skf = StratifiedGroupKFold()


# In[22]:


model = RandomForestClassifier(n_estimators=100, random_state=42)

for train_index, test_index in skf.split(x,y):
    x_train,x_test = x_selected[train_index],x_selected[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]

model.fit(x_train,y_train)


# In[23]:


from sklearn.model_selection import train_test_split



# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=0)

model.fit(x_train,y_train)


# In[25]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[26]:


y_pred = model.predict(x_test)


# In[27]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,r2_score


# In[28]:


print(accuracy_score(y_pred,y_test))


# In[29]:


print(classification_report(y_pred,y_test))


# In[30]:


cm = confusion_matrix(y_pred,y_test)
cm


# In[31]:


plt.figure(figsize=(8,6))
sns.heatmap(cm,annot = True, cmap = 'coolwarm')
plt.show()


# In[33]:


new_data = pd.DataFrame({
    'Time': [1],
    'V1': [-1.358354],
    'V2': [-1.340163],
    'V3': [1.773209],
    'V4': [0.379779],
    'V5': [-0.503198],
    'V6': [1.800499],
    'V7': [0.791461],
    'V8': [0.247676],
    'V9': [-1.514654],
    'V10': [0.207642],
    'V11': [0.624501],
    'V12': [0.066084],
    'V13': [0.717293],
    'V14': [-0.165945],
    'V15': [2.345865],
    'V16': [-2.890083],
    'V17': [1.109969],
    'V18': [-0.121359],
    'V19': [0.924081],
    'V20': [0.29266],
    'V21': [0.723648],
    'V22': [0.736482],
    'V23': [0.037676],
    'V24': [0.377485],
    'V25': [-0.059809],
    'V26': [-0.362761],
    'V27': [-0.032821],
    'V28': [0.055408],
    'Amount': [378.66]
})



# In[34]:


New_data_scaler = scaler.transform(new_data)


# In[35]:


predict = model.predict(new_data)


# In[36]:


if predict[0]==1:
    print(f' The transaction is predicted to be Fraude :{predict}')
else:
    print(f' The transaction is proper: {predict}')


# In[ ]:




