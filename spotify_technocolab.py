#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


track_fet=pd.read_csv(r'F:\Desktop\track_feats.csv')


# In[3]:


train_data=pd.read_csv(r'F:\Desktop\train_data.csv')


# In[4]:


track_fet.head()


# In[5]:


track_fet.info()


# In[6]:


track_fet.describe()


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(track_fet.corr(),cbar=True, annot=True, vmax=.8, square=True, fmt='.2f', annot_kws={'size': 10},cmap='Greens')


# In[8]:


train_data.head()


# In[9]:


train_data.describe()


# In[10]:


train_data.info()


# In[11]:


f, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(train_data.corr(),cbar=True, annot=True, vmax=.8, square=True, fmt='.2f', annot_kws={'size': 10},cmap='Greens')


# In[12]:


train_data['skip'] =  train_data['not_skipped'].replace({ 0 : 1, 1 : 0 })


# In[13]:


train_data = train_data.drop(['skip_1','skip_2', 'skip_3',	'not_skipped','hist_user_behavior_reason_end_appload'], axis=1)


# In[14]:


train_data.rename(columns={'track_id_clean': 'track_id'}, inplace=True)


# In[15]:


ft = pd.merge( train_data,track_fet, on=['track_id'], left_index=True, right_index=False, sort=True)
ft.shape


# In[16]:


ft.sort_values(axis=0, by=['session_id','session_position'], inplace=True)
ft.reset_index(drop=True,inplace=True)


# In[17]:


ft.columns.size


# In[18]:


ft.columns


# In[19]:


ft = ft.drop(columns=["session_id","track_id"],axis=1)
ft = pd.get_dummies(ft, drop_first=True)
ft.shape


# In[20]:


ft.info()


# In[21]:


X = ft.drop(['skip','short_pause_before_play','long_pause_before_play','hour_of_day','release_year','key','time_signature'],axis=1)
y = ft.skip


# In[22]:


from boruta import BorutaPy


# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


rf=RandomForestClassifier(random_state=1, n_estimators=100, n_jobs = -1).fit(X_train,y_train)


# In[26]:


rf.score(X_train,y_train)


# In[27]:


rf.score(X_test,y_test)


# In[28]:


import pickle
pickle.dump(rf,open('rf.pkl','wb'))


# In[29]:


pip freeze > requirements.txt


# In[30]:


X.columns


# In[ ]:





# In[ ]:




