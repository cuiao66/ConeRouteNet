
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('./interpolated.csv')


# In[5]:


groups = df.groupby('frame_id')


# In[14]:


left_camera = groups.get_group('left_camera')


# In[12]:


df.iloc[left_camera.index.to_list(),:].to_csv('./left.csv')


# In[15]:


right_camera = groups.get_group('right_camera')


# In[17]:


df.iloc[right_camera.index.to_list(),:].to_csv('./right.csv')


# In[18]:


center_camera = groups.get_group('center_camera')
df.iloc[center_camera.index.to_list(),:].to_csv('./center.csv')


# In[ ]:




