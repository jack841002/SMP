#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import requests
from PIL import Image
import urllib
import pandas as pd
import matplotlib.pyplot as plt

import plotly
from plotly import tools
from plotly.offline import iplot
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


# In[2]:


with open('SMP/train_img.txt', 'r') as f:
    imgs = f.read().splitlines()


# In[3]:


print('圖片數量', len(imgs))


# In[4]:


imgs[:3]


# In[5]:


with open('SMP/train_category.json', 'r') as f:
    category = json.load(f)


# In[6]:


print('category 數量', len(category))


# In[7]:


category[:3]


# In[8]:


category = pd.DataFrame.from_dict(category)


# In[9]:


category.head()


# In[10]:


category.nunique()


# In[11]:


trace = go.Pie(labels=category.Category.value_counts().index, values=category.Category.value_counts().values)
layout = go.Layout(
    title = 'All Categories'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[12]:


category.Concept.value_counts().describe()


# In[13]:


trace = go.Bar(x=category.Concept.value_counts().index[:10], y=category.Concept.value_counts().values[:10])
layout = go.Layout(
    title = 'Top 10 Concepts'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[14]:


trace = go.Box(y=category.Concept.value_counts())
layout = go.Layout(
    title = 'Concepts number distribution'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[15]:


category.Subcategory.value_counts().describe()


# In[16]:


sub_cat_hist = category.Subcategory.value_counts()
threshold = sub_cat_hist.quantile(.5)
mask = sub_cat_hist > threshold
others = sub_cat_hist.loc[~mask].sum()
sub_cat_hist = sub_cat_hist.loc[mask]
sub_cat_hist['others']=others


# In[17]:


trace = go.Pie(labels=sub_cat_hist.index, values=sub_cat_hist.values)
layout = go.Layout(
    title = 'Top 50% subcategories vs others'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[18]:


with open('SMP/train_tags.json', 'r') as f:
    tags = json.load(f)


# In[19]:


print('標籤數量', len(tags))


# In[20]:


tags[:3]


# In[21]:


tags = pd.DataFrame.from_dict(tags)


# In[22]:


tags.head()


# In[23]:


tag_dict = {}
for row in tags.Alltags:
    tag_list = row.split()
    for tag in tag_list:
        tag_dict[tag] = tag_dict.get(tag, 0) + 1


# In[24]:


top_10_tags = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True)[:10]


# In[25]:


sum([tag[1] for tag in top_10_tags]) / sum(tag_dict.values())


# In[26]:


trace = go.Pie(labels=[tag[0] for tag in top_10_tags], values=[tag[1] for tag in top_10_tags])
layout = go.Layout(
    title = 'Top 10 tags'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[27]:


trace = go.Pie(labels=tags.Mediatype.value_counts().index, values=tags.Mediatype.value_counts().values)
layout = go.Layout(
    title = 'Media Type'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[28]:


with open('SMP/train_temporalspatial.json', 'r') as f:
    temporalspatial = json.load(f)


# In[29]:


print('位置數量', len(temporalspatial))


# In[30]:


temporalspatial[:3]


# In[31]:


temporalspatial = pd.DataFrame.from_dict(temporalspatial)


# In[32]:


temporalspatial.head()


# In[33]:


temporalspatial[['Geoaccuracy', 'Longitude', 'Latitude']].describe()


# In[34]:


temporalspatial.Geoaccuracy.value_counts()[0] / temporalspatial.Geoaccuracy.value_counts().sum()


# In[35]:


temporalspatial.Longitude.value_counts()[0] / temporalspatial.Longitude.value_counts().sum()


# In[36]:


temporalspatial.Latitude.value_counts()[0] / temporalspatial.Latitude.value_counts().sum()


# In[37]:


with open('SMP/train_additional.json', 'r') as f:
    addition = json.load(f)


# In[38]:


print('額外數量', len(addition))


# In[39]:


addition[:3]


# In[40]:


addition = pd.DataFrame.from_dict(addition)


# In[41]:


addition.head()


# In[42]:


trace = go.Pie(labels=addition.Ispublic.value_counts().index, values=addition.Ispublic.value_counts().values)
layout = go.Layout(
    title = 'Ispublic'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[43]:


trace = go.Pie(labels=addition.Mediastatus.value_counts().index, values=addition.Mediastatus.value_counts().values)
layout = go.Layout(
    title = 'Media Status'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[44]:


print(addition.Pathalias.nunique(), '種')


# In[45]:


with open('SMP/train_label.txt', 'r') as f:
    label = f.read().splitlines()


# In[46]:


print('答案數量', len(label))


# In[47]:


label[:3]


# In[48]:


label = pd.DataFrame.from_dict(label)
label.head()


# In[49]:


label[0] = label[0].astype('float')
label[0].describe()


# In[50]:


trace = go.Box(y=label[0])
layout = go.Layout(
    title = 'Answer distribution'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[ ]:




