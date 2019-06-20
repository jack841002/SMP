#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import sys
from sklearn.preprocessing import LabelEncoder   #把字串符號轉數字
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from matplotlib import pyplot as plt
import math


# In[2]:


label = pd.read_csv("SMP/train_label.txt", header = None) #處理沒有feature name的方式
label = label.rename(columns={0:'score'})
label.head()


# In[3]:


with open('SMP/train_tags.json', 'r') as f:
    tags = json.load(f)
tags = pd.DataFrame(tags)
#tags.head()


# In[4]:


AllTags = []
for i in range(len(tags['Alltags'])):
    AllTags.append(len(tags['Alltags'][i].split()))

AllTags = pd.DataFrame(AllTags)
AllTags = AllTags.rename(columns={0:'TagsCount'})
AllTags.head()


# In[5]:


with open('SMP/train_temporalspatial.json', 'r') as ff:
    temporalspatial = json.load(ff)
temporalspatial = pd.DataFrame(temporalspatial)
#temporalspatial.head()


# In[6]:


postDate = temporalspatial[["Uid","Pid","Postdate"]]
postDate = pd.DataFrame(postDate)
postDate.head()


# In[7]:


with open('SMP/train_category.json', 'r') as ff:
    category = json.load(ff)
category = pd.DataFrame(category)


# In[8]:


category = category[["Category", "Subcategory", "Concept"]]
category = pd.DataFrame(category)
category.head()


# In[9]:


trainData = pd.concat( [postDate, category], axis=1 )
trainData = pd.concat( [trainData, AllTags], axis=1 )
trainData = pd.concat( [trainData, label], axis=1 )


# In[10]:


trainData.head()


# In[11]:


#trainData.isnull().any()


# In[12]:


with open('SMP/train_img.txt', 'r') as f:
    imgs = f.read().splitlines()


# In[13]:


imageID = []
for i in range(len(imgs)):
    imageID.append(imgs[i].split('/')[-1])
imageID = pd.DataFrame(imageID)
imageID = imageID.rename(columns={0:'imageID'})
imageID.head()


# In[14]:


trainData = pd.concat([imageID, trainData], axis=1)
trainData.head()


# In[15]:


with open('SMP/image_object.txt', 'r') as f:
    object = f.read().splitlines()
len(object[0].split(','))

imageID = []
for i in range(len(object)):
    imageID.append(object[i].split(',')[0])
imageID = pd.DataFrame(imageID)
imageID = imageID.rename(columns={0:'imageID'})
imageID.head()

AllObject = []
for i in range(len(object)):
    AllObject.append(len(object[i].split(','))-2)
AllObject = pd.DataFrame(AllObject)
AllObject = AllObject.rename(columns={0:'AllObject'})
AllObject.head()

trainObject = pd.concat( [imageID , AllObject], axis=1 )

trainObject.head()


# In[16]:


trainData = pd.merge(trainObject, trainData, how='left', on='imageID')
trainData.head()


# In[17]:


trainData = trainData.drop(['imageID'], axis=1)
#trainData.columns
trainData = trainData[['Uid', 'Pid', 'Postdate', 'Category', 'Subcategory','Concept', 'TagsCount', 'AllObject', 'score']]
trainData.head()


# In[18]:


trainData["Uid"] = trainData["Uid"].apply(str)
trainData["Uid"] = LabelEncoder().fit_transform(trainData["Uid"])
trainData["Category"] = trainData["Category"].apply(str)
trainData["Category"] = LabelEncoder().fit_transform(trainData["Category"])
trainData["Subcategory"] = trainData["Subcategory"].apply(str)
trainData["Subcategory"] = LabelEncoder().fit_transform(trainData["Subcategory"])
trainData["Concept"] = trainData["Concept"].apply(str)
trainData["Concept"] = LabelEncoder().fit_transform(trainData["Concept"])

trainData["Pid"] = trainData["Pid"].apply(int)
trainData["Postdate"] = trainData["Postdate"].apply(int)

trainData.head()


# In[19]:


temp , X_test= train_test_split(trainData, test_size=0.2)
X_train , X_valid = train_test_split(temp,test_size=0.1)
Y_train = X_train["score"]
X_train = X_train.drop(["score"],axis = 1)
Y_test = X_test["score"]
X_test = X_test.drop(["score"],axis = 1)
Y_valid = X_valid["score"]
X_valid = X_valid.drop(["score"],axis = 1)


# In[20]:


model = XGBRegressor(
    max_depth=9,
    n_estimators=1000,
    min_child_weight=1, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 15)


# In[21]:


Y_pred = model.predict(X_test).clip(0, 20)
print(Y_pred)


# In[22]:


error = []
for i in range(len(Y_test)):
    error.append(Y_test.values[i] - Y_pred[i])
    
#print("Errors: ", error)
#print(error)
squaredError = []
absError = []
for val in error:
    squaredError.append(val * val)#平方
    absError.append(abs(val))#誤差絕對值
    
#print("Square Error: ", squaredError)
#print("Absolute Value of Error: ", absError)
print("MSE = ", sum(squaredError) / len(squaredError))#平均平方誤差MSE


#from math import sqrt
#print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))#平均平方根誤差RMSE
print("MAE = ", sum(absError) / len(absError))#平均絕對誤差MAE


# In[23]:


plot_importance(model)
plt.show()


# In[ ]:





# In[ ]:




