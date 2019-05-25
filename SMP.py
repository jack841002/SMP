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


# In[2]:


with open('SMP/train_img.txt', 'r') as f:
    imgs = f.read().splitlines()


# In[3]:


imageID = []
for i in range(len(imgs)):
    imageID.append(imgs[i].split('/')[-1])
imageID = pd.DataFrame(imageID)
imageID = imageID.rename(columns={0:'imageID'})


# In[4]:


imageID.head()


# In[5]:


label = pd.read_csv("SMP/train_label.txt", header = None) #處理沒有feature name的方式
label = label.rename(columns={0:'score'})
label.head()


# In[6]:


imageInformation = pd.read_csv("SMP/image_information.txt",skiprows=[764,767], header = None).astype('int64')
imageInformation = imageInformation.rename(columns={0:'imageID'})
imageInformation = imageInformation.rename(columns={1:'ViewCount'})
imageInformation = imageInformation.rename(columns={2:'FavoriteCount'})
imageInformation = imageInformation.rename(columns={3:'Message'})
imageInformation = imageInformation.reindex(columns=['imageID', 'ViewCount', 'FavoriteCount', 'Message']) #重設置index
imageInformation.head()


# In[7]:


with open('SMP/train_tags.json', 'r') as f:
    tags = json.load(f)
tags = pd.DataFrame(tags)
#tags.head()


# In[8]:


AllTags = []
for i in range(len(tags['Alltags'])):
    AllTags.append(len(tags['Alltags'][i].split()))

AllTags = pd.DataFrame(AllTags)
AllTags = AllTags.rename(columns={0:'TagsCount'})


# In[9]:


AllTags.head()


# In[10]:


# with open('SMP/train_temporalspatial.json', 'r') as ff:
#     temporalspatial = json.load(ff)
# temporalspatial = pd.DataFrame(temporalspatial)
# temporalspatial.head()


# In[11]:


with open('SMP/train_category.json', 'r') as ff:
    category = json.load(ff)
category = pd.DataFrame(category)
#category.head()


# In[12]:


#把字串符號轉成數字
Uid = category["Uid"]
Uid = pd.DataFrame(Uid)
Uid["Uid"] = Uid["Uid"].apply(str)
Uid["Uid_code"] = LabelEncoder().fit_transform(Uid["Uid"])
Uid.drop(columns = ["Uid"],inplace=True)
Uid.head()


# In[13]:


#把字串符號轉成數字
Concept = category["Concept"]
Concept = pd.DataFrame(Concept)
Concept["Concept"] = Concept["Concept"].apply(str)
Concept["Concept_code"] = LabelEncoder().fit_transform(Concept["Concept"])
Concept.drop(columns = ["Concept"],inplace=True)
Concept.head()


# In[14]:


# with open('SMP/train_additional.json', 'r') as ff:
#     additional = json.load(ff)
# additional = pd.DataFrame(additional)
# additional.head()


# In[15]:


trainData = pd.concat( [imageID , Uid], axis=1 )   # axis=1是X軸，axis=0是y軸
trainData = pd.concat( [trainData , Concept], axis=1 )
trainData = pd.concat( [trainData , AllTags], axis=1 )
trainData = pd.concat( [trainData , label], axis=1 )
trainData.head()


# In[16]:


trainData["imageID"] = trainData["imageID"].apply(int)
trainData = pd.merge(imageInformation, trainData, how='left', on='imageID')
trainData = trainData[['imageID','Uid_code','Concept_code','ViewCount','FavoriteCount','Message','TagsCount','score']]
trainData.head()


# In[17]:


trainData.shape


# In[18]:


trainData.dropna(axis=0,inplace=True)   # axis=1是X軸，axis=0是y軸


# In[19]:


trainData.shape


# In[20]:


trainData.isnull().any()


# In[21]:


temp , X_test= train_test_split(trainData, test_size=0.2)
X_train , X_valid = train_test_split(temp,test_size=0.1)
Y_train = X_train["score"]
X_train = X_train.drop(["score"],axis = 1)
Y_test = X_test["score"]
X_test = X_test.drop(["score"],axis = 1)
Y_valid = X_valid["score"]
X_valid = X_valid.drop(["score"],axis = 1)


# In[22]:


model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 15)


# In[23]:


Y_pred = model.predict(X_test).clip(0, 20)
print(Y_pred)


# In[24]:


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


# In[ ]:




