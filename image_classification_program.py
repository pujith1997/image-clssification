#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 


# In[2]:


from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


data = pd.read_csv(r"C:\Users\pujith\OneDrive\Desktop\input23\fashion-mnist_train.csv")


# In[15]:


data.head()


# In[ ]:


#extracting the data from dataset and viewing them up close


# In[16]:


a = data.iloc[3,1:].values


# In[17]:


#reshaping the extracted data into resonable size
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[18]:


#preparing the data 
#separating lables and data values 
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[19]:


#creating test and train sizes
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size = 0.2,random_state=4)


# In[20]:


#checking the data 

x_train.head()


# In[21]:


y_train.head()


# In[23]:


#call rf classifier
rf = RandomForestClassifier(n_estimators=100)


# In[24]:


#fit the model
rf.fit(x_train,y_train)


# In[25]:


#prediction on the test data 
pred = rf.predict(x_test)


# In[26]:


pred


# In[27]:


#check prediction accuracy

correct = y_test.values

#calculate no of correctly predicted values 
count = 0
for i in range (len(pred)):
    if pred[i]==correct[i]:
        count+=1


# In[28]:


count


# In[29]:


#total values that prediction was run for 
len(pred)


# In[30]:


#accuracy rate 
10520/12000


# In[31]:


87.66


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




