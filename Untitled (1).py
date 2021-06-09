#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#THE SPARKS FOUNDATION
#TASK - 1
#TO PREDICT THE PERCENTAGE OF A STUDENT BASED ON NUMBER OF STUDY HOURS
#BY: HITHA M GOWDA


# In[33]:


#IMPORTING REQUIRED LIBS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', '')


# In[26]:


#READING DATA FROM DATASET
dataset = "https://bit.ly/w-data"
data = pd.read_csv(dataset)
print("data sucessfully imported")


# In[25]:


print(data.shape)


# In[28]:


#DESCRIPTION OF DATA
print(data.describe())


# In[32]:


#CHECKING NULL VALUE
print("First five records")
data.head(5)


# In[8]:


print("Last five records")
data.tail(5)


# In[9]:


print(data.corr())


# In[7]:



import pandas as pd
dataset = "https://bit.ly/w-data"
data = pd.read_csv(dataset)
import matplotlib.pyplot as plt


# In[8]:


#plotting score distribution
data.plot(x='Hours', y='Scores',style='o')
plt.title('Hours vs percentage')
plt.xlabel('Hours studied')
plt.ylabel('Score')
plt.show()


# In[9]:


#feature selection 
x=data.iloc[:, :-1].values
y=data.iloc[:, 1].values
print("feature selection completed")


# In[11]:


#testing, training and spliting model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=7)


# In[13]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
print("training completed!")


# In[14]:


y_pred = model.predict(x_test)


# In[15]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))


# In[16]:


from sklearn import metrics
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))


# In[18]:


#comparing
dataframe = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(dataframe)


# In[19]:


dataframe.plot(kind = 'line')


# In[21]:


#calculating predicted score
x_input = eval(input("Enter number of hours="))
predicted_value=model.predict([[x_input]])
print(predicted_value)


# In[ ]:




