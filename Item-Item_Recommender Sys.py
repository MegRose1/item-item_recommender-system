#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate
from surprise import SVD
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# In[15]:


pip install -U statsmodels


# In[19]:


#IDating dataset that I am using
mydata= pd.read_csv("C:\Python\Recommendation System\online_retailer_data.csv")
mydata.head()


# In[20]:


mydata.shape


# In[3]:


# Data visualization and analysis:


# In[4]:


#The pie chart illustrates most items are rated 4 by customers, therefore we can conclude most itmes are rated positively
ratings = mydata["Rating"].value_counts()
numbers = ratings.index
quantity = ratings.values
import plotly.express as px
fig = px.pie(mydata, values=quantity, names=numbers)
fig.show()


# In[21]:


#Filter data to only unique items and customers
unique_customers = mydata['CustomerID'].unique()
unique_customers_qt = len(unique_customers)
unique_items = mydata['ItemID'].unique()
unique_items_qt = len(unique_items)

print(f"There are {unique_customers_qt} unique customers and {unique_items_qt} unique itmes in the dataset")


# In[22]:


#The top 10 itmes that got rated 5 out of 5
mydata2 = mydata.query("Rating == 5")
print(mydata2["ItemID"].value_counts().head(10))


# In[24]:


# Top 10 users based on rating
#The highest number of ratings by a user is 464 which is far from the actual number of products present in the data. We can build a recommendation system to recommend products to users which they have not interacted with.
customers_rated_most = mydata.groupby(by='CustomerID')['Rating'].count().sort_values(ascending=False)[:10]
print('Top 10 customers based on their ratings: \n',customers_rated_most)


# In[30]:


#List all the itmes that customer 15574 bought and rated (an example)
id12433 = mydata2.loc[mydata2['CustomerID']==15574, 'ItemID']
id12433.head(20)


# In[31]:


counts = mydata['CustomerID'].value_counts()
mydata_final = mydata[mydata['CustomerID'].isin(counts[counts >= 50].index)]

print('The number of observations in the final data =', len(mydata_final))
print('Number of unique USERS in the final data = ', mydata_final['CustomerID'].nunique())
print('Number of unique ITEMS in the final data = ', mydata_final['ItemID'].nunique())


# In[32]:


#Group dataset by Item ID and calculate the mean rating for each item
mydata_final.groupby(['ItemID'])['Rating'].agg(['mean','count']).sort_values(["mean", "count"], ascending=False).head(20)


# In[33]:


#However,since an item that has high rating might have only been ASSORTED BAGS recieved the hightest rating counts
mydata_final.groupby(['ItemID'])['Rating'].agg(['mean','count']).sort_values(["count"], ascending=False).head(10)


# In[34]:


#create dataframe
ratings_mean_count = pd.DataFrame(mydata_final.groupby(['ItemID'])['Rating'].mean())
ratings_mean_count['Nr of Ratings'] = pd.DataFrame(mydata_final.groupby(['ItemID'])['Rating'].count())


# In[35]:


import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD
ratings_mean_count.head(30).plot(kind = "bar")


# In[37]:


#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Rating'].hist(bins=50)


# In[38]:


from matplotlib import pyplot as plt
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='Nr of Ratings', data=ratings_mean_count, alpha=0.5)


# In[39]:


#Finding Similarities
#The pivot table displays items and corresponding customer rating
customer_item_rating = mydata_final.pivot_table(index='CustomerID', columns='ItemID', values='Rating', fill_value=0)
customer_item_rating.head(10)


# In[40]:


#Transpose the customer item rating matrix and using Scikit-Learn, SVD can run
X = customer_item_rating.T
SVD = TruncatedSVD(n_components=12, random_state=5)
resultant_matrix = SVD.fit_transform(X)
resultant_matrix.shape
#Correlation coefficient()
corr_mat = np.corrcoef(resultant_matrix)
corr_mat.shape


# In[41]:


#Now let’s find similar items to ASSORTED BAGS
col_idx = customer_item_rating.columns.get_loc("METAL SIGN")
corr_specific = corr_mat[col_idx]
pd.DataFrame({'corr_specific':corr_specific, 'ItemID': customer_item_rating.columns})\
.sort_values('corr_specific', ascending=False).head()


# In[42]:


#find similar items to CHOCOLATE HOT WATER BOTTLE
col_idx = customer_item_rating.columns.get_loc("HAND WARMER ASSORTED")
corr_specific = corr_mat[col_idx]
pd.DataFrame({'corr_specific':corr_specific, 'ItemID': customer_item_rating.columns})\
.sort_values('corr_specific', ascending=False).head()
#Every item has a 100% Pearson Correlation with itself as expected 


# In[90]:


# Data correlation matrix
corr_metrics = customer_item_rating.corr()
corr_metrics.style.background_gradient()


# In[63]:


#Recommendation:
#1st the 20 items being rated/liked by the customer: 15574 are shown below:
dataset_sort_des = mydata_final.sort_values(['CustomerID', 'InvoiceDate'], ascending=[True, False])
filter1 = dataset_sort_des[dataset_sort_des['CustomerID'] == 15574].ItemID
filter1 = filter1.tolist()
filter1 = filter1[:20]
print("Items liked by customer 15574: ",filter1)


# In[91]:


#customer_item_rating
print("List of 10 items to recommend to a customer 15574 who has liked 'METAL SIGN'")
print(customer_item_rating.corr()['METAL SIGN'].sort_values(ascending=False).iloc[:20])

