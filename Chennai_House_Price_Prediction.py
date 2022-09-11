#!/usr/bin/env python
# coding: utf-8

# In[172]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression


# In[73]:


data=pd.read_csv('train-chennai-sale.csv')


# In[74]:


data.head(2)


# In[4]:


data['DATE_BUILD']
#need to remove this 


# In[5]:


data.head(2)


# In[6]:


data.shape


# In[7]:


data.info()


# In[ ]:





# In[8]:


#checking the the null value
data.isnull().sum()


# In[9]:


data['N_BEDROOM']


# In[75]:


data.isnull().sum()


# In[76]:


data.describe()


# # Handling Missing values

# In[77]:


data['N_BEDROOM']=data['N_BEDROOM'].fillna(data['N_BEDROOM'].mode()[0])
data['N_BATHROOM']=data['N_BATHROOM'].fillna(data['N_BATHROOM'].mode()[0])
data['QS_OVERALL']=data['QS_OVERALL'].fillna(data['QS_OVERALL'].median())


# In[13]:


data.isnull().sum()


# # Checking null values

# In[78]:


data.drop_duplicates().shape


# In[79]:


#We are checking for spelling mistakes
#data['PRT_ID'].duplicated().sum()
data['PRT_ID'].unique()


# In[80]:


data.head(3)


# # Checking spelling mistakes and correcting it.

# In[92]:


data['AREA'].value_counts()


# In[90]:


data.AREA = data.AREA.replace({'Velchery':'Velachery',
                                         'Ann Nagar':'Anna Nagar',
                       'Ana Nagar':'Anna Nagar',
                       'TNagar':'T Nagar',
                       'Chrmpet':'Chrompet',
                               'KKNagar':'KK Nagar',
                       'Karapakam':'Karapakkam',
                    
                                        })


# In[93]:


data['SALE_COND'].value_counts()


# In[20]:


data['UTILITY_AVAIL'].value_counts()


# In[98]:


data.UTILITY_AVAIL=data.UTILITY_AVAIL.replace({'All Pub':'AllPub','NoSewr ':'NoSeWa'})
                                               
                      


# In[99]:


print(data['UTILITY_AVAIL'].unique())


# In[94]:


data.SALE_COND = data.SALE_COND.replace({'adj land':'AdjLand',
                                         'Adj Land':'AdjLand',
                       'normal sale':'Normal Sale',
                       'Partiall':'Partial',
                       'PartiaLl':'Partial',
                       'Ab Normal':'AbNormal',
                    
                                        })


# In[95]:


print(data['SALE_COND'].unique())


# In[25]:


data['PARK_FACIL'].value_counts()


# In[100]:


data.PARK_FACIL=data.PARK_FACIL.replace({'Noo':'No'})


# In[104]:


data['BUILDTYPE'].unique()


# In[103]:


data.BUILDTYPE = data.BUILDTYPE.replace({'Comercial':'Commercial',
                                         'Other':'Others',
                    
                                        })


# In[28]:


data['SALE_COND']


# In[29]:


data['PARK_FACIL'].value_counts()


# In[105]:


data.PARK_FACIL = data.PARK_FACIL.replace({'Noo':'No'
                                        })


# In[31]:


data


# In[106]:


data['BUILDTYPE'].value_counts()
#data.BUILDTYPE = data.BUILDTYPE.replace({'Other':'Others','Comercial':'Commercial'})


# In[107]:


data['UTILITY_AVAIL'].value_counts()


# In[108]:


data.UTILITY_AVAIL = data.UTILITY_AVAIL.replace({'All Pub':'AllPub','NoSewr':'NoSeWa'
                                        })
data.UTILITY_AVAIL = data.UTILITY_AVAIL.replace({'NoSewr':'NoSeWa'
                                        })


# In[109]:


data.UTILITY_AVAIL = data.UTILITY_AVAIL.replace({'NoSewr':'NoSeWa'})
                                        
data['UTILITY_AVAIL'].value_counts()


# In[110]:


data['BUILDTYPE'].value_counts()


# In[111]:


data.BUILDTYPE=data.BUILDTYPE.replace({'Comercial':'Commercial','Other':'Others'})


# In[114]:


data.STREET=data.STREET.replace({'Pavd':'Paved','NoAccess':'No Access'})


# In[115]:


data['STREET'].value_counts()


# In[113]:


data['MZZONE'].value_counts()


# # checking all the Unique values

# In[116]:


print(data['AREA'].unique())
print(data['SALE_COND'].unique())
print(data['PARK_FACIL'].unique())
print(data['BUILDTYPE'].unique())
print(data['UTILITY_AVAIL'].unique())
print(data['STREET'].unique())
print(data['MZZONE'].unique())


# # changing the appropriate data types

# In[117]:


data.N_BEDROOM = data.N_BEDROOM.astype(int)
data.N_BATHROOM = data.N_BATHROOM.astype(int)
data.QS_ROOMS = data.QS_ROOMS.astype(int)
data.QS_BATHROOM = data.QS_BATHROOM.astype(int)
data.QS_BEDROOM = data.QS_BEDROOM.astype(int)


# In[42]:


data.info()


# In[43]:


data.head(2)


# In[118]:


#we have date column in object format we have to change it in date time format
data.DATE_SALE=pd.to_datetime(data.DATE_SALE,format='%d-%m-%Y')
data.DATE_BUILD=pd.to_datetime(data.DATE_BUILD,format='%d-%m-%Y')


# In[45]:


data.info()


# In[119]:


# Creating total_price column by adding reg_fee,commis and sales_price columns
data['TOTAL_PRICE'] = pd.DataFrame(data.REG_FEE	+ data.COMMIS	+ data.SALES_PRICE)


# In[120]:


#dropping unnecessary columns
data.drop(['PRT_ID','REG_FEE','COMMIS'], axis=1)


# # Creating the age of the building based on datesale and date build column 

# In[121]:


data.info()


# In[125]:


data['PROPERTY_AGE'] = pd.DatetimeIndex(data.DATE_SALE).year - pd.DatetimeIndex(data.DATE_BUILD).year


# In[126]:


data.head(2)


# In[122]:


data.head()


# # Exploratory Data Analysis

# In[127]:


plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True,linewidth=0.4,cmap='coolwarm');


# Sales Price is the traret features.
# There are some Correlation between features and target variables

# # Distributions of all the features

# In[128]:


plt.figure(figsize=(30, 18), dpi=200)

plt.subplot(5,4,1)
sns.histplot(data.AREA, linewidth=0,kde=True)

plt.subplot(5,4,2)
sns.histplot(data.INT_SQFT, linewidth=0,kde=True)

plt.subplot(5,4,3)
sns.histplot(data.DATE_SALE, linewidth=0,kde=True)

plt.subplot(5,4,4)
sns.histplot(data.DIST_MAINROAD, linewidth=0,kde=True)

plt.subplot(5,4,5)
sns.histplot(data.N_BEDROOM, linewidth=0,kde=True)

plt.subplot(5,4,6)
sns.histplot(data.N_BATHROOM, linewidth=0,kde=True)

plt.subplot(5,4,7)
sns.histplot(data.N_ROOM, linewidth=0,kde=True)

plt.subplot(5,4,7)
sns.histplot(data.SALE_COND, linewidth=0,kde=True)

plt.subplot(5,4,7)
sns.histplot(data.PARK_FACIL, linewidth=0,kde=True)

plt.subplot(5,4,8)
sns.histplot(data.UTILITY_AVAIL, linewidth=0,kde=True)

plt.subplot(5,4,9)
sns.histplot(data.STREET, linewidth=0,kde=True)

plt.subplot(5,4,10)
sns.histplot(data.MZZONE, linewidth=0,kde=True)

plt.subplot(5,4,11)
sns.histplot(data.QS_ROOMS, linewidth=0,kde=True)

plt.subplot(5,4,12)
sns.histplot(data.QS_BATHROOM, linewidth=0,kde=True)

plt.subplot(5,4,13)
sns.histplot(data.QS_BEDROOM, linewidth=0,kde=True)

plt.subplot(5,4,14)
sns.histplot(data.SALES_PRICE, linewidth=0,kde=True)

plt.subplot(5,4,15)
sns.histplot(data.TOTAL_PRICE, linewidth=0,kde=True)

plt.subplot(5,4,16)
sns.histplot(data.PROPERTY_AGE, linewidth=0,kde=True)

plt.suptitle("column wise Distribution of Data", fontsize=20)
plt.show()


# # Plotting Feature Columns vs Target Columns

# We are going to plot two types of Columns.
# 1) Numerical Columns
# 
# 2) Categorical Columns

# # Numerical data vs Target

# In[129]:


data.head(2)


# In[131]:


plt.figure(figsize=(15,7),dpi=150)
plt.subplot(2,2,1)
sns.regplot(data.INT_SQFT,data.TOTAL_PRICE, scatter_kws={"color":"black"},line_kws={"color":"red"})
plt.subplot(2,2,2)
sns.regplot(data.DIST_MAINROAD,data.TOTAL_PRICE,scatter_kws={"color":"Blue"},line_kws={"color":"red"})

plt.suptitle("Continous numerical variable VS TOTAL_PRICE", fontsize=18)
plt.show()


# In int sqft column we find good relation between the data.

# In[132]:


plt.figure(figsize=(10,5), dpi=150)
plt.subplot2grid((2,6),(0,0))
sns.regplot(data.N_BEDROOM, data.TOTAL_PRICE,scatter_kws={"color":"blue"},line_kws={"color":"red"})

plt.subplot2grid((2,6),(0,1))
sns.regplot(data.N_BATHROOM, data.TOTAL_PRICE,scatter_kws={"color":"brown"},line_kws={"color":"red"})

plt.subplot2grid((2,6),(0,2))
sns.regplot(data.N_ROOM, data.TOTAL_PRICE,scatter_kws={"color":"green"},line_kws={"color":"red"})

plt.subplot2grid((2,6),(0,3))
sns.regplot(data.QS_ROOMS, data.TOTAL_PRICE,scatter_kws={"color":"orange"},line_kws={"color":"red"})

plt.subplot2grid((2,6),(1,1))
sns.regplot(data.QS_BATHROOM, data.TOTAL_PRICE,scatter_kws={"color":"purple"},line_kws={"color":"red"})

plt.subplot2grid((2,6),(1,2))
sns.regplot(data.QS_BEDROOM, data.TOTAL_PRICE,scatter_kws={"color":"purple"},line_kws={"color":"red"})

plt.suptitle("Discrete variable vs TOTAL PRICE CHART",fontsize=18)
plt.show()


# FRom this graph
# In N_BEDROOM,N_BATHROOM and N_ROOM we find some good relations,so we will keep it.
# 
# 
# IN QS_ROOM,QS_BATHROOM and QS_BEDROOM we didnt find any relations we going to drop it.

# In[58]:


data.head(2)


# In[134]:


data['BUILDTYPE']


# # Categorical data Vs Target

# In[135]:


plt.figure(figsize=(20, 10))
plt.subplot2grid((2,6),(0,0),colspan=2)
sns.barplot(x=data.AREA,y=data.TOTAL_PRICE,order=data.groupby('AREA')['SALES_PRICE'].mean().reset_index().sort_values('SALES_PRICE')['AREA'])

plt.subplot2grid((2,6),(0,2),colspan=2)
sns.barplot(x=data.SALE_COND,y=data.TOTAL_PRICE,order=data.groupby('SALE_COND')['TOTAL_PRICE'].mean().reset_index().sort_values('TOTAL_PRICE')['SALE_COND'])

plt.subplot2grid((2,6),(0,4),colspan=2)
sns.barplot(x=data.MZZONE,y=data.TOTAL_PRICE,order=data.groupby('MZZONE')['TOTAL_PRICE'].mean().reset_index().sort_values('TOTAL_PRICE')['MZZONE'])

plt.subplot2grid((2,6),(1,0),colspan=2)
sns.barplot(x=data.UTILITY_AVAIL,y=data.TOTAL_PRICE,order=data.groupby('UTILITY_AVAIL')['TOTAL_PRICE'].mean().reset_index().sort_values('TOTAL_PRICE')['UTILITY_AVAIL'])

plt.subplot2grid((2,6),(1,2),colspan=2)
sns.barplot(x=data.STREET,y=data.TOTAL_PRICE,order=data.groupby('STREET')['TOTAL_PRICE'].mean().reset_index().sort_values('TOTAL_PRICE')['STREET'])

plt.subplot2grid((2,6),(1,4),colspan=2)
sns.barplot(x=data.PARK_FACIL,y=data.TOTAL_PRICE,order=data.groupby('PARK_FACIL')['TOTAL_PRICE'].mean().reset_index().sort_values('TOTAL_PRICE')['PARK_FACIL'])

plt.show()


# Conclusions:
# In all the above columns, we sort the columns in respect of total price and we are finding a good linear ordinal relations.
# So we will perform encoding techniques. 
# 

# # ENCODING

# In[136]:


print(data['STREET'].unique())


# # We are going to apply label encoder
# #We are going to map the value to the columns

# In[137]:


#Encoding the area column orderly#
data.AREA=data.AREA.map({'Karapakkam':1,'Anna Nagar':2,'Adyar':3,'Velachery':4,'Chrompet':5,'KK Nagar':6,'T Nagar':7})
#Encoding SALE_COND column
data.SALE_COND=data.SALE_COND.map({'AbNormal':1,'Family':2,'Partial':3,'AdjLand':4,'Normal Sale':5})
#Encoding MZZONE column
data.MZZONE=data.MZZONE.map({'A':1,'RH':2,'RL':3,'I':4,'C':5,'RM':6})
#Encoding UTILITY_AVAIL column
data.UTILITY_AVAIL=data.UTILITY_AVAIL.map({'AllPub':1,'ELO':2,'NoSeWa':3})
#Encoding STREET column
data.STREET=data.STREET.map({'Paved':1,'Gravel':2,'No Access':3})
#Encoding PARK_FACIL column
data.PARK_FACIL=data.PARK_FACIL.map({'Yes':1,'No':2})


# In[138]:


#Encoding BUILDTYPE column
data.BUILDTYPE=data.BUILDTYPE.map({'Commercial':1,'Others':2,'House':3})


# In[139]:


print(data['BUILDTYPE'].unique())


# In[201]:


data.AREA=data.AREA.map({'Karapakkam':1,'Anna Nagar':2,'Adyar':3,'Velachery':4,'Chrompet':5,'KK Nagar':6,'T Nagar':7})


# In[140]:


print(data['SALE_COND'].unique())
print(data['MZZONE'].unique())
print(data['UTILITY_AVAIL'].unique())
print(data['STREET'].unique())
print(data['PARK_FACIL'].unique())


# In[141]:


data=data.reindex(columns=['AREA','INT_SQFT','SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL','STREET','N_BEDROOM','N_BATHROOM','PARK_FACIL','N_ROOM','QS_ROOMS','QS_BATHROOM','QS_BEDROOM','PARK_FACIL'
                           
                           ,'MZZONE','QS_OVERALL','REG_FEE','SALES_PRICE','PROPERTY_AGE','TOTAL_PRICE'])


# In[202]:


data.head()


# In[143]:


plt.figure(figsize=(18,10), dpi=150)
sns.heatmap(data.corr(method='pearson'), annot=True, linewidth=0.2, cmap='coolwarm');


# In[144]:


data.head()


# # Removing unwanted columns

# In[204]:


df=data.copy()
df.drop(['REG_FEE','QS_OVERALL','QS_ROOMS','QS_BATHROOM','QS_BEDROOM'],axis=1,inplace=True)


# In[213]:


df


# # Splitting data to features and target

# In[214]:


train=['AREA','INT_SQFT','SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL','STREET','N_BEDROOM','N_BATHROOM','PARK_FACIL','N_ROOM','PARK_FACIL','MZZONE','PROPERTY_AGE']
Target1=['SALES_PRICE']
Target2=['TOTAL_PRICE']
input = df[train].copy()
target1 = df[Target1].copy()
target2 = df[Target2].copy()


# In[194]:


target2.head()


# # Splitting data for training and Testing 

# In[215]:


X_train, X_test, y_train, y_test = train_test_split(input, target1, test_size=0.2, random_state = 7)


# In[216]:


X_train


# In[217]:


pd.DataFrame(X_train, columns=X_train.columns).plot.box(figsize=(20,5),rot=90)
plt.show()


# In[218]:


mm=MinMaxScaler().fit(X_train)
X_train_mm = mm.transform(X_train)
X_train_mm = pd.DataFrame(X_train_mm, columns=X_train.columns)
X_test_mm = mm.transform(X_test)
X_test_mm = pd.DataFrame(X_test_mm, columns=X_test.columns)
X_train_mm.plot.box(figsize=(20,5), rot=90)
plt.show()


# In[220]:


ss = StandardScaler().fit(X_train)
X_train_ss = ss.transform(X_train)
X_train_ss = pd.DataFrame(X_train_ss, columns=X_train.columns)
X_test_ss = ss.transform(X_test)
X_test_ss = pd.DataFrame(X_test_ss, columns=X_test.columns)
X_train_ss.plot.box(figsize=(20,5), rot=90)
plt.show()


# # Model Training LINEAR REGRESSION

# In[237]:


lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test
                    
print('R2- SCORE:', metrics.r2_score(y_test,y_pred))
lr.fit(X_train_mm,y_train)
y_predlrmm=lr.predict(X_test_mm)
print('R2- SCORE(Minmaxscaled):', metrics.r2_score(y_test,y_predlrmm))
lr.fit(X_train_ss, y_train)
y_predlrss = lr.predict(X_test_ss)
print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_predlrss))


# # DECISION TREE

# In[248]:


dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('R2- SCORE:', metrics.r2_score(y_test,y_pred_dt))
dt.fit(X_train_mm, y_train)
y_pred_dtmm = dt.predict(X_test_mm)
print('R2- SCORE(Minmaxscaled):', metrics.r2_score(y_test,y_pred_dtmm))

dt.fit(X_train_ss, y_train)
y_pred_dtss = dt.predict(X_test_ss)
print('R2- SCORE(Standardscaler):', metrics.r2_score(y_test,y_pred_dtss))


# Here we concluded that we applllied Two algorithm linear regression and Decision Tree 
# and we find good R2 score in Decision Tree.

# In[ ]:





# In[ ]:




