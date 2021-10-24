#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime as dt
from datetime import time
import glob
import scipy
import scipy.stats
import statsmodels.api as sm
import pylab


# In[2]:


data = pd.read_csv('/Users/Ivan Junqi Wu/Desktop/EBA5002.csv')


# In[3]:


data = pd.DataFrame(data)


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


new = data[['views','likes','comment_count','Days_in_TopTrending']]


# In[7]:


new.info


# In[8]:


# in online video websites, views, comments, days on top trendings are normally related to likes.
# in this project, we aim to find the highly liked video for customers to place ads and maximize our and the customers' profits,
# we have to figure out which influence the likes in US YouTube market


# In[9]:


f, ax = plt.subplots(2,2,figsize=(20,15))
# 1st chart (likes : views)
g = sns.scatterplot(x=new['views'],y=new['likes'],data=new,ax=ax[0][0])
g.set_title('Correlation of Likes and Views')
# 2nd chart (likes : communt_count)
g1 = sns.scatterplot(x=new['comment_count'],y=new['likes'],data=new,ax=ax[0][1])
g1.set_title('Correlation of Likes and Comment Counts')
# 3rd chart (likes : Days_in_TopTrending)
g1 = sns.scatterplot(x=new['Days_in_TopTrending'],y=new['likes'],data=new,ax=ax[1][0])
g1.set_title('Correlation of Likes and Days in Top Trending')


# In[10]:


# From the scatter plots, it is seemed that views and comments have some relations with likes, while days on top trendings
# has no relation with likes.


# In[11]:


plt.figure(figsize=(15,10))
ax = sns.heatmap(new.corr(),annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[12]:


# From this correlation matrix, we can know that views had the strongly positive relations with likes
# and comments has the moderately positive relations with likes
# while the correlation for days on top trendings and likes is approximately 0, which means no relation is between them.


# In[13]:


# check the relation between likes and category.
# anova can be used for this.
# anova has three assumptions.
# before we apply anova, we first check the normality and homogeneity of variance.
# Sharopi-Wilk test is used for checking normality.
# Levene is used for checking homogeneity of variance.


# In[14]:


# we first draw the boxplot to have basic understanding of the relation between categories and likes.


# In[15]:


plt.figure(figsize=(15,10))
g = sns.boxplot(x=data['category'],y=np.log(data['likes']),data=data)
g.set_xticklabels(g.get_xticklabels(),rotation=30)
plt.title('Likes across categories')
plt.show()


# In[16]:


data1 = data[data.category=='Autos & Vehicles']


# In[17]:


scipy.stats.shapiro(data1['likes'])


# In[18]:


# p-value is < 0.05, reject H0, there is no normality.


# In[19]:


data2 = data[data.category=='Comedy']


# In[20]:


scipy.stats.shapiro(data2['likes'])


# In[21]:


# p-value is < 0.05, reject H0, there is no normality.


# In[22]:


data3 = data[data.category=='Education']


# In[23]:


scipy.stats.shapiro(data3['likes'])


# In[24]:


# p-value is < 0.05, reject H0, there is no normality.


# In[25]:


data4 = data[data.category=='Entertainment']


# In[26]:


scipy.stats.shapiro(data4['likes'])


# In[27]:


# p-value is < 0.05, reject H0, there is no normality.


# In[28]:


data5 = data[data.category=='Film & Animation']


# In[29]:


scipy.stats.shapiro(data5['likes'])


# In[30]:


# p-value is < 0.05, reject H0, there is no normality.


# In[31]:


data6 = data[data.category=='Gaming']


# In[32]:


scipy.stats.shapiro(data6['likes'])


# In[33]:


# p-value is < 0.05, reject H0, there is no normality.


# In[34]:


data7 = data[data.category=='Howto & Style']


# In[35]:


scipy.stats.shapiro(data7['likes'])


# In[36]:


# p-value is < 0.05, reject H0, there is no normality.


# In[37]:


data8 = data[data.category=='Music']


# In[38]:


scipy.stats.shapiro(data8['likes'])


# In[39]:


# p-value is < 0.05, reject H0, there is no normality.


# In[40]:


data9 = data[data.category=='News & Politics']


# In[41]:


scipy.stats.shapiro(data9['likes'])


# In[42]:


# p-value is < 0.05, reject H0, there is no normality.


# In[43]:


data10 = data[data.category=='Nonprofits & Activism']


# In[44]:


scipy.stats.shapiro(data10['likes'])


# In[45]:


# p-value is < 0.05, reject H0, there is no normality.


# In[46]:


data11 = data[data.category=='People & Blogs']


# In[47]:


scipy.stats.shapiro(data11['likes'])


# In[48]:


# p-value is < 0.05, reject H0, there is no normality.


# In[49]:


data12 = data[data.category=='Pets & Animals']


# In[50]:


scipy.stats.shapiro(data12['likes'])


# In[51]:


# p-value is < 0.05, reject H0, there is no normality.


# In[52]:


data13 = data[data.category=='Science & Technology']


# In[53]:


scipy.stats.shapiro(data13['likes'])


# In[54]:


# p-value is < 0.05, reject H0, there is no normality.


# In[55]:


data14 = data[data.category=='Shows']


# In[56]:


scipy.stats.shapiro(data14['likes'])


# In[57]:


# p-value is > 0.05, do not reject H0, there is no normality.


# In[58]:


data15 = data[data.category=='Sports']


# In[59]:


scipy.stats.shapiro(data15['likes'])


# In[60]:


# p-value is < 0.05, reject H0, there is no normality.


# In[61]:


data16 = data[data.category=='Travel & Events']


# In[62]:


scipy.stats.shapiro(data16['likes'])


# In[63]:


# p-value is < 0.05, reject H0, there is no normality.


# In[64]:


# Among all results above, only 1 result has normality and others do not have.


# In[65]:


Autos_Vehicles = data1['likes']
Comedy = data2['likes']
Education = data3['likes']
Entertainment = data4['likes']
Film_Animation = data5['likes']
Gaming = data6['likes']
Howto_Style = data7['likes']
Music = data8['likes']
News_Politics = data9['likes']
Nonprofits_Activism = data10['likes']
People_Blogs = data11['likes']
Pets_Animals = data12['likes']
Science_Technology = data13['likes']
Shows = data14['likes']
Sports = data15['likes']
Travel_Events = data16['likes']


# In[ ]:


# In addtion, we use Q-Q plot to check normality.


# In[71]:


sm.qqplot(Autos_Vehicles, line='s')
pylab.title('Q-Q Plot for Autos & Vehicles')
pylab.show()


# In[72]:


sm.qqplot(Comedy, line='s')
pylab.title('Q-Q Plot for Comedy')
pylab.show()


# In[75]:


sm.qqplot(Education, line='s')
pylab.title('Q-Q Plot for Education')
pylab.show()


# In[76]:


sm.qqplot(Entertainment, line='s')
pylab.title('Q-Q Plot for Entertainment')
pylab.show()


# In[77]:


sm.qqplot(Film_Animation, line='s')
pylab.title('Q-Q Plot for Film & Animation')
pylab.show()


# In[78]:


sm.qqplot(Gaming, line='s')
pylab.title('Q-Q Plot for Gaming')
pylab.show()


# In[79]:


sm.qqplot(Howto_Style, line='s')
pylab.title('Q-Q Plot for Howto & Style')
pylab.show()


# In[80]:


sm.qqplot(Music, line='s')
pylab.title('Q-Q Plot for Music')
pylab.show()


# In[81]:


sm.qqplot(News_Politics, line='s')
pylab.title('Q-Q Plot for News & Politics')
pylab.show()


# In[82]:


sm.qqplot(Nonprofits_Activism, line='s')
pylab.title('Q-Q Plot for Nonprofits & Activism')
pylab.show()


# In[83]:


sm.qqplot(People_Blogs, line='s')
pylab.title('Q-Q Plot for People & Blogs')
pylab.show()


# In[84]:


sm.qqplot(Pets_Animals, line='s')
pylab.title('Q-Q Plot for Pets & Animals')
pylab.show()


# In[85]:


sm.qqplot(Science_Technology, line='s')
pylab.title('Q-Q Plot for Science & Technology')
pylab.show()


# In[86]:


sm.qqplot(Shows, line='s')
pylab.title('Q-Q Plot for Shows')
pylab.show()


# In[87]:


sm.qqplot(Sports, line='s')
pylab.title('Q-Q Plot for Sports')
pylab.show()


# In[88]:


sm.qqplot(Travel_Events, line='s')
pylab.title('Q-Q Plot for Travel & Events')
pylab.show()


# In[132]:


scipy.stats.levene(data1['likes'],data2['likes'],data3['likes'],data4['likes'],data5['likes'],data6['likes'],
                   data7['likes'],data8['likes'],data9['likes'],data10['likes'],data11['likes'],data12['likes'],
                   data13['likes'],data14['likes'],data15['likes'],data16['likes'])


# In[133]:


# p-value < 0.05, reject H0, there is no homogeneity of variance.


# In[134]:


# anova can not be applied in this case.
# nonparametric may be more suitable.
# Kruskal-Wallis H Test may be considered.
# H0: There is no difference between the medians.
# H1: At least, one median is different from others.


# In[89]:


scipy.stats.kruskal(Autos_Vehicles,Comedy,Education,Entertainment,Film_Animation,Gaming,Howto_Style,Music,News_Politics,
                   Nonprofits_Activism,People_Blogs,Pets_Animals,Science_Technology,Shows,Sports,Travel_Events)


# In[90]:


# p-value < 0.05, reject H0, there is at least one group different from others.
# Category is a important factor influece the likes.

