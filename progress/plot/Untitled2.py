#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
import matplotlib.dates as mdates
data=pd.read_csv('covid19_trend.csv')
print(data.columns)


# In[26]:


date=list(Series(data['stateDt']))


# In[28]:


x=np.arange(95)
y1=list(Series(data['decCnt']))
y2=list(Series(data['clCnt']))
y3=list(Series(data['dthCnt']))

plt.plot(x,y1,'k',label='decCnt')
plt.plot(x,y2,'g',label='clCnt')
plt.plot(x,y3,'r',label='dthCnt')

plt.legend(['decision','clear','death'])
plt.xlabel('Date')
plt.ylabel('Population')
plt.title('dsd')
plt.figure(figsize=(5,10))
plt.show()
plt.savefig('gg.png')


# In[ ]:





# In[ ]:




