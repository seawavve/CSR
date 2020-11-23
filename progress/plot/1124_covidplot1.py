#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from pandas import Series
from matplotlib import pyplot as plt

data = pd.read_csv('./covid19_trend.csv',index_col=0)
print(data.columns)


# In[20]:


idx=Series(data['stateDt'])
idx=list(idx)


# In[30]:


x=range(95)
y=range(95)
plt.plot(x,y)
plt.plot([0, 1, 2, 3, 4], [0, 1, 8, 27, 64])
plt.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])
plt.legend(['decideCnt', 'examCnt','deathCnt'])

plt.xlabel('date')
plt.ylabel('population')
plt.show()


# In[ ]:




