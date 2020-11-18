#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install konlpy')


# In[3]:


from konlpy.tag import Hannanum

hannanum=Hannanum()
print(hannanum.analyze(u'히히 하하 신난다'))


# In[ ]:


'''
analyze(text): 사전검색차트, 분류되지않은 용어 두 부분으로 분할구성
morphs(text): 형태소 반환
nouns(text):명사 반환
pos(text):품사 부착하여 반환
'''

