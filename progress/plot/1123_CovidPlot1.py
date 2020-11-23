#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup
import requests


# In[2]:


url='http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson?serviceKey=nJyjaOJ8GENB%2F2nQSLsVCRkTZDsj6wUbx6iqEHNVH6I2IghnyySx3JDrp2JWMyqHG%2BWa0Y21QBVkjg%2Fr4OzD9w%3D%3D&pageNo=1&numOfRows=10&startCreateDt=20200310&endCreateDt=20200315'

req=requests.get(url)                         #공공API접근
html=req.text
soup=BeautifulSoup(html,'html.parser')


# In[3]:


print(soup)


# In[6]:


'''
기준일:stateDt (index)
확진자수:decideCnt
격리해제수:clearCnt
사망자수:deathCnt
검사진행수:examCnt
치료중환자수:careCnt
결과음성수:resultNegCnt
누적확진률:accDefRate
'''


# In[7]:


def gettext(html):#text만 추출
    text_list=[]
    for line in html:                      
      text=line.get_text()
      text_list.append(text)
    return text_list


# In[9]:


stateDt_list=[]
html_stateDt=soup.findAll('statedt')
print(html_stateDt)
stateDt_list.append(gettext(html_stateDt))
print(stateDt_list)


# In[ ]:




