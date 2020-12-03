#!/usr/bin/env python
# coding: utf-8

# In[1]:


#.csv파일 받아와 날짜별로 packing


# In[5]:


import pandas as pd
raw_data=pd.read_csv('./dataset_(0806-1106).csv')
raw_data.head(10)


# In[55]:


#일자| [ pos, neg , mod ] | [ e1 e2 e3 e4 e5 e6 e7 e8 ] | [확진자, 사망자, ...]
from pandas import Series
import numpy as np
data=pd.DataFrame()
data['일자']=Series(raw_data['일자'])

#일자 column
data=data.drop_duplicates(['일자'],keep='first')
date_idx_list=list(data.index)
date_idx_list.append(len(raw_data)) # 끝값 넣어주기
#print(date_idx_list)    #날짜별 인덱스 리스트 저장
data=data.reset_index(drop=True)

# emo2 [ pos, neg , mod ] column
# 날짜별 전체 뉴스의 기사를 긍부정 판단 후 소수로 확률 표현
emo2_list=['None']*93
for j in range(len(date_idx_list)-1):
    emo={'Pos':0,'Neg':0,'Mod':0}
    for i in range(date_idx_list[j],date_idx_list[j+1]):
        if raw_data.loc[i,'pos_neg']==1:emo['Pos']+=1
        elif raw_data.loc[i,'pos_neg']==-1:emo['Neg']+=1
        elif raw_data.loc[i,'pos_neg']==0: emo['Mod']+=1
    emo2_list[j]=emo
data['emo2']=emo2_list
display(data)


# In[57]:


# emo8 [ Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust ] column
emo8_list=['None']*93
for j in range(len(date_idx_list)-1):
    emo={'Anger':0,'Antic':0,'Disg':0,'Fear':0,
        'Joy':0,'Sad':0,'Sup':0,'Trust':0}
    for i in range(date_idx_list[j],date_idx_list[j+1]):
        emo['Anger']+=raw_data.loc[i,'Anger_val']
        emo['Antic']+=raw_data.loc[i,'Anticipation_val']
        emo['Disg']+=raw_data.loc[i,'Disgust_val']
        emo['Fear']+=raw_data.loc[i,'Fear_val']
        emo['Joy']+=raw_data.loc[i,'Joy_val']
        emo['Sad']+=raw_data.loc[i,'Sadness_val']
        emo['Sup']+=raw_data.loc[i,'Suprise_val']
        emo['Trust']+=raw_data.loc[i,'Trust_val']
    emo8_list[j]=emo
data['emo8']=emo8_list
display(data)


# In[ ]:


f=open('./Data.txt','w')
f.close()


# In[ ]:




