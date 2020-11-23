#!/usr/bin/env python
# coding: utf-8

# In[6]:


#전처리된 8감정lexicon
#결과물은 preprocessed_lexicon8.csv 형태로 저장됩니다.
# Run Time : 5min
# 14,182 words > 3,275 words


import pandas as pd

lex8 = pd.read_csv('./KO_NRC-emotion-Lexicon-v0.92.csv',index_col=0)
print(len(lex8))
lex8=lex8[['Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']]


#번역 잘 안된 부분 제거
lex8=lex8.drop(['NO TRANSLATION'])

#index공백 제거
for idx in lex8.index:
    lex8.rename(index={idx:idx.replace(' ','')}, inplace=True)
    
#index 중복데이터 제거
lex8['rownum']=lex8.index
lex8=lex8.drop_duplicates(subset='rownum',keep='first')
#print(lex8.index.is_unique)
lex8.drop(['rownum'],axis='columns',inplace=True)

#(0,0,0, ...) (1,1,1, ...)인 무의미 데이터 제거
del_list=[]
for i in range(len(lex8)):
    if sum(lex8.iloc[i])==0 or sum(lex8.iloc[i])==8:
        del_list.append(i)
lex8.drop(lex8.index[del_list],axis=0,inplace=True)

lex8.to_csv("preprocessed_lexicon8.csv", mode='w',encoding='utf-8')


# In[7]:


print(len(lex8))
display(lex8)


# In[ ]:




