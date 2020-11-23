#!/usr/bin/env python
# coding: utf-8

# In[3]:


#전처리된 2감정lexicon
#결과물은 preprocessed_lexicon2.csv 형태로 저장됩니다.
#Run Time : 5min
# 14,182 words > 4,001 words


import pandas as pd

lex2 = pd.read_csv('./KO_NRC-emotion-Lexicon-v0.92.csv',index_col=0)
print(len(lex2))
lex2=lex2[['Positive','Negative']]
lex2.astype({'Positive':'int'},{'Negative':'int'})



#번역 잘 안된 부분 제거
lex2=lex2.drop(['NO TRANSLATION'])

#index공백 제거
for idx in lex2.index:
    lex2.rename(index={idx:idx.replace(' ','')}, inplace=True)
    
#index 중복데이터 제거
lex2['rownum']=lex2.index
lex2=lex2.drop_duplicates(subset='rownum',keep='first')
#print(lex2.index.is_unique)
lex2.drop(['rownum'],axis='columns',inplace=True)

#(0,0) (1,1)인 무의미 데이터 제거
del_list=[]
for i in range(len(lex2)):
    if (lex2.iloc[i,0]==0 and lex2.iloc[i,1]==0) or (lex2.iloc[i,0]==1 and lex2.iloc[i,1]==1):
        del_list.append(i)
lex2.drop(lex2.index[del_list],axis=0,inplace=True)

lex2.to_csv("preprocessed_lexicon2.csv", mode='w',encoding='utf-8')


# In[4]:


print(len(lex2))
display(lex2)

