#!/usr/bin/env python
# coding: utf-8

# In[1]:


#전처리를 거친 뉴스본문으로 2감정, 8감정 분석
# Run Time: ? min 

import pandas as pd
contents = pd.read_csv('./NewsResult_20200806-20201106.csv',index_col=0)
contents=contents['본문']

#텍스트 전처리
import re
def clean_str(string):
  string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)     
  string = re.sub(r"\'s", " \'s", string) 
  string = re.sub(r"\'ve", " \'ve", string) 
  string = re.sub(r"n\'t", " n\'t", string)  
  string = re.sub(r"\'re", " \'re", string) 
  string = re.sub(r"\'d", " \'d", string) 
  string = re.sub(r"\'ll", " \'ll", string) 
  string = re.sub(r",", " , ", string) 
  string = re.sub(r"!", " ! ", string) 
  string = re.sub(r"\(", " \( ", string) 
  string = re.sub(r"\)", " \) ", string) 
  string = re.sub(r"\?", " \? ", string) 
  string = re.sub(r"\s{2,}", " ", string)
  string=re.sub(r"\'{2,}", "\' ",string)
  string=re.sub(r"\' ", "",string)

  return string.lower()

#인덱스|본문 dataset 생성
contents = contents.reset_index(drop = True)
for idx in contents.index:
    contents[idx] = clean_str(contents[idx])
contents.columns=['idx','content']
contents=contents.to_frame()
dataset=contents

#2감정 분석
lex2 = pd.read_csv('./preprocessed_lexicon2.csv',index_col=0)
from konlpy.tag import Okt
okt=Okt()

Hist = lex2.copy()
Hist['Frequency'] = 0

res=[0,0,0] #Positive,Negative,Moderative
pre_texts=[] #다음 감정 학습을 위해 2차원 배열로 전처리된 형태소를 보관
for i in range(20):   #  range(len(dataset)) or range(20)
    text=''
    text=dataset.loc[i,'본문']
    pre_text=okt.morphs(text) #형태소로 자르기
    pre_texts.append(pre_text)
    pos=0
    neg=0
    for j in range(len(pre_text)): #형태소가 사전과 일치하고 긍부정 중 1이 있으면 count
        for k in range(len(lex2)):
            if pre_text[j]==str(lex2.index[k]):
                if lex2.iloc[k,0]==1:
                    pos+=1
                if lex2.iloc[k,1]==1:
                    neg+=1
                Hist.iloc[k,2] += 1
    if pos>neg:
        dataset.loc[i,'pos_neg']=1
        res[0]+=1
    elif neg>pos:
        dataset.loc[i,'pos_neg']=-1
        res[1]+=1
    else:#중립
        dataset.loc[i,'pos_neg']=0
        res[2]+=1
#print(res)

#histogram
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(3)
y=res
plt.bar(x,y)
plt.xticks(x,['Positive','Negative','Moderative'])
plt.xlabel('Emotional Distribution')
plt.ylabel('Amount of News')
plt.show()

# 키워드 히스토그램 추출
Valid_Hist_cheack = Hist['Frequency'] > 1
Valid_Hist = Hist[Valid_Hist_cheack ]
Valid_Hist = Valid_Hist.sort_values(by='Frequency',ascending=False)


Positive_Hist_cheack = Valid_Hist['Positive'] == 1
Negative_Hist_cheack = Valid_Hist['Negative']  == 1

Positive_Hist =  Valid_Hist[Positive_Hist_cheack]
Negative_Hist =  Valid_Hist[Negative_Hist_cheack]
Moderative_Hist = Valid_Hist[ ~Positive_Hist_cheack & ~Negative_Hist_cheack]

Positive_Hist = Positive_Hist.sort_values(by='Frequency',ascending=False)
Negative_Hist = Negative_Hist.sort_values(by='Frequency',ascending=False)
Moderative_Hist = Moderative_Hist.sort_values(by='Frequency',ascending=False)

Valid_Hist.to_csv("Keyword.csv", mode='w',encoding='utf-8')
Positive_Hist.to_csv("Positive_Keyword.csv", mode='w',encoding='utf-8')
Negative_Hist.to_csv("Negative_Keyword.csv", mode='w',encoding='utf-8')
Moderative_Hist.to_csv("Moderative_Keyword.csv", mode='w',encoding='utf-8')


# In[2]:


#8감정 분석
#2감정을 쓰고 남은 data를 이용해 더 적은시간으로 돌아가도록 설정했습니다.
#그러므로, 꼭 2감정 학습을 돌린 이후에 8감정을 실행하시길 바랍니다.

lex8 = pd.read_csv('./preprocessed_lexicon8.csv',index_col=0)
res=[0,0,0,0,0,0,0,0,0] #Anger,Anticipation,Disgust,Fear,Joy,Sadness,Surprise,Trust,Moderation | 전체 데이터의 8감정+Moderation
for i in range(20):   # range(len(dataset)) or range(20)
    pre_text=pre_texts[i]
    emo8=[0,0,0,0,0,0,0,0] #뉴스 하나의 8감정+Moderation

    for j in range(len(pre_text)): #형태소가 사전과 일치하고 8감정 중 1이 있으면 count
        for k in range(len(lex8)):
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,0]==1:
                emo8[0]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,1]==1:
                emo8[1]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,2]==1:
                emo8[2]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,3]==1:
                emo8[3]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,4]==1:
                emo8[4]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,5]==1:
                emo8[5]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,6]==1:
                emo8[6]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,7]==1:
                emo8[7]+=1
    if emo8.count(0)==8:
        res[8]+=1
    else:
        news_emo=emo8.index(max(emo8))
        for r in range(8):
            if news_emo==r:res[r]+=1
#print(res)

#histogram
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(9)
y=res
ax=plt.subplot(1,1,1)
plt.bar(x,y)
plt.xticks(x,['Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust','Moderation'])
for label in ax.xaxis.get_ticklabels():
    label.set_rotation(45)
plt.xlabel('Emotional Distribution')
plt.ylabel('Amount of News')
plt.show()


# In[ ]:




