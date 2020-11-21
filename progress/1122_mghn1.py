#!/usr/bin/env python
# coding: utf-8

# In[1]:


#데이터 라벨링

import pandas as pd
contents = pd.read_csv('./NewsResult_20200806-20201106.csv',index_col=0)


# In[2]:


#contents


# In[3]:


#len(contents):20,000
contents=contents['본문']
print(contents)


# In[4]:


#텍스트 전처리 함수
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


# In[5]:



contents = contents.reset_index(drop = True)
for idx in contents.index:
    contents[idx] = clean_str(contents[idx])


# In[6]:


#인덱스|본문|긍부정 dataset 생성
contents.columns=['idx','content']
print(contents.columns)
print(contents.shape)
contents=contents.to_frame()


# In[7]:


contents['pos_neg']=None
#contents.head(10)


# In[8]:


#display(contents)


# In[9]:


print(type(contents))
print(contents.columns)
print(display(contents['본문'].head(10)))
dataset=contents


# In[10]:


#display(dataset)


# In[11]:


#형태소 분석기 tutorial

'''
한국어 형태소분석기 KoNLPy를 사용
'''


# In[12]:


get_ipython().system(' pip install konlpy')


# In[4]:


from konlpy.tag import Okt
okt=Okt()


# In[5]:


text=u'나는 오늘 배가 고프지 않아'
print(okt.morphs(text))
print('***********')
print(okt.nouns(text))


# In[12]:


#Emolex 긍부정


# In[13]:


lexicon = pd.read_csv('./KO_NRC-emotion-Lexicon-v0.92.csv',index_col=0)

#display(lexicon.head(30))


# In[14]:


#index전처리(번역 잘 안된 부분 제거, 공백 제거)
lexicon=lexicon.drop(['NO TRANSLATION'])
#이게 오래걸리더라
for idx in lexicon.index:
    lexicon.rename(index={idx:idx.replace(' ','')}, inplace=True)
#display(lexicon.head(30))


# In[15]:


print(type(lexicon))
display(lexicon)
print(lexicon.columns)


# In[16]:


lex2=lexicon[['Positive','Negative']]
lex2.astype({'Positive':'int'},{'Negative':'int'})
display(lex2)


# In[23]:


lex2.head(20)


# In[22]:


'''
for k in range(len(lex2)):
    print(lex2.iloc[k,0])
'''


# In[24]:


lex2.index[0]


# In[ ]:


#2감정 분석

#본문이 진짜 쌩 본문인줄 알았는데 아니었네............
#핵 오래걸ㄹㅣㅁ 1:20-

from konlpy.tag import Okt
okt=Okt()

res=[0,0,0] #Positive,Negative,Moderative
for i in range(len(dataset)):   # 이거 잘 돌아가는지 확인 못함 안되면 range(20)이렇게 바꿔
    text=''
    text=dataset.loc[i,'본문']
    pre_text=okt.morphs(text) #형태소로 자르기
    pos=0
    neg=0
    for j in range(len(pre_text)): #형태소가 사전과 일치하고 긍부정 중 1이 있으면 count
        for k in range(len(lex2)):
            if pre_text[j]==str(lex2.index[k]) and lex2.iloc[k,0]==1:
                pos+=1
            if pre_text[j]==str(lex2.index[k]) and lex2.iloc[k,1]==1:
                neg+=1
    if pos>neg:
        dataset.loc[i,'pos_neg']=1
        res[0]+=1
    elif neg>pos:
        dataset.loc[i,'pos_neg']=-1
        res[1]+=1
    else:#중립
        dataset.loc[i,'pos_neg']=0
        res[2]+=1
print(res)


# In[37]:


import matplotlib.pyplot as plt
import numpy as np

x=np.arange(3)
y=res


plt.bar(x,y)
plt.xticks(x,['Positive','Negative','Moderative'])
plt.xlabel('Emotional Distribution')
plt.ylabel('Amount of News')

plt.show()


# In[ ]:





# In[ ]:


#Emolex 8감정


# In[12]:


lex8=lexicon[['Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']]
display(lex8)


# In[ ]:


lex8.astype({'Anger':'int'},{'Anticipation':'int'},{'Disgust':'int'}
                ,{'Fear':'int'},{'Joy':'int'},{'Sadness':'int'},{'Surprise':'int'},{'Trust':'int'})
display(lex8)


# In[ ]:


lex8.head(20)


# In[ ]:


#8감정 분석
#공사중


#2감정 분석할 때 사용했던 형태소 데이터를 저장해두면 8감정 분석할 때 시간이 좀 더 적게 걸리겠지
#이건 효율면의 문제니까 나중에 시간나면 고치자~~

from konlpy.tag import Okt
okt=Okt()

res=[0,0,0,0,0,0,0,0] #Anger,Anticipation,Disgust,Fear,Joy,Sadness,Surprise,Trust | 전체 데이터의 8감정
for i in range(20):   # 이거 잘 돌아가는지 확인 못함 안되면 range(len(dataset))이렇게 바꿔
    text=''
    text=dataset.loc[i,'본문']
    pre_text=okt.morphs(text) #형태소로 자르기
    emo8=[0,0,0,0,0,0,0,0] #뉴스 하나의 8감정
    
    
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
                emo[5]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,6]==1:
                emo[6]+=1
            if pre_text[j]==str(lex8.index[k]) and lex8.iloc[k,7]==1:
                emo[7]+=1
    '''
    news_emo=emo8.index(max(emo8))
    if news_emo==0:  #8감정칸에 ang 라고 써라
        dataset.loc[i,'pos_neg']=0
        res[0]+=1
    '''
print(res)


# In[ ]:





# In[ ]:





# In[ ]:


#--------------------------------------------


# In[ ]:


###긍부정시각화
#히스토그램
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
train_data['label'].value_counts().plot(kind='bar')


# In[ ]:


test_data['label'].value_counts().plot(kind='bar')


# In[ ]:


print(train_data.groupby('label').size().reset_index(name='count')) 
print(test_data.groupby('label').size().reset_index(name='count'))


# In[ ]:


###데이터훈련

from keras.layers import Embedding,Dense,LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
max_len=2000

X_train=pad_sequences(X_train,maxlen=max_len)
X_test=pad_sequences(X_test,maxlen=max_len)

#model의 레이어,Hparameter,optimizer를 바꿔서 실험할 수 있음
#학습그래프그리기
#EarlyStopping,ModelCheckPoint기법 적용하기

model=Sequential()
model.add(Embedding(max_words,100))
model.add(LSTM(128))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,Y_train,epochs=10,batch_size=10,validation_split=0.1) 
#전체데이터에서 10%만 validation_data로 활용
print('accuracy:{:.2f}'.format(model.evaluate(X_test,Y_test)[1]))


# In[ ]:


#+)
import numpy as np
predict=model.predict(X_test)
predict_labels=np.argmax(predict,axis=1)
original_labels=np.argmax(Y_test,axis=1)
for i in range(10):
  print(test_data['content'].iloc[i])
  print('원래:',original_labels[i],'예측:,'predict_labels[i])

