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


# In[6]:


#Emolex 긍부정


# In[12]:


lexicon = pd.read_csv('./KO_NRC-emotion-Lexicon-v0.92.csv',index_col=0)

#display(lexicon.head(30))


# In[13]:


#index전처리(번역 잘 안된 부분 제거, 공백 제거)
lexicon=lexicon.drop(['NO TRANSLATION'])
#이게 오래걸리더라
for idx in lexicon.index:
    lexicon.rename(index={idx:idx.replace(' ','')}, inplace=True)
#display(lexicon.head(30))


# In[14]:


print(type(lexicon))
display(lexicon)
print(lexicon.columns)


# In[15]:


lex2=lexicon[['Positive','Negative']]
#display(lex2)


# In[16]:


#lex2.head(20)


# In[17]:


#본문이 진짜 쌩 본문인줄 알았는데 아니었네............
#핵 오래걸ㄹㅣㅁ
from konlpy.tag import Okt
okt=Okt()

res=[0,0,0] #pos수,neg수,중간
for i in range(30): #len(dataset)
    text=''
    text=dataset.loc[i,'본문']
    pre_text=okt.morphs(text) #형태소로 자르기
    pos=0
    neg=0
    for j in range(len(pre_text)): #형태소가 사전과 일치하면 count
        for k in lex2.index:
            if pre_text[j]==str(k):
                pos+=1
            if pre_text[j]==str(k):
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


# In[ ]:





# In[ ]:





# In[ ]:


#Emolex 8감정


# In[12]:


lex8=lexicon[['Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']]
display(lex8)


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

