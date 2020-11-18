#!/usr/bin/env python
# coding: utf-8

# In[4]:


#데이터 라벨링

import pandas as pd
contents = pd.read_csv('./NewsResult_20200806-20201106.csv',index_col=0)


# In[ ]:





# In[5]:


contents


# In[6]:


#len(contents):20,000
contents=contents['본문']
print(contents)


# In[7]:


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


# In[8]:



contents = contents.reset_index(drop = True)
for idx in contents.index:
    contents[idx] = clean_str(contents[idx])


# In[9]:


#인덱스|본문|긍부정 dataset 생성
contents.columns=['idx','content']
print(contents.columns)
print(contents.shape)
contents=contents.to_frame()


# In[10]:


contents['pos_neg']=None
contents.head(10)


# In[11]:


display(contents)


# In[15]:


print(type(contents))
print(contents.columns)
print(display(contents['본문'].head(10)))


# In[ ]:





# In[ ]:





# In[ ]:


#형태소 분석기

'''
한국어 형태소분석기 KoNLPy를 사용
'''


# In[16]:


get_ipython().system(' pip install konlpy')


# In[17]:


from konlpy.tag import Okt

okt=Okt()
text=u'나는 오늘 배가 고프지 않아'
print(okt.pos(text))
print('***********')
print(okt.nouns(text))


# In[ ]:


#Emolex 긍부정


# In[ ]:





# In[ ]:


#Emolex 8감정


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

