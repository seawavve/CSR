#!/usr/bin/env python
# coding: utf-8

# In[9]:


#dataset에서 제공하는 키워드로 2감정,8감정 분석
# Run Time: min
from tqdm import tqdm
import pandas as pd

raw_data = pd.read_csv('./NewsResult_20200806-20201106.csv',index_col=0)
dataset = pd.DataFrame({'본문': raw_data['본문'],'키워드':raw_data['키워드']})
dataset = dataset.reset_index(drop = True)

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

#인덱스|본문 전처리
for idx in contents.index:
    dataset['본문'][idx] = clean_str(dataset['본문'][idx])

#긍부정키를 따로 나눔
lex2 = pd.read_csv('./preprocessed_lexicon2.csv',index_col=0)
pos_key_cheack = lex2['Positive']  == 1
neg_key_cheack = lex2['Negative']  == 1

pos_keys = lex2[pos_key_cheack].index
neg_keys = lex2[neg_key_cheack].index
mod_keys = lex2[~pos_key_cheack & ~neg_key_cheack].index


from konlpy.tag import Okt
from tqdm import tqdm
okt=Okt()

##추가 코드 Histogram
Hist = lex2.copy()
Hist['Frequency'] = 0

res=[0,0,0] #Positive,Negative,Moderative
c = 0

for i in tqdm(range(len(dataset))):   # 이거 잘 돌아가는지 확인 못함 안되면  range(len(dataset)) or range(20)이렇게 바꿔
    text=''
    text=dataset.loc[i,'키워드']
    pos=0
    neg=0
    for j in text.split(','): #형태소가 사전과 일치하고 긍부정 중 1이 있으면 count
        c = 0 # 데이터가 제대로 count했는지 확인하는 부분
        if j in pos_keys:
            pos+=1
            c = 1
            Hist.loc[j]['Frequency'] += 1
        if j in neg_keys:
            neg+=1
            if c == 0:
                Hist.loc[j]['Frequency'] += 1
    dataset.loc[i,'pos_val'] = pos
    dataset.loc[i,'neg_val'] = neg
    if pos>neg:
        dataset.loc[i,'pos_neg']=1
        if neg == 0:
            dataset.loc[i,'P&N RATIO'] = pos
        else:
            dataset.loc[i,'P&N RATIO'] = pos/neg
        res[0]+=1
    elif neg>pos:
        dataset.loc[i,'pos_neg']=-1
        if pos == 0:
            dataset.loc[i,'P&N RATIO'] = neg
        else:
            dataset.loc[i,'P&N RATIO'] = -(neg/pos)
        res[1]+=1
    else:#중립
        dataset.loc[i,'pos_neg']=0
        dataset.loc[i,'P&N RATIO'] = 0
        res[2]+=1
print(res)

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
dataset.to_csv("dataset_pos_neg.csv",mode='w',encoding='utf-8')
dataset.to_csv("dataset_pos_neg(UTF-8-SIG).csv",mode='w',encoding='utf-8-sig')

###데이터훈련
import keras
from keras.layers import Embedding,Dense,LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer 

max_len=2000

data = pd.read_csv("./Keyword_Dataset/dataset_pos_neg.csv")

X = data['키워드']
Y = data['pos_neg']
X_train = X[:1800]
Y_train = Y[:1800]
X_test  = X[1800:2000]
Y_test  = Y[1800:2000]

max_words = 35000 
tokenizer = Tokenizer(num_words = max_words) 
tokenizer.fit_on_texts(X_train) 
X_train = tokenizer.texts_to_sequences(X_train) 
X_test = tokenizer.texts_to_sequences(X_test)

X_train=pad_sequences(X_train,maxlen=max_len)
X_test=pad_sequences(X_test,maxlen=max_len)

#model의 레이어,Hparameter,optimizer를 바꿔서 실험할 수 있음
#학습그래프그리기
#EarlyStopping,ModelCheckPoint기법 적용하기

model=Sequential()
model.add(Embedding(max_words,100))
model.add(LSTM(128))
model.add(Dense(1,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,Y_train,epochs=10,batch_size=10,validation_split=0.1) 
#전체데이터에서 10%만 validation_data로 활용
print('accuracy:{:.2f}'.format(model.evaluate(X_test,Y_test)[1]))


# In[ ]:


#+)
import numpy as np

