#!/usr/bin/env python
# coding: utf-8

# In[29]:


#데이터 라벨링

import pandas as pd
contents = pd.read_csv('./NewsResult_20200806-20201106.csv',index_col=0)


# In[30]:


#contents


# In[31]:


#len(contents):20,000
contents=contents['본문']
print(contents)


# In[32]:


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


# In[33]:



contents = contents.reset_index(drop = True)
for idx in contents.index:
    contents[idx] = clean_str(contents[idx])


# In[34]:


#인덱스|본문|긍부정 dataset 생성
contents.columns=['idx','content']
print(contents.columns)
print(contents.shape)
contents=contents.to_frame()


# In[35]:


'''
contents['emo2']=None
contents['emo8']=None
'''
#contents.head(10)


# In[37]:


display(contents)


# In[38]:


print(type(contents))
print(contents.columns)
print(display(contents['본문'].head(10)))
dataset=contents


# In[39]:


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


# In[40]:


#Emolex 긍부정


# In[41]:


lexicon = pd.read_csv('./KO_NRC-emotion-Lexicon-v0.92.csv',index_col=0)

#display(lexicon.head(30))


# In[42]:


#index전처리(번역 잘 안된 부분 제거, 공백 제거)
lexicon=lexicon.drop(['NO TRANSLATION'])
#이게 오래걸리더라
for idx in lexicon.index:
    lexicon.rename(index={idx:idx.replace(' ','')}, inplace=True)
#display(lexicon.head(30))


# In[43]:


print(type(lexicon))
display(lexicon)
print(lexicon.columns)


# In[44]:


lex2=lexicon[['Positive','Negative']]
lex2.astype({'Positive':'int'},{'Negative':'int'})
display(lex2)


# In[45]:


lex2.head(20)


# In[46]:


'''
for k in range(len(lex2)):
    print(lex2.iloc[k,0])
'''


# In[47]:


lex2.index[0]


# In[48]:


#2감정 분석
#Run Time: 12 hours

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
print(res)


# In[22]:


'''
pre_texts 사용설명
[ 
    ['나','는','어제','교자만두','를','먹었어']  #뉴스기사1
    ,['부럽','겠지','민태','야']                 #뉴스기사2
    ,['히힉']                                    #뉴스기사3

]
'''


# In[49]:


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


# In[ ]:


#Emolex 8감정


# In[50]:


lex8=lexicon[['Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']]
display(lex8)


# In[57]:


lex8.head(20)


# In[65]:


#8감정 분석
#2감정을 쓰고 남은 data를 이용해 더 적은시간으로 돌아가도록 설정했습니다.
#그러므로, 꼭 2감정 학습을 돌린 이후에 8감정을 실행하시길 바랍니다.

res=[0,0,0,0,0,0,0,0,0] #Anger,Anticipation,Disgust,Fear,Joy,Sadness,Surprise,Trust,Moderation | 전체 데이터의 8감정+Moderation
for i in range(20):   # 이거 잘 돌아가는지 확인 못함 안되면 range(len(dataset))이렇게 바꿔
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
    
print(res)


# In[66]:


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

