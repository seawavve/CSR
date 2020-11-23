from konlpy.tag import Okt
okt=Okt()

##추가 코드 Histogram
Hist = lex2.copy()
Hist['Frequency'] = 0

res=[0,0,0] #Positive,Negative,Moderative
pre_texts=[] #다음 감정 학습을 위해 2차원 배열로 전처리된 형태소를 보관

for i in range(len(dataset)):   # 이거 잘 돌아가는지 확인 못함 안되면  range(len(dataset)) or range(20)이렇게 바꿔
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
