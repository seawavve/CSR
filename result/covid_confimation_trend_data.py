#코로나 확진자 추이 크롤링
# Run Time: 0 min

import pandas as pd
from bs4 import BeautifulSoup
import requests

'''
기준일:stateDt (index)
확진자수:decideCnt
격리해제수:clearCnt
사망자수:deathCnt
검사진행수:examCnt
치료중환자수:careCnt
결과음성수:resultNegCnt
누적확진률:accDefRate
stateDt,decideCnt,clearCnt,deathCnt,examCnt,careCnt,resnegCnt,accdefRate
'''

#text만 추출하는 함수
def gettext(html):
    text_list=[]
    for line in html:
        text=line.get_text()
        text_list.append(text)
    return text_list

stateDt_list=[]
decideCnt_list=[]
clearCnt_list=[]
deathCnt_list=[]
examCnt_list=[]
careCnt_list=[]
resnegCnt_list=[]
accdefRate_list=[]

for i in range(1):
    url='http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson?serviceKey=nJyjaOJ8GENB%2F2nQSLsVCRkTZDsj6wUbx6iqEHNVH6I2IghnyySx3JDrp2JWMyqHG%2BWa0Y21QBVkjg%2Fr4OzD9w%3D%3D&pageNo='+str(i)+'&numOfRows=10&startCreateDt=20200806&endCreateDt=20201106'

    req=requests.get(url)                         #공공API접근
    html=req.text
    soup=BeautifulSoup(html,'html.parser')
    #기준일
    html_stateDt=soup.findAll('statedt')
    stateDt_list.append(gettext(html_stateDt))

    #확진자수
    html_decideCnt=soup.findAll('decidecnt')
    decideCnt_list.append(gettext(html_decideCnt))

    #격리해제수
    html_clearCnt=soup.findAll('clearcnt')
    clearCnt_list.append(gettext(html_clearCnt))

    #사망자수
    html_deathCnt=soup.findAll('deathcnt')
    deathCnt_list.append(gettext(html_deathCnt))

    #검사진행수
    html_examCnt=soup.findAll('examcnt')
    examCnt_list.append(gettext(html_examCnt))

    #치료중인환자수
    html_careCnt=soup.findAll('carecnt')
    careCnt_list.append(gettext(html_careCnt))

    #결과음성수
    html_resnegCnt=soup.findAll('resutlnegcnt')
    resnegCnt_list.append(gettext(html_resnegCnt))

    #누적확진률
    html_accdefRate=soup.findAll('accdefrate')
    accdefRate_list.append(gettext(html_accdefRate))

stateDt_list = sum(stateDt_list, [])
decideCnt_list = sum(decideCnt_list, [])
clearCnt_list = sum(clearCnt_list, [])
deathCnt_list = sum(deathCnt_list, [])
examCnt_list = sum(examCnt_list, [])
careCnt_list = sum(careCnt_list, [])
resnegCnt_list = sum(resnegCnt_list, [])
accdefRate_list = sum(accdefRate_list, [])

result=[]
for stateDt,decideCnt,clearCnt,deathCnt,examCnt,careCnt,resnegCnt,accdefRate in zip(stateDt_list,decideCnt_list,clearCnt_list,deathCnt_list,examCnt_list,careCnt_list,resnegCnt_list,accdefRate_list):       
    row_data=[stateDt,decideCnt,clearCnt,deathCnt,examCnt,careCnt,resnegCnt,accdefRate]
    result.append(row_data)

covid_df=pd.DataFrame(result,columns=['stateDt','decideCnt','clearCnt','deathCnt','examCnt','careCnt','resnegCnt','accdefRate']) 
covid_df.to_csv('covid19_trend.csv')
#display(covid_df)
