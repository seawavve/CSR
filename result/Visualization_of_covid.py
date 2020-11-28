
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
import matplotlib.dates as mdates
data=pd.read_csv('covid19_trend.csv')
print(data.columns)



#그래프1
date=list(Series(data['stateDt']))
x_date=[]
for i in range(len(date)):
    tmp=str(date[i])
    if i%5==0:x_date.append(tmp[4:6]+'.'+tmp[6:])
x_date.reverse()


x=range(95)
y1=list(Series(data['decCnt']))
y2=list(Series(data['clCnt']))
y3=list(Series(data['dthCnt']))
y1.reverse()
y2.reverse()
y3.reverse()

plt.plot(x,y1,'#ffa700',label='decCnt')
plt.plot(x,y2,'#008744',label='clCnt')
plt.plot(x,y3,'#d62d20',label='dthCnt')
plt.xticks(np.arange(0,95,step=5),x_date,rotation=45)

plt.legend(['confirmed','cleared','dead'])
plt.xlabel('Date')
plt.ylabel('Population')
plt.title('Changes in the number of patients')
plt.figure(figsize=(5,10))
plt.show()

#그래프2
y=list(Series(data['examCnt']))
y.reverse()
plt.bar(x,y,color='b',alpha=0.7,width=0.5)
plt.xticks(np.arange(0,95,step=5),x_date,rotation=45)
plt.title('Changes in the number of examined patients')
plt.xlabel('Date')
plt.ylabel('Population')
plt.figure(figsize=(5,10))
plt.show()
