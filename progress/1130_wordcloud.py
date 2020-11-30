#DataFrame을 dictionary형태로 변환
#Positive WordCloud
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv("./Keyword_Dataset/Positive_Keyword.csv")
pos_text=''
#display(data)
cloud_dic=data.set_index('Korean (ko)').to_dict()['Frequency']
#print(cloud_dic)
keyword=wordcloud.generate_from_frequencies(cloud_dic)
array=keyword.to_array()

plt.figure(figsize=(10,10))
plt.imshow(array,interpolation='bilinear')
plt.axis('off')
plt.show()

#Negative WordCloud
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv("./Keyword_Dataset/Negative_Keyword.csv")
neg_text=''
cloud_dic=data.set_index('Korean (ko)').to_dict()['Frequency']
#print(cloud_dic)
keyword=wordcloud.generate_from_frequencies(cloud_dic)
array=keyword.to_array()

plt.figure(figsize=(10,10))
plt.imshow(array,interpolation='bilinear')
plt.axis('off')
plt.show()

#Desktop/virus 이미지 마스킹
