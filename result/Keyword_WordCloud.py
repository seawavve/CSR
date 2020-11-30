#Positive WordCloud
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from PIL import Image

mask= np.array(Image.open("./new_virus.png"))

data=pd.read_csv("./Keyword_Dataset/Positive_Keyword.csv")
#DataFrame을 dictionary형태로 변환
cloud_dic=data.set_index('Korean (ko)').to_dict()['Frequency']

wc=WordCloud(font_path='./NanumSquare_acB.ttf',
             mask=mask,background_color='black',
             colormap='Blues'
            ).generate_from_frequencies(cloud_dic)
plt.figure(figsize=(10,10))
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

#Negative WordCloud
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from PIL import Image

mask= np.array(Image.open("./new_virus.png"))
data=pd.read_csv("./Keyword_Dataset/Negative_Keyword.csv")
#DataFrame을 dictionary형태로 변환
cloud_dic=data.set_index('Korean (ko)').to_dict()['Frequency']

wc=WordCloud(font_path='./NanumSquare_acB.ttf',
             mask=mask,background_color='black',
             colormap='autumn'
            ).generate_from_frequencies(cloud_dic)
plt.figure(figsize=(10,10))
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()
