! pip install wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
text= "밥 먹이 밥 먹이 도시락"
wordcloud=WordCloud(font_path='./NanumSquare_acB.ttf',max_font_size=100).generate(text)

fig=plt.figure()
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
