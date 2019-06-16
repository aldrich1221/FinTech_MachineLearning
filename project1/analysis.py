import pandas as pd
import numpy as np
from datetime import datetime
import jieba.analyse
from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import jieba
import codecs
import jieba.posseg as pseg




df = pd.read_csv('tradingdata.csv')

UserID=df['編號'].unique()

for user in UserID:

	index=np.where(df['編號']==user)[0]
	print(index)
	EachUserFrame=df.loc(user)
	print(EachUserFrame)
	print("----")
# fund=df['基金簡稱']
print(fund)