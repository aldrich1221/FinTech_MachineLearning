# -*- coding: utf-8 -*
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




df = pd.read_csv('data.csv')



#df.drop('Unnamed: 0', axis = 1, inplace = True)

for i in df.index:
    df.at[i, '資料日期'] = datetime.strptime(df.at[i, '資料日期'], '%Y%m%d %H%M%S')

df_time = pd.DataFrame()
for i in df['Unique ID'].value_counts().index:
    for j in df[df['Unique ID'] == i].reset_index().index :
        df_time.at[i, j] = df[df['Unique ID'] == i].reset_index().at[j, '資料日期']
df_time.head()

df_event = pd.DataFrame()
for i in df['Unique ID'].value_counts().index:
    for j in df[df['Unique ID'] == i].reset_index().index :
        df_event.at[i, j] = df[df['Unique ID'] == i].reset_index().at[j, '客戶事件描述']
#print(df_event.head())

stopWords = []
with open('stopwords.txt', 'r',encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)

stoplst = [' ', '\xa0']
for words in stoplst:
    stopWords.append(words)


df_terms = pd.read_csv('data.csv')
#df_terms.drop('Unnamed: 0', axis = 1, inplace = True)


# error_lst = []
# terms=[]
# for i in range(len(df_terms['客戶事件描述'])):
#     try:
#         for j in list(jieba.cut(df_terms['客戶事件描述'][i], cut_all = False)):
#             if j not in stopWords:
#                 terms.append(j)
#     except:
#         error_lst.append([i, df_terms['客戶事件描述'][i]])

# sorted(Counter(terms).items(), key=lambda x:x[1], reverse=True)[:10]


# wc = WordCloud(background_color = "white", width = 1440, height = 900, margin= 2, font_path="STHeiti Light.ttc")
# wc.generate_from_frequencies(Counter(terms))
# plt.figure(figsize = (10, 10))
# plt.imshow(wc)
# plt.axis("off")
#plt.show()


names = {}          
relationships = {}  
lineNames = []  
for i in range(len(df_terms['客戶事件描述'])):
    try:
        poss = jieba.cut(df_terms['客戶事件描述'][i], cut_all = False)
        lineNames.append([])
        for w in poss:
            if w not in stopWords:
                lineNames[-1].append(w) 
                #print("W:",w)       
            if names.get(w) is None and w not in stopWords:    
                relationships[w] = {}            
    except:
        pass

term_dic = dict()
for sentence in lineNames:
    for term in sentence:
        if term not in term_dic:
            term_dic[term] = {}

for i in range(len(lineNames)):
    for term1 in term_dic:
        num = 0
        for term2 in lineNames[i]:
            if term1 == term2:
                num += 1
        term_dic[term1][df_terms['客戶事件描述'][i]] = num

TDM = pd.DataFrame.from_dict(term_dic)
print(TDM.head())


