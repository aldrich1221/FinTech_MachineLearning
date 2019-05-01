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


UniqueID=df_terms['Unique ID'].unique()

#print(UniqueID)


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

MyText=np.array(len(UniqueID))
#print(MyText.shape)
for i in range(len(df_terms['客戶事件描述'])):
    try:
        poss = jieba.cut(df_terms['客戶事件描述'][i], cut_all = False)
        lineNames.append([])
        #MyText.append([])
        for w in poss:
            if w not in stopWords:
                #lineNames[-1].append(w) 

                lineNames[np.min(np.where(df_terms['Unique ID'][i]==UniqueID))].append(w) 
                #print(np.min(np.where(df_terms['Unique ID'][i]==UniqueID)))
                #MyText[np.where(df_terms['Unique ID'][i]==UniqueID)].append(w)
                #print("W:",w)       
            if names.get(w) is None and w not in stopWords:    
                relationships[w] = {}            
    except:
        pass

#print(MyText)
lineNames = list(filter(lambda a: a != [], lineNames))
#print(lineNames)


# coding=utf-8
# import jieba 
# from hanziconv import HanziConv

# fileTrainRead = []
# with open('./mytest_corpus.txt') as fileTrainRaw:
#   for line in fileTrainRaw:
#       fileTrainRead.append(HanziConv.toTraditional(line)) 


#import word2vec
from gensim.models import word2vec


model=word2vec.Word2Vec(lineNames,size=300)
# model=word2vec.load(myword2vecmodel.bin)
model.train(lineNames, total_examples=len(lineNames), epochs=1)
#print(model.vectors)
#print(model.most_similar(['玉山']))
vector=model.wv
print(vector)


terms = ['亞太複合債','IPO','nn','NN','基金','新興市場','環球','新興債','優惠','Q1','Q3','債','中國','亞高']  
df = {}
for t in terms:
   
    try:
    	df[t] = [term for term, score in model.most_similar(t)]  
    except:
   	
   		print("不在詞庫")

df = pd.DataFrame(df)
print(df)

#IPO   NN   基金   環球  新興債   優惠   Q1   Q3    債   中國
#
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# pca = PCA(n_components=2)
# X = model[model.wv.vocab]
# result = pca.fit_transform(X)
# # 可视化展示
# plt.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
# plt.show()







keys=model.wv.vocab.keys()
    

wordvector=[]
for key in keys:
    wordvector.append(model[key])

#分类
# clf = KMeans(n_clusters=5)
# s = clf.fit(wordvector)
kmeans_model = KMeans(n_clusters=3, max_iter=100)

X = kmeans_model.fit(wordvector)
labels=kmeans_model.labels_.tolist()
#l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(wordvector)
datapoint = pca.transform(wordvector)
import matplotlib.pyplot as plt

plt.figure
label1 = ['#FFFF00', '#008000', '#0000FF', '#800080']
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='x', s=150, c='#000000')

#print(list(model.wv.vocab))
words = list(model.wv.vocab)
for i, word in enumerate(words):
	print(word)
	#plt.annotate(i, xy=(datapoint[i, 0], datapoint[i, 1]))
	plt.text(datapoint[i, 0], datapoint[i, 1], word,
         fontdict={'size': 16, 'color': 'r'})
plt.show()

