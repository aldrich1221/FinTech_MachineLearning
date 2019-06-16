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


###找出只有一筆資料的人
oneDataUser=list()
for userid in UniqueID:
    # print(np.where(df_terms['Unique ID']==userid))
    # print(userid,len(np.where(df_terms['Unique ID']==userid)[0]))
    if len(np.where(df_terms['Unique ID']==userid)[0])==1:
       oneDataUser.append(userid) 
names = {}          
relationships = {}  
lineNames = [] 
AllText=[]
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
Choose=list()
#for word in ['推','說','客戶','開戶','想','買','會','說明','基金','月','收到','表單','萬','人民','來','手機','銀行','追','寄回']:
for word in ['推','說','想','買','說','月']:
    for i in range(len(lineNames)):
        #print(word,"vs",lineNames[i])
        if word in lineNames[i]:
            Choose.append(UniqueID[i])

#print(Choose)
print(set(Choose))


#print(lineNames[0])






import sys
import gensim
import numpy as np
 
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
 
TaggededDocument = gensim.models.doc2vec.TaggedDocument
x_train = []

for i, text in enumerate(lineNames):
        
        
        document = TaggededDocument(text, tags=[i])
        x_train.append(document)

#print(x_train[2])

model_dm = Doc2Vec(x_train,min_count=1, window = 3, size = 100, sample=1e-3, negative=5, workers=4)
model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=100)

wordvector=[]
for text, label in x_train:
    vector = model_dm.infer_vector(text)
    wordvector.append(vector)
    i += 1


print(wordvector[0])
kmeans_model = KMeans(n_clusters=2)
kmeans_model.fit(wordvector)
#labels= kmean_model.predict(infered_vectors_list[0:100])
#cluster_centers = kmean_model.cluster_centers_



X = kmeans_model.fit(wordvector)
labels=kmeans_model.labels_.tolist()
#l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
pca = PCA(n_components=3).fit(wordvector)
datapoint = pca.transform(wordvector)
print("datashape:",datapoint.shape)
import matplotlib.pyplot as plt

plt.figure
label1 = ['#FFFF00', '#008000', '#0000FF', '#800080']
color = [label1[i] for i in labels]

plt.subplot(3,1,1)
#plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='x', s=150, c='#000000')
plt.xlim([-12,12])
plt.ylim([-12,12])
plt.xlabel("PC1")
plt.ylabel("PC2")
for i, word in enumerate(UniqueID):
    #print(word)
    #plt.annotate(i, xy=(datapoint[i, 0], datapoint[i, 1]))
    if labels[i]==0:
        thiscolor='r'
    elif labels[i]==1:
        thiscolor='b'
    plt.text(datapoint[i, 0], datapoint[i, 1], word,
         fontdict={'size': 5, 'color': thiscolor})

plt.subplot(3,1,2)
#plt.scatter(datapoint[:, 1], datapoint[:, 2], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 1], centroidpoint[:, 2], marker='x', s=150, c='#000000')
plt.xlim([-12,12])
plt.ylim([-12,12])
plt.xlabel("PC2")
plt.ylabel("PC3")
for i, word in enumerate(UniqueID):
    #print(word)
    #plt.annotate(i, xy=(datapoint[i, 0], datapoint[i, 1]))
    if labels[i]==0:
        thiscolor='r'
    elif labels[i]==1:
        thiscolor='b'
    plt.text(datapoint[i, 1], datapoint[i, 2], word,
         fontdict={'size': 5, 'color': thiscolor})
plt.subplot(3,1,3)
#plt.scatter(datapoint[:, 0], datapoint[:, 2], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 2], marker='x', s=150, c='#000000')
plt.xlim([-12,12])
plt.ylim([-12,12])
plt.xlabel("PC1")
plt.ylabel("PC3")
for i, word in enumerate(UniqueID):
    #print(word)
    #plt.annotate(i, xy=(datapoint[i, 0], datapoint[i, 1]))
    if labels[i]==0:
        thiscolor='r'
    elif labels[i]==1:
        thiscolor='b'
    plt.text(datapoint[i, 0], datapoint[i, 2], word,
         fontdict={'size': 5, 'color': thiscolor})
#print(list(model.wv.vocab))
#words = list(model.wv.vocab)
print(len(UniqueID),len(wordvector))
ans0=list()
ans1=list()
for i in range(len(UniqueID)):
    if labels[i]==0:
        ans0.append(UniqueID[i])
    elif labels[i]==1:
        ans1.append(UniqueID[i])


ans1.sort()
print("ANS0:",ans0)
#print("ANS1:",ans1)
print("-------------------------")
#print(set(ans0)-set(oneDataUser))
print(oneDataUser)

# # plt.show()


