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

print(UniqueID)


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


# term_dic = dict()
# for sentence in lineNames:
#     for term in sentence:
#         if term not in term_dic:
#             term_dic[term] = {}

# for i in range(len(lineNames)):
#     for term1 in term_dic:
#         num = 0
#         for term2 in lineNames[i]:
#             if term1 == term2:
#                 num += 1
#         term_dic[term1][df_terms['客戶事件描述'][i]] = num


# for sentence in MyText:
#     for term in sentence:
#         if term not in term_dic:
#             term_dic[term] = {}

# for i in range(len(MyText)):
#     for term1 in term_dic:
#         num = 0
#         for term2 in MyText[i]:
#             if term1 == term2:
#                 num += 1
#         term_dic[term1][df_terms['客戶事件描述'][i]] = num


# TDM = pd.DataFrame.from_dict(term_dic)
# print(TDM)



from collections import defaultdict
wordsCount = defaultdict(int)
tfs = []

for sentence in lineNames:
	for word in sentence:
		wordsCount[word] += 1



for sentence in lineNames:
	tf = defaultdict(int)
	for word in sentence:
		if wordsCount[word] > 2:
			tf[word] = sentence.count(word) / len(sentence)
	tfs.append(tf)

#print(tfs[0])

idf = defaultdict(int)
import math
for word in wordsCount:
	count = 0
	for sentence in lineNames:
		if word in sentence:
			count += 1
	idf[word] = math.log(len(lineNames) / count)

#print(idf)

tf_idfs = []


def mysort(adict):
	items=adict.items()
	items.sort()
	return [value for key,value in items]

for tf in tfs:
	tf_idf = defaultdict()
	for word in tf:
		tf_idf[word] = tf[word] * idf[word]
	#sorted_tfidf=mysort(tf_idf)
	#tf_idfs.append(sorted_tfidf)
	tf_idfs.append(tf_idf)

print(UniqueID[0],tf_idfs[0])

print(UniqueID[1],tf_idfs[1])

print(UniqueID[2],tf_idfs[2])
print(UniqueID[3],tf_idfs[3])
print(UniqueID[4],tf_idfs[4])
# Maxlist=list()
# for i in range(len(tf_idfs)):
# 	#print(UniqueID[i],max(tf_idfs[i],key=tf_idfs[i].get))
# 	Maxlist.append(max(tf_idfs[i],key=tf_idfs[i].get()))




# Uniquelist=np.unique(Maxlist)
# #print(Maxlist)
# Countdict=defaultdict(int)
# TextDict = defaultdict(list)

# for i in range(len(Uniquelist)):
# 	#print(np.where(Uniquelist[i]==np.array(Maxlist)))
	
# 	#Countdict{Uniquelist[i]: UniqueID[np.where(Uniquelist[i]==np.array(Maxlist))]}
# 	Countdict[Uniquelist[i]]= UniqueID[np.where(Uniquelist[i]==np.array(Maxlist))]

	
# 	TextDict[Uniquelist[i]] =len(UniqueID[np.where(Uniquelist[i]==np.array(Maxlist))])

# sorted_countDict = sorted(Countdict.items(), key = lambda kv: len(kv[1]))

# # print(sorted_countDict)

# ourList = []
# for item in sorted_countDict:
# 	a = []
# 	a.append(item[0])
# 	a.append(item[1])
# 	a.append(len(item[1]))
# 	ourList.append(a)
# 	#print(Countdict)
# 	#Countlist(Maxlist.count(Uniquelist[i]))

# for item in ourList:
# 	if item[2] >= 4:
# 		print(item)





##. 亞太複合債、IPO、nn、NN、基金、新興市場、環球、新興債、優惠、Q1、Q3、債、中國、亞高


# import nltk

# tree1=nltk.Tree('NP',['aa'])
# tree2=nltk.Tree('N',['aa','bb'])
# tree3=nltk.Tree('S',[tree1,tree2])

# print(tree3)
#tree3.draw()




#print(TDM.head())


# import jieba.posseg as pseg

# cutcorpusiter = lineNames.copy()
# cutcorpus = lineNames.copy()
# cixingofword = []  # 儲存分詞後的詞語對應的詞性
# wordtocixing = []  # 儲存分詞後的詞語
# for i in range(len(lineNames)):
#     cutcorpusiter[i] = pseg.cut(lineNames[i])
#     cutcorpus[i] = ""
#     for every in cutcorpusiter[i]:   
#         cutcorpus[i] = (cutcorpus[i] + " " + str(every.word)).strip()
#         cixingofword.append(every.flag)
#         wordtocixing.append(every.word)
# # 自己造一個{“詞語”:“詞性”}的字典，方便後續使用詞性
# word2flagdict = {wordtocixing[i]:cixingofword[i] for i in range(len(wordtocixing))}
# print(word2flagdict)
