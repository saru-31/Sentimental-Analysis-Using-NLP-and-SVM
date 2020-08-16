# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 22:04:34 2020

@author: sarve
"""
''' 

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random
import nltk
import re
import pathlib
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
text_files=np.array([])

stopwords=nltk.corpus.stopwords.words('english')
for path in pathlib.Path(r'C:\Users\sarve\Sentimental Analysis_NLP').glob('*.txt'):
    text_files=np.append(text_files,path)

stemming=PorterStemmer()
wnl=nltk.WordNetLemmatizer()
##RAW DATA PROCESSING
def data_load(files):
    doc_lines=[]
    doc_vals=[]

    text=open(files,"r")
    for line in text:
        doc_lines.append(line.split('\t')[0])
        doc_vals.append(line.split('\t')[1])
        print(line)
    
    data=pd.DataFrame()
    data['doc_lines']=doc_lines
    data['label']=doc_vals
    return data

#TOKENIZATION
def tokenizer(data):
    refined_data=pd.Series()
    token=pd.Series()
    for i in range(len(data)):
        refined_data=refined_data.set_value(i,re.sub(r'[.|,|;|/|?|*|^|:]',' ',data[i]))
        #refined_data=refined_data.set_value
        token=token.set_value(i,nltk.word_tokenize(refined_data[i]))
    return token

#TEXT_CLEANING 
#Stemming
def stemmer(tokenized_text):
    stem_text=pd.Series()
    i=0
    for tok in tokenized_text:
        pointer=[text for text in tok if not text in stopwords]
        #for corpus
        stem_text=stem_text.set_value(i,' '.join([stemming.stem(wrd) for wrd in pointer]))
        #for normal:stem_text=stem_text.set_value(i,[stemming.stem(wrd) for wrd in pointer])
        i=i+1
    return stem_text

#Lemmatization
def lemmetizer(tokenized_text):
    lem_text=pd.Series()
    i=0
    for tok in tokenized_text:
        pointer=[text for text in tok if not text in stopwords]
        #for corpus
        lem_text=lem_text.set_value(i,' '.join([wnl.lemmatize(wrd) for wrd in pointer]))
        #for normal: lem_text=lem_text.set_value(i,[wnl.lemmatize(wrd) for wrd in pointer])
        i=i+1
    return lem_text
def s_analyse(k):
    data=data_load(text_files[k])    
    random.seed(1985)
    data=data.sample(frac=1)
    data=data.reset_index()
    if (data.columns.size > 2):
        data=data.drop([data.columns[0]],axis=1)
    
    #TOKENIZATION
    
    tokenized_text1=tokenizer(data['doc_lines'])
    #TEXT_CLEANING 
        
    wordlist=lemmetizer(tokenized_text1)
    wordlist2=stemmer(tokenized_text1)
    return wordlist,data['label']
#VECTORIZATION
corpus,sentiment=s_analyse(0)
corpus2,sentiment2=s_analyse(1)
corpus3,sentiment3=s_analyse(3)

corpus=(corpus.append([corpus2,corpus3]).reset_index())
corpus=corpus.drop([corpus.columns[0]],axis=1)
sentiment=sentiment.append([sentiment2,sentiment3]).reset_index()
sentiment=sentiment.drop([sentiment.columns[0]],axis=1)
vectorizer=CountVectorizer(ngram_range=(1,2))
cv_model=vectorizer.fit_transform(corpus[0])
vocab=vectorizer.get_feature_names()
cv_model.toarray()
vec_data=pd.DataFrame(cv_model.toarray(),columns=vocab).drop(['the'],axis=1)
vec_data['Sentiment']=sentiment
for i in range(len(corpus[0])):
    vec_data=vec_data.set_value(i,'Polarity',TextBlob(corpus[0][i]).sentiment.polarity)
    vec_data=vec_data.set_value(i,'Subjectivity',TextBlob(corpus[0][i]).sentiment.subjectivity)
        
#ML-ALGO
temp=vec_data.loc[:,vec_data.columns!='Sentiment']
  
x_train,x_test,y_train,y_test=train_test_split(temp,vec_data['Sentiment'],train_size=0.85,random_state=456)
#log_reg=LogisticRegression()
svc=SVC(kernel='linear',C=1.0)
svc.fit(x_train,y_train)
#model=log_reg.fit(X=x_train,y=y_train)
y_pred=svc.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#Most positive word
posit_wrds=vec_data.loc[vec_data['Sentiment']=='1\n']
print("The most Positive word and its frequency of occurrence in total respectively:",
posit_wrds.drop(['Sentiment','Polarity','Subjectivity'],axis=1).sum(axis=0,skipna=True).max(),
posit_wrds.drop(['Sentiment','Polarity','Subjectivity'],axis=1).sum(axis=0,skipna=True).idxmax(axis=1)
)

#WordCloud
cloud_cont=(posit_wrds.drop(['Sentiment'],axis=1).columns.values).tolist()
wordcloud = WordCloud(width = 1080, height = 720,
background_color ='white',
stopwords = stopwords,
min_font_size = 10).generate(' '.join([str(elem) for elem in cloud_cont]))
#comment_words -> string of words for the wordcloud separated by spaces
im = wordcloud.to_image()
im.save("wordcloud.png")
# plot the WordCloud image
plt.figure(figsize = (16, 9), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
