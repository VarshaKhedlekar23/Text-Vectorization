# Important - when a word is repeated most in a document, but not present in any other document  (sentence)
# Stopwords  - score will be less
# unique words repeated in a sentece - score will be high 
#tf idf is bag of words iwth removal of stopwords

import pandas as pd
import math
import sklearn

first_sent = "Data Science is an Amazing career in the current world"
second_sent = "Deep learning is a subset of machine learning"

first_sent = first_sent.split(" ")
second_sent = second_sent.split(' ')
vocab = set(first_sent).union(set(second_sent)) 
print(vocab)


wordDict1 = dict.fromkeys(vocab,0)
wordDict2 = dict.fromkeys(vocab,0) 
print(wordDict1)
print(wordDict2)


for word in first_sent:
  wordDict1[word] += 1

for word in second_sent:
  wordDict2[word] += 1

print(wordDict1)
print(first_sent)

print(wordDict2)
print(second_sent)


df = pd.DataFrame([wordDict1,wordDict2])
print(df)



#tf = freq of a word in a document/total number of words in a document
def calculateTF(wordDict, doc):
  tfDict = {}
  len_doc = len(doc)  

  for word,count in wordDict.items(): 
    tfDict[word] = count/len_doc
  
  return tfDict


tf1 = calculateTF(wordDict1,first_sent) 
tf2 = calculateTF(wordDict2,second_sent) 

tf = pd.DataFrame([tf1,tf2])
print(tf)







import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

f1 = [word for word in wordDict1 if word.lower() not in stopwords.words('english') ]   #Lambda func 
f2 = [word for word in wordDict2 if word.lower() not in stopwords.words('english') ] 

print(f1)
print(f2)


def calculateIDF(doc):
  idfDict = {}
  len_doc = len(doc) 

  idfDict = dict.fromkeys(doc[0].keys(), 0) 
  for word, val in idfDict.items():
      idfDict[word] = math.log10(len_doc / (float(val) + 1))  
        
  return(idfDict)

idfs = calculateIDF([wordDict1, wordDict2])
print(idfs)

print(tf1)

def computeTFIDF(tfBow, idfs): 
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word] 
    return(tfidf)
idf1 = computeTFIDF(tf1, idfs)
idf2 = computeTFIDF(tf2, idfs) 

#putting it in a dataframe 
idf= pd.DataFrame([idf1, idf2])
print(idf)


TF_IDF=computeTFIDF(tf,idf)
print(TF_IDF)





################### Using Scikit Learn #############

from sklearn.feature_extraction.text import TfidfVectorizer
first_sent = "Data Science is an Amazing career in the current world"
second_sent = "Deep learning is a subset of machine learning"

vec = TfidfVectorizer()
result = vec.fit_transform([first_sent,second_sent])
result.toarray()


TF_IDF_sci=pd.DataFrame(result.toarray(),columns= vec.get_feature_names_out())

print(TF_IDF_sci)
