import pandas as pd
import numpy as np
import collections 


doc1 = "Harry Potter is an amazing movie!!"
doc2 = "Harry Potter is the best movie!"
doc3 = "Harry potter is so great"


import re
doc1 = re.sub(r"[^a-zA-Z0-9]"," ",doc1.lower()).split()
doc2 = re.sub(r"[^a-zA-Z0-9]"," ",doc2.lower()).split()
doc3 = re.sub(r"[^a-zA-Z0-9]"," ",doc3.lower()).split()


all_words = set(doc1+doc2+doc3)   #set datatype-doesnt allow duplicate data #creating corpus
print(all_words)



def BOWrepresentation(all_words, doc):
  bow = dict.fromkeys(all_words,0) 
  for word in doc:
    bow[word] = doc.count(word)
  return bow

bow1 = BOWrepresentation(all_words,doc1)
print(bow1)

bow2 = BOWrepresentation(all_words,doc2)
bow3 = BOWrepresentation(all_words,doc3)

df_bow = pd.DataFrame([bow1,bow2,bow3])
print(df_bow)




##### Using Sci-kit learn ########

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True) # Sentiment classification 
doc1 = "Harry Potter is an amazing movie!!"  
doc2 = "Harry Potter is the best movie!"
doc3 = "Harry potter is so great" 
cv_out = cv.fit_transform([doc1,doc2,doc3]) #if bag of words is used in binary form its one hot encoding ...(binary=True)

print(cv_out.toarray())

print(pd.DataFrame(cv_out.toarray(), columns= cv.get_feature_names_out()))



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() # Counting of how many times a word is being used
doc1 = "Harry Potter is an amazing movie , harry!!"
doc2 = "Harry Potter is the best movie!"
doc3 = "Harry potter is so great"
cv_out = cv.fit_transform([doc1,doc2,doc3])
print(pd.DataFrame(cv_out.toarray(), columns= cv.get_feature_names_out()))



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,3)) # unigram, bigram, trigram
doc1 = "Harry Potter is an amazing movie , harry!!"
doc2 = "Harry Potter is the best movie!"   
doc3 = "Harry potter is so great" 
cv_out = cv.fit_transform([doc1,doc2,doc3])
print(pd.DataFrame(cv_out.toarray(), columns= cv.get_feature_names_out()))


