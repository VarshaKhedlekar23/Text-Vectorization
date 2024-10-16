corpus = ['dog eats meat','man eats meat']  

  
vocab = {}
count = 0
for doc in corpus:
  for word in doc.split():
    if word not in vocab:
      count = count+1
      vocab[word] = count
print(vocab)



def one_hot(doc):
  one_hot = []
  for word in doc.split():
    temp = [0]*len(vocab)
    if word in vocab:
      temp[vocab[word]-1] = 1
    one_hot.append(temp) 
  return one_hot 


print(one_hot("dog eats meat"))  

print(one_hot("man eats meat"))
  
  
  

  
