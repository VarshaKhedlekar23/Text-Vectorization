doc1 = "dog bites meat"
doc2 = 'man eats meat'
doc3 = 'dog bites man'  


corpus = [doc1.split(),doc2.split(),doc3.split()]
my_overall_data  = corpus[0] + corpus[1] + corpus[2] 

print(f"My overall data: {my_overall_data}")

#implement Label Encoder
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder() 
integer_data = le.fit_transform(my_overall_data)
print(f"Integer Values are: {integer_data}")


from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
print(one_hot_encoder.fit_transform(corpus).toarray())


print(one_hot_encoder.transform(["dog eats meat".split()]).toarray())
