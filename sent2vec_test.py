from sent2vec.vectorizer import Vectorizer
from scipy import spatial
from array import *
import numpy

sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]
#encoding sentences using BERT language model 
vectorizer = Vectorizer()
vectorizer.bert(sentences)
vectors = vectorizer.vectors
newv=[]
for i in vectors:
	newv.append(i.tolist())

print(newv[1])

#computing cosine distance vectors. Smaller distance -> greater similarity
dist_1 = spatial.distance.cosine(numpy.array(newv[0]), numpy.array(newv[1]))
dist_2 = spatial.distance.cosine(numpy.array(newv[0]), numpy.array(newv[2]))
print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
assert dist_1 < dist_2
# dist_1: 0.043, dist_2: 0.192