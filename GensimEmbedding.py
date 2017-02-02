import gensim
import numpy as np

model = gensim.models.Word2Vec.load('GensimModel/model') #size 47133
vocab_size = len(model.wv.index2word)
embed_size = 200

embeddings = np.ndarray(shape=(vocab_size+1,200),dtype="f")

dictionary = dict()

for i,word in enumerate(model.wv.index2word):
	dictionary[word] = i
	embeddings[i] = model[word]

dictionary["UNK"] = vocab_size
embeddings[vocab_size] = embeddings[vocab_size-1] # the last word in the vocab is the rarest


def wordVec(word):
	try:
		return model[word]
	except Exception as e:
		return embeddings[vocab_size]
		
def wordIndex(word):
	try:
		return model.wv.index2word.index(word)
	except Exception as e:
		return vocab_size

if __name__ == '__main__':
	print (wordVec('wordd'))
