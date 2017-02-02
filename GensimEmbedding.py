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

def emb_tester():
	from sklearn.manifold import TSNE
	import matplotlib.pyplot as plt
	wordlist = [
	'steering','wheel','control',
	'self','driving',
	'voice',
	'charging',
	'car','interaction',
	'sleeping',
	'sensor',
	'parking',
	'hand','commands',
	'keyboard','mouse',
	'wireless',
	'autonomous','connectivity','display',
	'automatic','entry','exit',
	'luggage','collection']

	embeddings = np.ndarray(shape=(len(wordlist),200))
	for i,word in enumerate(wordlist):
		embeddings[i] = wordVec(word)

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	low_dim_embs = tsne.fit_transform(embeddings)

	#plot
	plt.figure(figsize=(9,9))

	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=6, random_state=0).fit(embeddings)
	print kmeans.labels_

	for i, word in enumerate(wordlist):
		x,y = low_dim_embs[i]
		plt.scatter(x,y,s=100,alpha=0.8,edgecolors='face')
		plt.annotate(word+str(kmeans.labels_[i]),
						xy=(x,y),
						xytext=(5,2),
						textcoords='offset points',
						ha='right',
						va='bottom')

	plt.show()


if __name__ == '__main__':
	# emb_tester()
	print (model.most_similar(positive=['woman', 'king'], negative=['man']))

