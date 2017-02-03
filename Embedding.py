import pickle

with open('LearntModel/final_embeddings','rb') as f:
	embeddings = pickle.load(f)

with open('LearntModel/dictionary','rb') as f:
	dictionary = pickle.load(f)

with open('LearntModel/reverse_dictionary','rb') as f:
	reverse_dictionary = pickle.load(f)

def wordVec(word):
	try:
		return embeddings[dictionary[word]]
	except Exception as e:
		return embeddings[dictionary["UNK"]]

def wordIndex(word):
	try:
		return dictionary[word]
	except Exception as e:
		return dictionary["UNK"]

def emb_tester():
	from sklearn.manifold import TSNE
	import numpy as np
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

	embeddings = np.ndarray(shape=(len(wordlist),128))
	for i,word in enumerate(wordlist):
		embeddings[i] = wordVec(word)

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	low_dim_embs = tsne.fit_transform(embeddings)

	#plot
	plt.figure(figsize=(9,9))

	for i, word in enumerate(wordlist):
		x,y = low_dim_embs[i]
		plt.scatter(x,y,s=100,alpha=0.8,edgecolors='face')
		plt.annotate(word,
						xy=(x,y),
						xytext=(5,2),
						textcoords='offset points',
						ha='right',
						va='bottom')

	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=6, random_state=0).fit(embeddings)
	print (kmeans.labels_)


	plt.show()
		

if __name__ == '__main__':
	# print (wordVec('wordd'))
	# emb_tester()
	pass