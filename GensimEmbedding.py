import gensim
import numpy as np


model = gensim.models.Word2Vec.load('GensimModel/model') #size 47133
vocab_size = len(model.wv.index2word)
embed_size = 200

embeddings = np.ndarray(shape=(vocab_size+1,200),dtype="f")

dictionary = dict()

for i,word in enumerate(model.wv.index2word):
	dictionary[word] = i #{word:index}
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
	cluster_size = 12
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

	# wordlist = [
	# 'steering',
	# 'phone',
	# 'voice',
	# 'gps',
	# 'charing',
	# 'interaction',
	# 'sleeping',
	# 'sensor',
	# 'navigation',
	# 'parking',
	# 'command',
	# 'keyboard',
	# 'wireless',
	# 'connectivity']

	

	embeddings = np.ndarray(shape=(len(wordlist),200))
	for i,word in enumerate(wordlist):
		embeddings[i] = wordVec(word)

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	low_dim_embs = tsne.fit_transform(embeddings)

	#plot
	plt.figure(figsize=(9,9))

	from sklearn.cluster import KMeans,SpectralClustering, AgglomerativeClustering
	# clustering = KMeans(n_clusters=cluster_size, random_state=0).fit(embeddings)
	# clustering = SpectralClustering(n_clusters=cluster_size, random_state=0).fit(embeddings)
	clustering = AgglomerativeClustering(n_clusters=cluster_size, affinity='cosine',linkage='complete').fit(embeddings)

	# Write to CSV
	def write2csv():
		import csv
		with open('output_csv/emb_tester.csv','wb') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for i in range(len(wordlist)):
				spamwriter.writerow([i,wordlist[i],clustering.labels_[i]])
		# print clustering.labels_

	def plot():
		import matplotlib.cm as cm
		colors = cm.rainbow(np.linspace(0,1,16))

		for i, word in enumerate(wordlist):
			x,y = low_dim_embs[i]
			plt.scatter(x,y,c=colors[clustering.labels_[i]],s=100,alpha=0.8,edgecolors='face')
			plt.annotate(word,
						xy=(x,y),
						xytext=(5,2),
						textcoords='offset points',
						ha='right',
						va='bottom')

		plt.show()

	write2csv()

def similarity_tester():
	# wordlist = [
	# 'steering','wheel','control',
	# 'self','driving',
	# 'voice',
	# 'charging',
	# 'car','interaction',
	# 'sleeping',
	# 'sensor',
	# 'parking',
	# 'hand','commands',
	# 'keyboard','mouse',
	# 'wireless',
	# 'autonomous','connectivity','display',
	# 'automatic','entry','exit',
	# 'luggage','collection']

	wordlist = [
	'steering',
	'phone',
	'voice',
	'gps',
	'charging',
	'interaction',
	'sleeping',
	'sensor',
	'navigation',
	'parking',
	'command',
	'keyboard',
	'wireless',
	'connectivity']

	# for i in range(len(wordlist)):
	# 	print (i,wordlist[i])
	import matplotlib.pyplot as plt

	dist_matrix = np.ndarray(shape=(len(wordlist),len(wordlist)))
	for i in range(len(wordlist)):
		for j in range(len(wordlist)):
			dist_matrix[i][j] = model.similarity(wordlist[i],wordlist[j])


	def dendrogram():
		from scipy.cluster.hierarchy import dendrogram, linkage
		Z = linkage(dist_matrix,'average',metric='cosine')

		plt.figure(figsize=(9,9))
		dendrogram(Z,labels=wordlist)
		plt.show()
	dendrogram()


	# from sklearn.cluster import AgglomerativeClustering
	# clustering = AgglomerativeClustering(n_clusters=5, affinity='cosine',linkage='complete').fit(embeddings)
	# print (clustering.children_)
	# print (clustering.labels_)

	# print embeddings
	# plt.matshow(embeddings,cmap=plt.cm.gray)
	# plt.show()





if __name__ == '__main__':
	# emb_tester()
	# print (model.most_similar(positive=['woman', 'king'], negative=['man']))
	# print (model.most_similar(positive=['steering'],negative=[]))

	# print (model.similarity('car','driver'))
	similarity_tester()

