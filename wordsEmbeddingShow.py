"""Show the 2-D embedding of every words with a circle for a certain concept"""

import numpy as np
import GensimEmbedding as ge
import matplotlib.pyplot as plt
from ConceptManager import ConceptManager as CM
conceptM = CM(14)

noun_list = list()
verb_list = list()

for concept in conceptM.conceptList:
	nv = concept.noun_and_verb()
	for noun in nv['Noun']:
		if noun not in noun_list:
			noun_list.append(noun)
	for verb in nv['Verb']:
		if verb not in verb_list:
			verb_list.append(verb)

noun_matrix = np.ndarray(shape=(len(noun_list),200))

for i,word in enumerate(noun_list):
	noun_matrix[i] = ge.wordVec(word)

from sklearn.manifold import TSNE
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0,1,14))
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(noun_matrix)

i = 0

for concept in conceptM.conceptList:
	nv = concept.noun_and_verb()
	for noun in nv['Noun']:
		# print noun_list.index(noun)
		x,y = low_dim_embs[noun_list.index(noun)]
		plt.scatter(x,y,c=colors[i],s=50)
		plt.annotate(noun,
				xy=(x,y),
						xytext=(5,2),
						textcoords='offset points',
						ha='right',
						va='bottom')
	i = i+1 




plt.show()