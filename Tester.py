import numpy as np
import GensimEmbedding as emb
from NewPlot import Plot 
class GensimEmbeddingTester(object):
	"""docstring for GensimEmbeddingTester"""
	
	def __init__(self):
		super(GensimEmbeddingTester, self).__init__()
	
	def words_from_5concepts_dendrogram(self):
		from ConceptManager import ConceptManager as CM

		words = list()

		cm_size = 5
		cm = CM(cm_size)
		for i in range(cm_size):
			for word in cm.conceptList[i].fullConcept():
				if word not in words:
					if word in emb.model.wv.index2word:
						words.append(word)

		dist_matrix = np.ndarray(shape=(len(words),len(words)))
		for i in range(len(words)):
			for j in range(len(words)):
				dist_matrix[i][j] = emb.model.similarity(words[i],words[j])

		Plot().dendrogram(dist_matrix,words)

		


def main():
	GensimEmbeddingTester().words_from_5concepts_dendrogram()

if __name__ == '__main__':
	main()
		