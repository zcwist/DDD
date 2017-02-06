from ConceptManager import ConceptManager as CM
import heapq
import numpy as np
import GensimEmbedding as emb
from NewPlot import Plot
from scipy.cluster.hierarchy import linkage, fcluster


class SWSC(object):
	"""docstring for SWSC(Similar word, similar concept)
	Basically, we calculate the similarity of concepts with the average of k most word pairs from the concept pair
	"""
	def __init__(self, cm, k=3):
		"""
		cm: ConceptMangager
		k: how many words pairs count
		"""
		super(SWSC, self).__init__()
		self.cm = cm
		self.k = k
		self.simi_mat = self.simi_mat()

	def simi_mat(self):
		"""similarity matrix, calculated by cosine distance, 1 for exactly the same"""
		cm_size = len(self.cm.conceptList)
		simi_mat = np.ndarray(shape=(cm_size,cm_size)) 

		for y,concept_y in enumerate(self.cm.conceptList):
			nvs_y = concept_y.noun_and_verb()['Noun'] + concept_y.noun_and_verb()['Verb']
			for x, concept_x in enumerate(self.cm.conceptList):
				nvs_x = concept_x.noun_and_verb()['Noun'] + concept_x.noun_and_verb()['Verb']
				simi_list = list()
				for nv_y in nvs_y:
					for nv_x in nvs_x:
						simi_list.append(emb.similarity(nv_x,nv_y))
				simi_mat[x][y] = np.mean(heapq.nlargest(self.k,simi_list))

		return simi_mat

	def linkage(self):
		
		Z = linkage(self.simi_mat, 'average', metric = 'cosine')
		return Z

	def hierachical_clustering(self, cluster_size = None):
		if cluster_size == None:
			cluster_size = len(self.cm.categoryList)
		labels = fcluster(self.linkage(), cluster_size, criterion='maxclust')
		return labels

	def dendrogram(self):
		
		Plot().dendrogram(self.simi_mat, self.cm.concept_name_list)

	def dendro_heat(self):
		Plot().heatmap(self.simi_mat, self.cm.concept_name_list)



if __name__ == '__main__':
	cm = CM(80,filename="dataset/ConceptTeam2.csv")
	# cm = CM(80)
	# print (SWSC(cm,3).simi_mat)
	# print (cm.concept_name_list)
	# SWSC(cm, 5).dendrogram()
	SWSC(cm,5).dendro_heat()





		
		