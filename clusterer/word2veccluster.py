from cluster import Cluster
import GensimEmbedding as emb
import numpy as np
import heapq
from scipy.cluster.hierarchy import linkage, fcluster
import math
from word2vecLabel import Word2VecLabel


class Word2VecCluster(Cluster):
	"""docstring for Word2VecCluster
	Basically, we calculate the similarity of concepts with the average of 
	k most word pairs from the concept pair
	"""
	k_percent = 0.05
	def __init__(self, conceptManager, k = None):
		if k is None:
			k = len(conceptManager.categoryList)
		self.k = k
		super(Word2VecCluster, self).__init__(conceptManager)
		self.doCluster()
		Word2VecLabel(self)

	def doCluster(self):
		cm = self.conceptManager
		cluster_labels = self.hierachical_clustering()
		cluster_dict = dict()
		for i,label in enumerate(cluster_labels):
			if (label-1) not in cluster_dict.keys():
				cluster_dict[label-1] = list()
			cluster_dict[label-1].append(cm.conceptList[i])
		self.cluster = cluster_dict
	
	def simi_mat(self):
		"""similarity matrix, calculated by cosine distance, 1 for exactly the same"""
		cm = self.conceptManager
		cm_size = len(cm.conceptList)
		simi_mat = np.ndarray(shape=(cm_size,cm_size))

		for y, concept_y in enumerate(cm.conceptList):
			nvs_y = concept_y.noun_and_verb()['Noun'] + concept_y.noun_and_verb()['Verb']
			for x, concept_x in enumerate(cm.conceptList):
				if x == y:
					simi_mat[x][y] = 1
					continue
				if x > y:
					continue
				nvs_x = concept_x.noun_and_verb()['Noun'] + concept_x.noun_and_verb()['Verb']
				simi_list = list()
				for nv_y in nvs_y:
					for nv_x in nvs_x:
						simi_list.append(emb.similarity(nv_x,nv_y))
					k = int(math.ceil(len(nvs_y) * len(nvs_x) * self.k_percent))
					try:
						simi_mat[x][y] = np.mean(heapq.nlargest(k,simi_list))
						simi_mat[y][x] = simi_mat[x][y]
					except Exception as e:
						print(simi_list)
						raise e
		return simi_mat

	def linkage(self):
		Z = linkage(self.simi_mat(), 'complete', metric = 'cosine')
		return Z

	def hierachical_clustering(self):
		# print self.k
		return fcluster(self.linkage(), self.k, criterion="maxclust")

if __name__ == '__main__':
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
	from concept.ConceptManager import ConceptManager as CM
	from dataset.datautils import datapath
	cm = CM(filename=datapath("DesInv",1))

	print Word2VecCluster(cm).getCluster()
	print Word2VecCluster(cm).getMapping()
