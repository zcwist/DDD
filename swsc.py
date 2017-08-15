from ConceptManager import ConceptManager as CM
import heapq
import numpy as np
import GensimEmbedding as emb
from NewPlot import Plot
from scipy.cluster.hierarchy import linkage, fcluster
import math
import operator


class SWSC(object):
	"""docstring for SWSC(Similar word, similar concept)
	Basically, we calculate the similarity of concepts with the average of k most word pairs from the concept pair
	"""
	k_percent = 0.05
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
				k = int(math.ceil(len(nvs_y) * len(nvs_x) * self.k_percent))
				try:
					simi_mat[x][y] = np.mean(heapq.nlargest(k,simi_list))
				except Exception as e:
					print(simi_list)
					raise e
				

		return simi_mat

	def linkage(self):
		
		Z = linkage(self.simi_mat, 'complete', metric = 'cosine')
		return Z

	def hierachical_clustering(self, cluster_size = None):
		if cluster_size == None:
			cluster_size = len(self.cm.categoryList)
		labels = fcluster(self.linkage(), cluster_size, criterion='maxclust')
		return labels

	def spectral_clustering(self, cluster_size = None):
		from sklearn.cluster import SpectralClustering as sc
		if cluster_size == None:
			cluster_size = len(self.cm.categoryList)
		labels = sc.fit_predict(np.asmatrix(self.simi_mat))
		return labels

	def dendrogram(self):
		Plot().dendrogram(self.simi_mat, self.cm.concept_name_list)

	def dendro_heat(self):
		Plot().heatmap(self.simi_mat, self.cm.concept_name_list)

	def label_clusters(self, cluster_size = None):
		if cluster_size == None:
			cluster_size = len(self.cm.categoryList)
		concept_labels_index = self.hierachical_clustering(cluster_size)

		cluster_concept = dict() #{cluster_index:[concept_index]}
		cluster_label = dict() #{cluster_index:label}
		for i in range(1,cluster_size+1):
			cluster_concept[i] = list()
		for i,cluster_index in enumerate(concept_labels_index):
			cluster_concept[cluster_index].append(i)
		
		for cluster_index in cluster_concept:
			cluster_label[cluster_index] = self.label_a_cluster(cluster_concept[cluster_index])

		concept_labels = list()
		for i in concept_labels_index:
			concept_labels.append(cluster_label[i])

		return concept_labels_index,concept_labels,cluster_label.values()

	def label_a_cluster(self,concept_index_list,top_n = None):

		cl = self.cm.conceptList
		cil = concept_index_list
		if len(cil) == 1 :
			return cl[cil[0]].conceptName()

		contri_word_freq = dict() #{word:freq}

		def freq_counter(word):
			if word not in contri_word_freq.keys():
				contri_word_freq[word] = 0
			contri_word_freq[word] = contri_word_freq[word] + 1


		for y,ci_y in enumerate(cil):
			nvs_y = cl[ci_y].noun_and_verb()['Noun'] #+ cl[ci_y].noun_and_verb()['Verb']
			for x,ci_x in enumerate(cil):
				if x <= y:
					continue
				nvs_x = cl[ci_x].noun_and_verb()['Noun'] #+ cl[ci_x].noun_and_verb()['Verb']
				simi_dic = dict() #{(nv_x,nv_y):similarity}
				for nv_y in nvs_y:
					for nv_x in nvs_x:
						simi_dic[(nv_x, nv_y)] = emb.similarity(nv_x,nv_y)
				sorted_sim_dic = sorted(simi_dic.items(), key=operator.itemgetter(1),reverse=True)
				k = int(math.ceil(len(nvs_y) * len(nvs_x) * self.k_percent))
				for i in range(k):
					freq_counter(sorted_sim_dic[i][0][0])
					freq_counter(sorted_sim_dic[i][0][1])

		sorted_word_freq = sorted(contri_word_freq.items(), key=operator.itemgetter(1), reverse=True)

		if top_n == None:
			top_n = 2
		label = ""
		for i in range(top_n):
			try:
				label = label + sorted_word_freq[i][0] + ";"
			except Exception as e:
				print ("not enough keywords")
			
		label = label[0:-1]
		return label


if __name__ == '__main__':
	cm = CM(filename="dataset/InFEWs.csv")
	# cm = CM(80)
	# print (SWSC(cm,3).simi_mat)
	# print (cm.concept_name_list)
	# SWSC(cm, 5).dendrogram()
	SWSC(cm).dendro_heat()
	# print (cm.categoryList)
	# print (cm.concept_category_list)
	

	# print (SWSC(cm).label_clusters())





		
		