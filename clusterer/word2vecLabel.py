from label import Label
import GensimEmbedding as emb
import operator
import math
class Word2VecLabel(Label):
	"""docstring for Word2VecLabel"""
	k_percent = 0.05
	def __init__(self, cluster):
		super(Word2VecLabel, self).__init__(cluster)
		self.label_cluster()

	def label_cluster(self):
		cluster_data = self.cluster.getCluster()
		for i in cluster_data:
			self.cluster.mapping[i] = self.label_a_cluster(cluster_data[i])

	def label_a_cluster(self, one_cluster):

		contri_word_freq = dict()

		def freq_counter(word):
			if word not in contri_word_freq.keys():
				contri_word_freq[word] = 0
			contri_word_freq[word] += 1

		for y, ci_y in enumerate(one_cluster):
			nvs_y = ci_y.noun_and_verb()['Noun']
			for x, ci_x in enumerate(one_cluster):
				if x <= y:
					continue
				nvs_x = ci_x.noun_and_verb()['Noun']
				simi_dic = dict()
				for nv_y in nvs_y:
					for nv_x in nvs_x:
						simi_dic[(nv_x, nv_y)] = emb.similarity(nv_x,nv_y)
				sorted_sim_dic = sorted(simi_dic.items(), key=operator.itemgetter(1), reverse=True)
				k = int(math.ceil(len(nvs_x)*len(nvs_y)*self.k_percent))
				for i in range(k):
					freq_counter(sorted_sim_dic[i][0][0])
					freq_counter(sorted_sim_dic[i][0][1])
		sorted_word_freq = sorted(contri_word_freq.items(), key=operator.itemgetter(1), reverse=True)

		top_n = 2
		label = ""
		for i in range(top_n):
			try:
				label = label + sorted_word_freq[i][0] + ";"
			except Exception as e:
				print ("not enough keywords")
				return one_cluster[0].conceptName()

		label = label[0:-1]
		return label

if __name__ == '__main__':
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
	from concept.ConceptManager import ConceptManager as CM
	from dataset.datautils import datapath
	from word2veccluster import Word2VecCluster
	cm = CM(filename=datapath("ME292",2))

	cluster = Word2VecCluster(cm)

	Word2VecLabel(cluster)
	print cluster.mapping





