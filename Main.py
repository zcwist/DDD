import tensorflow as tf
from para2vec import *
from ConceptManager import ConceptManager as CM

flags = tf.app.flags

flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer("para_embedding_size", 20, "The embedding dimension size of paragraph vector")
flags.DEFINE_integer("batch_size", 128,
					 "Number of training paragraph examples processed per step "
					 "(size of a minibatch).")
flags.DEFINE_integer("window_size", 2,
					 "Size of sampling window")
flags.DEFINE_integer("cluster_size", 13,
					 "Size of cluster for k means")
flags.DEFINE_integer("num_steps",10000, "The number of training times")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
					 "Negative samples per training example.")
flags.DEFINE_bool("random_order", False,"random order of data set")

FLAGS = flags.FLAGS

class MainTester(object):
	"""docstring for MainTester"""
	def __init__(self):
		super(MainTester, self).__init__()
		self.opts = Options(FLAGS)

	def testSimplifiedSet(self):
		""" We select 14 words to represent the first 14 concepts.
		When we use the word embeddings, the clusters looks good.
		So now, we want to test if para2vec works good enough 
		to predict one word A with the word A"""
		with tf.Graph().as_default(), tf.Session() as session:
			# model = Para2vec(CM(14,filename="simplified_data_set.csv"),self.opts,session)
			model = Para2VecConc(CM(14,filename="simplified_data_set.csv"),self.opts,session)
			model.train()
			# model.clustering()
			model.draw_dendrogram()		


class SWSCTester(object):
	"""docstring for SWSCTester"""
	def __init__(self):
		super(SWSCTester, self).__init__()

	def data_sample(self):
		from swsc import SWSC
		from ConceptManager import ConceptManager as CM
		from PlotHM import plotHM
		from HMcsv import exportHM
		cm = CM(80)
		swsc = SWSC(cm)

		m_labels = swsc.hierachical_clustering()
		h_labels = cm.category_index_list
		concept_list = cm.concept_name_list

		exportHM(cm.categoryList, m_labels, concept_list, self.__class__.__name__)



		plotHM(h_labels,m_labels,concept_list,xticklabels=cm.categoryList,sort=True)
		swsc.dendro_heat()

	def export(self, number):
		from swsc import SWSC
		from ConceptManager import ConceptManager as CM
		# from PlotHM import plotHM
		from PlotBubble import plotBubble
		from HMcsv import exportHM,exportHM2

		filename = "dataset/ConceptTeam" + str(number) + ".csv"
		cm = CM(filename=filename)
		swsc = SWSC(cm)

		m_labels,concept_labels,h_ticklabel = swsc.label_clusters()
		m_labels = [x-1 for x in m_labels]
		# print (m_labels)
		h_labels = cm.category_index_list
		# print (h_lab6ls)
		concept_list = cm.concept_name_list

		concept_index = range(1,len (concept_list)+1)

		exportHM(concept_index,cm.concept_category_list, concept_labels, concept_list, "Human_machine_details_Team"+str(number))
		exportHM2(cm.conceptList,concept_labels,"Human_manchine_details_des_Team"+str(number))

		plotBubble(h_labels,m_labels,concept_index,xticklabels=cm.categoryList,yticklabels=h_ticklabel,sort=True)
		plt.savefig("Graph/HMPlot/HMPlotTeam"+str(number))
		swsc.dendro_heat()
		plt.savefig("Graph/Dendrogram/DendrogramTeam"+str(number))
		plt.show()
		

if __name__ == '__main__':
	SWSCTester().export(11)