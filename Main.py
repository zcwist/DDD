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
		concept_list = cm.concept_name_list

		concept_index = range(1,len (concept_list)+1)

		# exportHM(concept_index,cm.concept_category_list, concept_labels, concept_list, "Human_machine_details_Team"+str(number))
		exportHM2(cm.conceptList,concept_labels,"Human_manchine_details_des_Team"+str(number))

		plotBubble(h_labels,m_labels,concept_index,xticklabels=cm.categoryList,yticklabels=h_ticklabel,sort=True)
		plt.savefig("Graph/HMPlot/HMPlotTeam"+str(number))
		swsc.dendro_heat()
		plt.savefig("Graph/Dendrogram/DendrogramTeam"+str(number))
		plt.show()

	def exportOC(self, number):
		from swsc import SWSC
		from ConceptManager import OCConceptManager as CM
		# from PlotHM import plotHM
		from PlotBubble import plotBubble, plotBubbleOC
		from HMcsv import exportHM,exportHMOC

		filename = "dataset/ConceptTeamOC" + str(number) + ".csv"
		cm = CM(filename=filename)
		swsc = SWSC(cm)

		m_labels,concept_labels,h_ticklabel = swsc.label_clusters()
		m_labels = [x-1 for x in m_labels]
		# print (m_labels)
		h_labels = cm.category_index_list
		concept_list = cm.concept_name_list

		concept_index = range(1,len (concept_list)+1)

		# exportHM(concept_index,cm.concept_category_list, concept_labels, concept_list, "Human_machine_details_Team"+str(number))
		exportHMOC(cm.conceptList,concept_labels,"Human_manchine_details_des_Team"+str(number))

		plotBubbleOC(h_labels,m_labels,concept_index,xticklabels=cm.categoryList,yticklabels=h_ticklabel,sort=True)
		plt.savefig("Graph/HMPlot/HMPlotTeam"+str(number))
		
		# swsc.dendro_heat()
		# plt.savefig("Graph/Dendrogram/DendrogramTeam"+str(number))
		# plt.show()


	def simplified_plot(self,number,xFilteredStr=None, yFilteredStr=None,showLabel=True):
		from swsc import SWSC
		from ConceptManager import ConceptManager as CM
		# from PlotHM import plotHM
		from PlotBubble import plotBubble, plotSimpifiedBubble
		from HMcsv import exportHM,exportHM2

		filename = "dataset/ConceptTeam" + str(number) + ".csv"
		cm = CM(filename=filename)
		swsc = SWSC(cm)

		m_labels,concept_labels,h_ticklabel = swsc.label_clusters()
		m_labels = [x-1 for x in m_labels]

		h_labels = cm.category_index_list
		concept_list = cm.concept_name_list

		concept_index = range(1,len (concept_list)+1)

		# plotBubble(h_labels,m_labels,concept_index,xticklabels=cm.categoryList,yticklabels=h_ticklabel,sort=True)
		plotSimpifiedBubble(h_labels,m_labels,concept_index,xticklabels=cm.categoryList,yticklabels=h_ticklabel,sort=True,xFilteredStr=xFilteredStr,yFilteredStr=yFilteredStr,showLabel = showLabel,team_number=number)
		con = ""
		div = ""
		labeled = ""
		if xFilteredStr != None:
			div = "Div"
		if yFilteredStr != None:
			con = "Con"
		if showLabel:
			labeled = "Labeled"
		plt.savefig("Graph/HMPlot/SimplifiedHMPlotTeam"+con+div+labeled+str(number))
		# plt.show()

	def simplified_plot_oc(self,number,xFilteredStr=None, yFilteredStr=None, showLabel=True):
		from swsc import SWSC
		from ConceptManager import OCConceptManager as CM
		from PlotBubble import plotBubble, plotSimpifiedBubbleOC
		from HMcsv import exportHM,exportHM2

		filename = "dataset/ConceptTeamOC" + str(number) + ".csv"
		cm = CM(filename = filename)
		swsc = SWSC(cm)

		m_labels,concept_labels,h_ticklabel = swsc.label_clusters()
		m_labels = [x-1 for x in m_labels]

		h_labels = cm.category_index_list
		concept_list = cm.concept_name_list

		concept_index = range(1,len (concept_list)+1)

		#export HM file


		#HMPlot
		plotSimpifiedBubbleOC(h_labels,m_labels,concept_index,xticklabels=cm.categoryList,yticklabels=h_ticklabel,sort=True,xFilteredStr=xFilteredStr,yFilteredStr=yFilteredStr,showLabel = showLabel,team_number=number)
		

		con = ""
		div = ""
		labeled = ""
		if xFilteredStr != None:
			div = "Div"
		if yFilteredStr != None:
			con = "Con"
		if showLabel:
			labeled = "Labeled"
		plt.savefig("Graph/HMPlot/SimplifiedHMPlotTeam"+con+div+labeled+str(number))
		# plt.show()

	def export_me110(self, number):
		from swsc import SWSC
		from ConceptManager import ConceptManager as CM

		from HMcsv import exportHM2

		filename = "dataset/ME110/Team" + str(number) + ".csv"
		cm = CM(filename=filename)
		swsc = SWSC(cm)

		# m_labels,concept_labels,h_ticklabel = swsc.label_clusters()

		# exportHM2(cm.conceptList, concept_labels, "ME110_HM_Team" + str(number) )




if __name__ == '__main__':

	#ME110
	# SWSCTester().export_me110(2)






	#ME310

	SWSCTester().export(1)
	# SWSCTester().simplified_plot(1,yFilteredStr="car;people",showLabel=True)
	# SWSCTester().simplified_plot(7,xFilteredStr="Met-eared",showLabel=True)
	# SWSCTester().simplified_plot(6,xFilteredStr="Children Friendly Designs",showLabel=True)
	# SWSCTester().simplified_plot(4,yFilteredStr="app;challenge",showLabel=True)
	

	#Team1
	# SWSCTester().simplified_plot(1,yFilteredStr="car;people",showLabel=False)
	# SWSCTester().simplified_plot(1,xFilteredStr="SAFETY",showLabel=False)

	#Team2
	# SWSCTester().simplified_plot(2,yFilteredStr="vehicle",showLabel=False)
	# SWSCTester().simplified_plot(2,xFilteredStr="Controller",showLabel=False)

	#Team4
	# SWSCTester().simplified_plot(4,yFilteredStr="glasses;display",showLabel=False)
	# SWSCTester().simplified_plot(4,xFilteredStr="service",showLabel=False)

	#Team6
	# SWSCTester().simplified_plot(6,yFilteredStr="sculpture;tail",showLabel=False)
	# SWSCTester().simplified_plot(6,xFilteredStr="Modern Wind Farms",showLabel=False)

	#Team7
	# SWSCTester().simplified_plot(7,yFilteredStr="caps;water",showLabel=False)
	# SWSCTester().simplified_plot(7,xFilteredStr="Met-eared",showLabel=False)

	#Team8
	# SWSCTester().simplified_plot(8,yFilteredStr="arrays;patch",showLabel=False)
	# SWSCTester().simplified_plot(8,xFilteredStr="Improve the effectiveness of drug delivery",showLabel=False)

	#Team9
	# SWSCTester().simplified_plot(9,yFilteredStr="structure;structures",showLabel=False)
	# SWSCTester().simplified_plot(9,xFilteredStr="Tensportation",showLabel=False)

	#Team11 Balloon Modifications
	# SWSCTester().simplified_plot(11,yFilteredStr="vertebra;compression",showLabel=False)
	# SWSCTester().simplified_plot(11,xFilteredStr="Balloon Modifications",showLabel=False)

	#Team12
	# SWSCTester().exportOC(12) 
	# SWSCTester().simplified_plot_oc(12,yFilteredStr="legs;spine",showLabel=False)
	# SWSCTester().simplified_plot_oc(12,xFilteredStr="Communication",showLabel=False)

	#Team13
	# SWSCTester().exportOC(13)
	# SWSCTester().simplified_plot_oc(13,yFilteredStr="rubber;glass",showLabel=False)
	# SWSCTester().simplified_plot_oc(13,xFilteredStr="Scalable",showLabel=False)

	#Team14
	# SWSCTester().exportOC(14)
	# SWSCTester().simplified_plot_oc(14,yFilteredStr="voice;speaker",showLabel=False)
	# SWSCTester().simplified_plot_oc(14,xFilteredStr="Facial expressions",showLabel=False)



