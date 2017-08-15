"""Generate data, store data, visualize data, in this version"""

from ConceptManager import ConceptManager as CM
from swsc import SWSC
from PlotBubble import plotBubble, plotSimpifiedBubble, plotSimpifiedBubbleOC,plotBubbleOC
from HMcsv import exportHM,exportHM2
import os.path
import json
import matplotlib.pyplot as plt

class NewSWSCTester(object):
	"""docstring for NewSWSCTester"""
	def __init__(self,teamNumber,OC=False):
		super(NewSWSCTester, self).__init__()
		self.teamNumber = teamNumber 
		
		self.OC = OC
		self.data4HM = self.getDataJson()

	def getDataJson(self):
		number = self.teamNumber
		dataFileName = "HMData/HM"+str(number)+".dat"

		if self.OC:
			from ConceptManager import OCConceptManager as CM
		else:
			from ConceptManager import ConceptManager as CM

		data4HM = {}
		if not (os.path.isfile(dataFileName)):
			filename = "dataset/ConceptTeam" + str(number) + ".csv"
			if self.OC:
				filename = "dataset/ConceptTeamOC" + str(number) + ".csv"
			cm = CM(filename=filename)
			swsc = SWSC(cm)

			m_labels,concept_labels,h_ticklabel = swsc.label_clusters()
			m_labels = [x-1 for x in m_labels]
			# print (m_labels)
			h_labels = cm.category_index_list
			concept_list = cm.concept_name_list

			concept_index = range(1,len (concept_list)+1)

			categoryList = cm.categoryList

			data4HM["m_labels"] = m_labels
			data4HM["h_labels"] = h_labels
			data4HM["concept_list"] = concept_list
			data4HM["concept_index"] = concept_index
			data4HM["categoryList"] = categoryList
			data4HM["concept_labels"] = concept_labels
			data4HM["h_ticklabel"] = h_ticklabel


			with open(dataFileName, "w") as outfile:
				json.dump(data4HM, outfile)
		
		else:
			with open(dataFileName) as data_file:
				data4HM = json.load(data_file)

		return data4HM

	def parseJson(self):
		data4HM = self.data4HM
		return data4HM["m_labels"],data4HM["h_labels"],data4HM["concept_list"],data4HM["concept_index"],data4HM["categoryList"],data4HM["concept_labels"],data4HM["h_ticklabel"]
	def exportFull(self):
		m_labels, h_labels, concept_list, concept_index, categoryList,concept_labels,h_ticklabel = self.parseJson()
		plotBubble(h_labels,m_labels,concept_index,xticklabels=categoryList,yticklabels=h_ticklabel,sort=True)
		plt.savefig("Graph/HMPlot/HMPlotTeam"+str(self.teamNumber))
		# plt.show()

	def exportThumbnail(self):
		m_labels, h_labels, concept_list, concept_index, categoryList,concept_labels,h_ticklabel = self.parseJson()
		plotBubble(h_labels,m_labels,concept_index,xticklabels=None,yticklabels=None,sort=True)
		plt.savefig("Graph/Thumbnail/HMPlotTeam"+str(self.teamNumber))
		# plt.show()
		plt.close()

	def exportOCFull(self):
		number = self.teamNumber
		m_labels, h_labels, concept_list, concept_index, categoryList,concept_labels,h_ticklabel = self.parseJson()
		plotBubbleOC(h_labels,m_labels,concept_index,xticklabels=categoryList,yticklabels=h_ticklabel,sort=True)
		plt.savefig("Graph/HMPlot/HMPlotTeam"+str(number))
		# plt.show()

	def exportOCFullThumbnail(self):
		number = self.teamNumber
		m_labels, h_labels, concept_list, concept_index, categoryList,concept_labels,h_ticklabel = self.parseJson()
		plotBubbleOC(h_labels,m_labels,concept_index,xticklabels=None,yticklabels=None,sort=True)
		plt.savefig("Graph/Thumbnail/HMPlotTeam"+str(number))
		# plt.show()

	def simplified_plot(self,xFilteredStr=None, yFilteredStr=None,showLabel=True):
		number = self.teamNumber
		m_labels, h_labels, concept_list, concept_index, categoryList,concept_labels,h_ticklabel = self.parseJson()
		plotSimpifiedBubble(h_labels,m_labels,concept_index,xticklabels=categoryList,yticklabels=h_ticklabel,sort=True,xFilteredStr=xFilteredStr,yFilteredStr=yFilteredStr,showLabel = showLabel,team_number=number)
		con = ""
		div = ""
		labeled = ""
		if xFilteredStr != None:
			div = "Div"
		if yFilteredStr != None:
			con = "Con"
		if showLabel:
			labeled = "Labeled"
		plt.savefig("Graph/Thumbnail/SimplifiedHMPlotTeam"+con+div+labeled+str(number))
		# plt.show()
		plt.close()

	def simplified_plot_oc(self,xFilteredStr=None, yFilteredStr=None, showLabel=True):
		number = self.teamNumber
		m_labels, h_labels, concept_list, concept_index, categoryList,concept_labels,h_ticklabel = self.parseJson()
		#HMPlot
		plotSimpifiedBubbleOC(h_labels,m_labels,concept_index,xticklabels=categoryList,yticklabels=h_ticklabel,sort=True,xFilteredStr=xFilteredStr,yFilteredStr=yFilteredStr,showLabel = showLabel,team_number=number)
		

		con = ""
		div = ""
		labeled = ""
		if xFilteredStr != None:
			div = "Div"
		if yFilteredStr != None:
			con = "Con"
		if showLabel:
			labeled = "Labeled"
		plt.savefig("Graph/Thumbnail/SimplifiedHMPlotTeam"+con+div+labeled+str(number))

class SWSCTesterVariableK(NewSWSCTester):
	"""docstring for SWSCTesterVariableK"""
	def __init__(self,teamNumber,OC=False,k=None):
		super(SWSCTesterVariableK, self).__init__(teamNumber, OC)
		self.k = k
		self.data4HM = self.getDataJson(k)

	def getDataJson(self, k):
		number = self.teamNumber
		dataFileName = "HMData/HM"+str(number)+"k_" + k +".dat"

		if self.OC:
			from ConceptManager import OCConceptManager as CM
		else:
			from ConceptManager import ConceptManager as CM

		data4HM = {}
		if not (os.path.isfile(dataFileName)):
			filename = "dataset/ConceptTeam" + str(number) + ".csv"
			if self.OC:
				filename = "dataset/ConceptTeamOC" + str(number) + ".csv"
			cm = CM(filename=filename)
			swsc = SWSC(cm)

			m_labels,concept_labels,h_ticklabel = swsc.label_clusters(cluster_size=k)
			m_labels = [x-1 for x in m_labels]
			# print (m_labels)
			h_labels = cm.category_index_list
			concept_list = cm.concept_name_list

			concept_index = range(1,len (concept_list)+1)

			categoryList = cm.categoryList

			data4HM["m_labels"] = m_labels
			data4HM["h_labels"] = h_labels
			data4HM["concept_list"] = concept_list
			data4HM["concept_index"] = concept_index
			data4HM["categoryList"] = categoryList
			data4HM["concept_labels"] = concept_labels
			data4HM["h_ticklabel"] = h_ticklabel


			with open(dataFileName, "w") as outfile:
				json.dump(data4HM, outfile)
		
		else:
			with open(dataFileName) as data_file:
				data4HM = json.load(data_file)

		return data4HM

		


if __name__ == '__main__':
	# Team1
	# team1 = NewSWSCTester(1)
	# team1.exportFull()
	# team1.exportThumbnail()
	# team1.simplified_plot(yFilteredStr="car;people",showLabel=False)
	# team1.simplified_plot(xFilteredStr="SAFETY",showLabel=False)

	# # Team2
	# team2 = NewSWSCTester(2)
	# team2.exportThumbnail()
	# team2.simplified_plot(yFilteredStr="vehicle",showLabel=False)
	# team2.simplified_plot(xFilteredStr="Controller",showLabel=False)

	# #Team4
	# team4 = NewSWSCTester(4)
	# team4.exportThumbnail()
	# team4.simplified_plot(yFilteredStr="app;challenge",showLabel=False)
	# team4.simplified_plot(yFilteredStr="app;challenge",showLabel=True)
	# team4.simplified_plot(xFilteredStr="service",showLabel=False)


	# #Team6
	# team6 = NewSWSCTester(6)
	# team6.exportThumbnail()
	# team6.simplified_plot(yFilteredStr="sculpture;tail",showLabel=False)
	# team6.simplified_plot(xFilteredStr="Children Friendly Designs",showLabel=False)
	# team6.simplified_plot(xFilteredStr="Children Friendly Designs",showLabel=True)

	# #Team7
	# team7 = NewSWSCTester(7)
	# team7.exportThumbnail()
	# team7.simplified_plot(yFilteredStr="caps;water",showLabel=False)
	# team7.simplified_plot(xFilteredStr="Met-eared",showLabel=False)

	# # Team8
	# team8 = NewSWSCTester(8)
	# team8.exportThumbnail()
	# team8.simplified_plot(yFilteredStr="arrays;patch",showLabel=False)
	# team8.simplified_plot(xFilteredStr="Improve the effectiveness of drug delivery",showLabel=False)


	# #Team9
	# team9 = NewSWSCTester(9)
	# team9.exportThumbnail()
	# team9.simplified_plot(yFilteredStr="structure;structures",showLabel=False)
	# team9.simplified_plot(xFilteredStr="Tensportation",showLabel=False)

	# # Team11 Balloon Modifications
	# team11 = NewSWSCTester(11)
	# team11.exportThumbnail()
	# team11.simplified_plot(yFilteredStr="vertebra;compression",showLabel=False)
	# team11.simplified_plot(xFilteredStr="Balloon Modifications",showLabel=False)

	# #Team12
	# team12 = NewSWSCTester(12, OC=True)
	# team12.exportOCFullThumbnail()
	# team12.simplified_plot_oc(yFilteredStr="legs;spine",showLabel=False)
	# team12.simplified_plot_oc(xFilteredStr="Communication",showLabel=False)

	# #Team13
	# team13 = NewSWSCTester(13, OC=True)
	# team13.exportOCFullThumbnail()
	# team13.simplified_plot_oc(yFilteredStr="rubber;glass",showLabel=False)
	# team13.simplified_plot_oc(xFilteredStr="Scalable",showLabel=False)

	# #Team14
	# team14 = NewSWSCTester(14, OC=True)
	# team14.exportOCFullThumbnail()
	# team14.simplified_plot_oc(yFilteredStr="voice;speaker",showLabel=False)
	# team14.simplified_plot_oc(xFilteredStr="Facial expressions",showLabel=False)