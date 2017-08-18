from ConceptItem import ConceptItem
from CSVFile import CSVFile
import numpy as np

class ConceptManager(object):
	"""docstring for ConceptManager"""

	def __init__(self, filename, size=None):
		super(ConceptManager, self).__init__()

		self.conceptList = list()
		self.notfoundList = list()
		self.categoryList = list() #concept categories in all concepts
		self.concept_category_list = list() # concept category name of a concept
		self.category_index_list = list() # concept category index of a concept
		self.concept_name_list= list()
		
		file = CSVFile(filename)
		if size==None:
			content = file.getContent()
		try:
			content = file.getContent()[0:size]
		except Exception as e:
			print ("We don't have so many concepts")
			content = file.getContent()

		self.concept_size = len(file.getContent())
		
		for item in content:
			newconcept = ConceptItem(item)
			self.conceptList.append(newconcept)
			self.concept_name_list.append(newconcept.conceptName())
			self.concept_category_list.append(newconcept.getCategory())
			if (newconcept.getCategory() not in self.categoryList):
				self.categoryList.append(newconcept.getCategory())
			self.category_index_list.append(self.categoryList.index(newconcept.getCategory()))
		self.size = len(self.conceptList)

	def foo(self):
		print (self.conceptList[0].conceptBag())

	def conceptL(self):
		return self.conceptList

	def getCategoryList(self):
		return self.categoryList

	def getCateIndex(self,category):
		return self.categoryList.index(category)

	def itemCoordMat(self):
		coordMat = np.zeros((len(self.conceptList),128))
		notfoundindex = list()
		for i,concept in enumerate(self.conceptList):
			coordMat[i] = concept.itemVector()
			if (not concept.isFound()):
				self.notfoundList.append(concept)
				print ("missing " + concept.conceptName())
				self.conceptList.remove(concept)
				notfoundindex.append(i)
		coordMat = np.delete(coordMat,notfoundindex,0)
		return coordMat


	def dimRed(self,algname='tsne'):
		coordMat = self.itemCoordMat()
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		mds = MDS(n_components=2)
		se = SpectralEmbedding()
		alg = {"tsne":tsne,"mds":mds,"se":se}
		low_dim_embs = alg[algname].fit_transform(coordMat)
		for i,emb in enumerate(low_dim_embs):
			# print emb
			self.conceptList[i].setLowEmb(emb)

	def description_statistics(self):
		desc_len_sum = 0.0
		noun_num_sum = 0.0
		verb_num_sum = 0.0
		for concept in self.conceptList:
			desc_len, noun_num, verb_num = concept.description_stat()
			desc_len_sum += desc_len
			noun_num_sum += noun_num
			verb_num_sum += verb_num
		concept_num = len(self.conceptList)
		return desc_len_sum/concept_num, noun_num_sum/concept_num, verb_num_sum/concept_num

	def printHumanCluster(self):
		return self.categoryList



class OCConceptManager(ConceptManager):
	"""Concept Manager for overlapping clustered concepts"""
	def __init__(self, size=None, filename="dataset/ConceptTeamOC12.csv"):
		# super(OCConceptManager, self).__init__()

		file = CSVFile(filename)
		if size==None:
			content = file.getContent()
		try:
			content = file.getContent()[0:size]
		except Exception as e:
			print ("We don't have so many concepts")
			content = file.getContent()

		self.concept_size = len(file.getContent())
		
		for item in content:
			newconcept = OCConceptItem(item)
			self.conceptList.append(newconcept)
			self.concept_name_list.append(newconcept.conceptName())
			self.concept_category_list.append(newconcept.getCategory())

			concept_category_index = list()

			for categoryName in newconcept.category:

				if categoryName not in self.categoryList:
					self.categoryList.append(categoryName)
				concept_category_index.append(self.categoryList.index(categoryName))

			self.category_index_list.append(concept_category_index)

		self.size = len(self.conceptList)
		

if __name__ == '__main__':
	# cm = ConceptManager(10,filename="dataset/AllConcepts.csv")
	# cm.dimRed()
	# print (cm.conceptList[1].fullConcept())

	# cm = ConceptManager(10,filename="simplified_data_set.csv")
	# cm = OCConceptManager()
	# print (cm.concept_category_list)
	# print (cm.description_statistics())

	# cm = OCConceptManager(filename="dataset/ConceptTeamOC14.csv")
	# print (cm.description_statistics())

	# print datapath("abc",1)
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
	from dataset.datautils import datapath

	className = "DesInv"
	for teamNo in range(15):
		try:
			cm = ConceptManager(filename=datapath(className,teamNo))
			print "%s;%d;%s" % (className,teamNo,cm.printHumanCluster())
		except Exception as e:
			pass
		
		
		# print "%s;%d;%s" % (className,teamNo,cm.printHumanCluster())
		# print datastr

		