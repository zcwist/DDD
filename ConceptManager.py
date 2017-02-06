from ConceptItem import ConceptItem
from CSVFile import CSVFile
from sklearn.manifold import TSNE,MDS,SpectralEmbedding,LocallyLinearEmbedding
import numpy as np

class ConceptManager(object):
	"""docstring for ConceptManager"""

	conceptList = list()
	notfoundList = list()
	categoryList = list()
	category_index_list = list()
	concept_name_list= list()
	# vecList = 
	def __init__(self, size, filename="dataset/ConceptTeam1.csv"):
		super(ConceptManager, self).__init__()

		self.concept_size = size
		file = CSVFile(filename)
		try:
			content = file.getContent()[0:size]
		except Exception as e:
			print ("We don't have so many concepts")
			content = file.getContent()
		
		for item in content:
			newconcept = ConceptItem(item)
			self.conceptList.append(newconcept)
			self.concept_name_list.append(newconcept.conceptName())
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

if __name__ == '__main__':
	# cm = ConceptManager(10,filename="dataset/AllConcepts.csv")
	# cm.dimRed()
	# print (cm.conceptList[1].fullConcept())

	cm = ConceptManager(10,filename="simplified_data_set.csv")
	print (cm.conceptList[1].fullConcept())


