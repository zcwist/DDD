from cluster import Cluster
class HumanCluster(Cluster):
	"""docstring for HumanClaster"""
	def __init__(self, conceptManager):
		super(HumanCluster, self).__init__(conceptManager)
		self.doCluster()
	
	def doCluster(self):
		iter = 0
		conceptList = self.conceptManager.conceptL()
		reverseMapping = {}
		for i in conceptList:
			human_catergory = i.getCategory()
			if human_catergory in reverseMapping:
				curIndex = reverseMapping[human_catergory]
			else:
				reverseMapping[human_catergory] = iter
				curIndex = iter
				iter += 1

			self.mapping[curIndex] = human_catergory
			if curIndex in self.cluster:
				self.cluster[curIndex].append(i)
			else:
				self.cluster[curIndex] = [i]



if __name__ == '__main__':

	from concept.ConceptManager import ConceptManager as CM
	from dataset.datautils import datapath
	cm = CM(filename=datapath("DesInv",1))

	print HumanCluster(cm).getCluster()
	print HumanCluster(cm).getMapping()