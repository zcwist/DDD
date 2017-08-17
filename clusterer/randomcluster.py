from cluster import Cluster
from random import randint
class RandomCluster(Cluster):

    def __init__(self, conceptManager, k):
        super(RandomCluster, self).__init__(conceptManager)
        if k is None:
            self.k = len(conceptManager.concept_category_list)
        else:
            self.k = k
        self.doCluster()

    def doCluster(self):
        list = self.conceptManager.conceptList
        for i in list:
            randIndex = randint(0, self.k - 1)
            if randIndex in self.cluster:
                self.cluster[randIndex].append(i)
            else:
                self.cluster[randIndex] = [i]



if __name__ == '__main__':

	from concept.ConceptManager import ConceptManager as CM
	from dataset.datautils import datapath
	cm = CM(filename=datapath("DesInv",1))

	print RandomCluster(cm, 7).getCluster()
	# print RandomCluster(cm).getMapping()