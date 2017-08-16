from cluster import Cluster
class HumanClaster(Cluster):
	"""docstring for HumanClaster"""
	def __init__(self, conceptManager):
		super(HumanClaster, self).__init__(conceptManager)
	
	def doCluster(self):
		pass

if __name__ == '__main__':
	HumanClaster()
		