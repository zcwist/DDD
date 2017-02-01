import matplotlib.pyplot as plt
from ConceptManager import ConceptManager as CM
import matplotlib.cm as cm
import numpy as np

class Plot(object):
	"""docstring for Plot"""
	colors = cm.rainbow(np.linspace(0,1,14))
	def __init__(self, conceptManager):
		super(Plot, self).__init__()
		self.conceptManager = conceptManager

	def draw(self,save=False,filename="plot.png"):
		plt.figure(figsize=(9,9))
		for i, concept in enumerate(self.conceptManager.conceptL()):
			# print concept.lowEmb()
			x, y = concept.lowEmb()
			plt.scatter(x,y,c=self.colors[self.conceptManager.getCateIndex(concept.getCategory())],s=100,linewidths=0,alpha=0.8,edgecolors='face')
			# plt.annotate(concept.conceptName(),
			# 			xy=(x,y),
			# 			xytext=(5,2),
			# 			textcoords='offset points',
			# 			ha='right',
			# 			va='bottom')
		if save:
			plt.savefig(filename)
		plt.show()

	def drawWithTag(self,save=False,filename="plot.png"):
		plt.figure(figsize=(9,9))
		for i, concept in enumerate(self.conceptManager.conceptL()):
			x, y = concept.lowEmb()
			plt.scatter(x,y,c=self.colors[self.conceptManager.getCateIndex(concept.getCategory())],s=100,linewidths=0,alpha=0.8,edgecolors='face')
			plt.annotate(concept.conceptName(),
						xy=(x,y),
						xytext=(5,2),
						textcoords='offset points',
						ha='right',
						va='bottom')
		if save:
			plt.savefig(filename)
		plt.show()


if __name__ == '__main__':
	cm = CM(20)
	cm.dimRed('tsne')
	Plot(cm).drawWithTag()