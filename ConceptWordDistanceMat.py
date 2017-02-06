import numpy as np
import GensimEmbedding as ge
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ConceptManager import ConceptManager as CM
conceptM = CM(80)

def averageof3lowest():
	import heapq
	cm_size = len(conceptM.conceptList)
	dist_mat = np.ndarray(shape=(cm_size,cm_size))
	x=0
	y=0
	conceptName = list()
	for concepty in conceptM.conceptList:
		conceptName.append(concepty.conceptName())
		x = 0
		nouns_y = concepty.noun_and_verb()['Noun'] + concepty.noun_and_verb()['Verb']
		for conceptx in conceptM.conceptList:
			nouns_x = conceptx.noun_and_verb()['Noun'] + conceptx.noun_and_verb()['Verb']
			dist_list = list()
			
			for noun_y in nouns_y:
				for noun_x in nouns_x:
					dist_list.append(ge.similarity(noun_x,noun_y))
			dist_mat[x][y] = np.mean(heapq.nlargest(3,dist_list))
			x = x + 1
		y = y + 1

	from scipy.cluster.hierarchy import linkage
	from scipy.cluster.hierarchy import fcluster
	Z = linkage(dist_mat,'average',metric='cosine')
	m_labels = fcluster(Z, len(conceptM.categoryList), criterion='maxclust')
	h_labels = list()
	concept_list = list()

	# for i,concept in enumerate(conceptM.conceptList):
	# 	concept_list.append(concept.conceptName())
	# 	h_labels.append(conceptM.getCateIndex(concept.getCategory()))
	# 	# print (concept.conceptName(),concept.getCategory(),m_labels[i])

	# from PlotHM import plotHM
	# plotHM(h_labels,m_labels,concept_list,conceptM.categoryList,sort=True)

	def write2csv():
		import csv
		with open('output_csv/ConceptWordDistanceMat.csv','wb') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for i,concept in enumerate(conceptM.conceptList):
				spamwriter.writerow([concept.conceptName(),concept.getCategory(),m_labels[i]])

	# write2csv()



	

	# print dist_mat

	# import networkx as nx
	# G=nx.from_numpy_matrix(dist_mat)
	# nx.draw_circular(G)
	# plt.show()

	from NewPlot import Plot
	Plot().dendrogram(dist_mat,conceptName,show=True)


	# plt.imshow(dist_mat)
	# plt.colorbar()
	# plt.show()
			
def allwords():

	noun_list = list()
	i = 0
	for concept in conceptM.conceptList:
		print (i,concept.conceptName())
		i = i+1
		nv = concept.noun_and_verb()
		for noun in nv['Noun']:
			noun_list.append(noun)

	dist_mat = np.ndarray(shape=(len(noun_list),len(noun_list)))

	for i in range(len(noun_list)):
		for j in range(len(noun_list)):
			dist_mat[i][j] = ge.similarity(noun_list[i],noun_list[j])

	fig = plt.figure()

	plt.imshow(dist_mat)
	ax = fig.add_subplot(1, 1, 1)

	rect_x=0
	rect_y=0

	for concepty in conceptM.conceptList:
		rect_x=0
		for conceptx in conceptM.conceptList:
			rect = plt.Rectangle((rect_x,rect_y),len(conceptx.noun_and_verb()['Noun']),len(concepty.noun_and_verb()['Noun']),color='k',alpha=0.1)
			ax.add_patch(rect)
			# print(rect_x,rect_y,len(conceptx.noun_and_verb()['Noun']),len(concepty.noun_and_verb()['Noun']))
			rect_x = rect_x + len(conceptx.noun_and_verb()['Noun'])
		rect_y = rect_y + len(concepty.noun_and_verb()['Noun'])


	# rect = plt.Rectangle((0, 0), 10, 10, color='k', alpha=0.3)
	# ax.add_patch(rect)
	# rect = plt.Rectangle((10, 0), 20, 10, color='k', alpha=0.3)
	# ax.add_patch(rect)



	plt.show()

if __name__ == '__main__':
	averageof3lowest()