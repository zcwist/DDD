from Plot import Plot
from ConceptManager import ConceptManager as CM

if __name__ == '__main__':
	cm = CM(40)
	cm.dimRed('tsne')
	Plot(cm).drawWithTag()
	#let's do something different