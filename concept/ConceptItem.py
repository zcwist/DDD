from CSVFile import CSVFile
import nltk
nltk.download('averaged_perceptron_tagger')

import numpy as np
from random import shuffle

#Part-of-speech tagger
def tagPOS(sentence):
	text = nltk.word_tokenize(sentence)
	# print text
	print (nltk.pos_tag(text))

tagWeight = {'NN':2, 'NNP':2, 'VBG':1,'VB':1,'VBD':1}

class ConceptItem(object):
	"""docstring for ConceptItem"""
	def __init__(self, arg):
		super(ConceptItem, self).__init__()
		self.concept = arg[0].lower().replace('/',' ').replace('-',' ')
		self.description = arg[1].lower().replace('/',' ').replace('-',' ')
		self.category = arg[2]
		self.found = True
		self.lowemb = []

	def conceptName(self):
		return self.concept

	def getCategory(self):
		return self.category

	def tagBag(self, sentence):
		sentence = sentence.replace("/"," ")
		
		tagBag = tagWeight = {'NN':[], 'NNP':[],'NNS':[], 'VBG':[],'VB':[],'VBD':[]}
		text = nltk.word_tokenize(sentence)
		posList = nltk.pos_tag(text)
		# print (posList)
		for pos in posList:
			try:
				if pos[0] not in tagBag[pos[1]]:
					tagBag[pos[1]].append(pos[0])
			except Exception as e:
				pass
		return tagBag

	def noun_and_verb(self):
		sentence = self.concept + " " + self.description
		# print (sentence)
		tagBag = {'Noun':[],'Verb':[]}
		text = nltk.word_tokenize(sentence)
		posList = nltk.pos_tag(text)
		# print (posList)
		for pos in posList:
			gotit = False
			if "NN" in pos[1]: 
				tag = "Noun"
				gotit = True
			elif "VB" in pos[1]: 
				tag = "Verb"
				gotit = True

			if gotit and pos[0] not in tagBag[tag]:
				tagBag[tag].append(pos[0])
		return tagBag

	def NVConcept(self):
		nv = list()
		nvdict = self.noun_and_verb()
		for key in nvdict:
			for word in nvdict[key]:
				nv.append(word)
		# shuffle(nv)
		return nv	

	def conceptBag(self):
		return self.tagBag(self.concept + " " + self.description)

	def isFound(self):
		return self.found


	def fullConcept(self):
		return (self.concept.replace('-',' ') + " " +self.description.replace('-',' ')).split()

	def description_stat(self):
		desc_len = len(self.description.split())
		nvdict = self.noun_and_verb()
		return desc_len, len(nvdict["Noun"]),len(nvdict["Verb"])
		
	def getDescription(self):
		return self.description
if __name__ == '__main__':
	# file = CSVFile()
	file = CSVFile("../dataset/ConceptTeamOC12.csv")
	conceptlist = file.getContent()[0:1]
	for item in conceptlist:
		conceptItem = OCConceptItem(item)
		print (conceptItem.description)
		print (conceptItem.description_stat())

		