import csv
import string

class CSVFile(object):
	"""docstring for CSVFile"""
	def __init__(self, filename="dataset/ConceptTeam1.csv"):
		super(CSVFile, self).__init__()
		self.filename = filename

	def getContent(self):
		conceptlist = list()

		with open(self.filename,'rb') as csvfile:
			firstline = True
			reader = csv.reader(csvfile)
			for row in reader:
				if firstline:
					firstline = False
					continue
				conceptlist.append([row[1],row[2].translate(None,string.punctuation),row[3]])
		if False:
			from random import shuffle
			shuffle(conceptlist)


		return conceptlist

if __name__ == '__main__':
	# file = CSVFile("dataset/ConceptTeam1.csv")
	# file = CSVFile("dataset/AllConcepts.csv")
	file = CSVFile()
	print file.getContent()