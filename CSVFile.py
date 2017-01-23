import csv
import string

class CSVFile(object):
	"""docstring for CSVFile"""
	def __init__(self, fileanme="dataset/ConceptTeam1.csv"):
		super(CSVFile, self).__init__()
		self.fileanme = fileanme

	def getContent(self):
		conceptlist = list()

		with open(self.fileanme,'rb') as csvfile:
			firstline = True
			reader = csv.reader(csvfile)
			for row in reader:
				if firstline:
					firstline = False
					continue
				conceptlist.append([row[1],row[2].translate(None,string.punctuation),row[3]])
		return conceptlist

	def generateTaggedDocument(self,filename="dataset/mytext.txt"):
		f = open(filename, 'w')

		with open(self.fileanme, 'rb') as csvfile:
			firstline = True
			reader = csv.reader(csvfile)
			for row in reader:
				if firstline:
					firstline = False
					continue
				f.write(row[1].lower() + '. ' + row[2].lower() + '\n')


if __name__ == '__main__':
	file = CSVFile("dataset/ConceptTeam1.csv")
	print (file.generateTaggedDocument("dataset/data4gensim.txt"))