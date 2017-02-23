import csv
import string
import sys

class CSVFile(object):
	"""docstring for CSVFile"""
	def __init__(self, filename="dataset/ConceptTeam1.csv"):
		super(CSVFile, self).__init__()
		self.filename = filename


	def getContent(self):
		conceptlist = list()

		if (sys.version_info < (3,0)):

			with open(self.filename) as csvfile:
				firstline = True
				reader = csv.reader(csvfile)
				for row in reader:
					if firstline:
						firstline = False
						continue
					conceptlist.append([row[1],row[2].translate(None,string.punctuation),row[3]])
		else:
			with open(self.filename,errors="ignore") as csvfile:
				firstline = True
				reader = csv.reader(csvfile)
				for row in reader:
					if firstline:
						firstline = False
						continue
					def trim_str(str, exclude): return ''.join(ch for ch in str if ch not in exclude)
					ex = string.punctuation
					conceptlist.append([trim_str(row[1],ex),trim_str(row[2],ex),trim_str(row[3],ex)])

		if False:
			from random import shuffle
			shuffle(conceptlist)


		return conceptlist


if __name__ == '__main__':
	file = CSVFile("dataset/ConceptTeamOC12.csv")
	# file = CSVFile("dataset/AllConcepts.csv")
	# file = CSVFile(filename="simplified_data_set.csv")

	print (file.getContent())