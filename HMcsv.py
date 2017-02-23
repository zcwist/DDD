import csv
def exportHM(concept_index,h_labels, m_labels, concept_list,filename):
	filename = "output_csv/" + filename + ".csv"
	with open(filename, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['Concept Index','Concept Name', 'Human Label', 'Machine Label'])
		for i in range(len(concept_list)):
			spamwriter.writerow([concept_index[i],concept_list[i],h_labels[i],m_labels[i]])

def exportHM2(concept_list,m_labels,filename):
	filename = "output_csv/" + filename + ".csv"
	with open(filename, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['Concept Index','Concept Name', 'Description', 'Human Label', 'Machine Label'])
		for i in range(len(concept_list)):
			concept = concept_list[i]
			spamwriter.writerow([i+1,concept.concept,concept.description,concept.category,m_labels[i]])

def exportHMOC(concept_list, m_labels,filename):
	filename = "output_csv/" + filename + ".csv"
	with open(filename, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['Concept Index','Concept Name', 'Description', 'Human Label', 'Machine Label'])
		for i in range(len(concept_list)):
			concept = concept_list[i]
			for cate in concept.category:
				spamwriter.writerow([i+1,concept.concept,concept.description,cate,m_labels[i]])




