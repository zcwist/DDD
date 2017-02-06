import csv
def exportHM(h_labels, m_labels, concept_list,filename):
	filename = "output_csv/" + filename + ".csv"
	with open(filename, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['concept name', 'human labels', 'machine labels'])
		for i in range(len(concept_list)):
			spamwriter.writerow([concept_list[i],h_labels[i],m_labels[i]])

