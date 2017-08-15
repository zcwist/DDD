import csv
import subprocess
def summary(filename):
	member_list = []
	concept_list = []
	cluster_list = []
	with open(filename) as csvfile:
		firstline = True
		reader = csv.reader(csvfile)
		for row in reader:
			if firstline:
				firstline = False
				continue
			# conceptlist.append([row[1],row[2].translate(None,string.punctuation),row[3]])
			if row[0] not in member_list:
				member_list.append(row[0])
			if row[1] not in concept_list:
				concept_list.append(row[1])
			if row[3] not in cluster_list: 
				cluster_list.append(row[3])
	print "%d,%d,%d" %(len(member_list),len(concept_list),len(cluster_list))
	write_to_clipboard("%d,%d,%d" %(len(member_list),len(concept_list),len(cluster_list)))

def write_to_clipboard(output):
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode('utf-8'))

if __name__ == '__main__':
	for x in range(6):
		summary("DesInv/team"+str(x+1)+".csv")
