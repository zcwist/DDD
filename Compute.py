import heapq
import csv

class compareableNode():
    def __init__(self, y_pos, x_pos, value):
        self.x = x_pos
        self.y = y_pos
        self.val = value

    def __cmp__(self, other):
        return cmp(len(other.val), len(self.val))

class ClusterCompare:

    def __init__(self, cluster1, cluster2):
        #cluster 1 is the row(y axis) and cluster 2 is the column (x axis)
        self.cluster1 = cluster1
        self.cluster2 = cluster2
        self.ClusterPair = {}
        self.matrix =[[]]
        self.cluster1Mapping = {}
        self.cluster2Mapping = {}


    def compare(self):
        len1 = len(self.cluster1.getCluster())
        len2 = len(self.cluster2.getCluster())
        rowCluster = self.cluster1.getCluster()
        colCluster = self.cluster2.getCluster()

        matrix = [[[] for x in range(len2)] for y in range(len1)]
        pq = []

        # Build the matrix mappings
        for i in range(len1):
            for j in range(len2):
                matrix[i][j] = self.countSame(rowCluster[i], colCluster[j])
                heapq.heappush(pq, compareableNode(i, j, matrix[i][j]))

        RownewPos = {}
        ColnewPos = {}
        defaultX = 0
        defaultY = 0
        pqCopy = list(pq)
        #Obtain the new mapping of the matrix arrange the largest Number in the diagonal
        while len(pq) > 0:
            curNode = heapq.heappop(pq)
            if curNode.y not in RownewPos and curNode.x not in ColnewPos:
                RownewPos[curNode.y] = defaultY
                ColnewPos[curNode.x] = defaultX
                defaultY += 1
                defaultX += 1

        #if the length is not symmetric, further explore the larger cluster and assign new arrangement(large -> small)

        while len(pqCopy) != 0:
            curNode = heapq.heappop(pqCopy)
            if len1 > len2:
                if curNode.y not in RownewPos:
                    RownewPos[curNode.y] = defaultY
                    defaultY += 1
            elif len1 < len2:
                if curNode.x not in ColnewPos:
                    ColnewPos[curNode.x] = defaultX
                    defaultX += 1
            else:
                break


        result= [[[] for x in range(len2)] for y in range(len1)]

        # Map the transfered matrix into result
        for i in range(len1):
            for j in range(len2):
                result[RownewPos[i]][ColnewPos[j]] = matrix[i][j]
                if (len(matrix[i][j]) != 0):
                    self.ClusterPair[(RownewPos[i], ColnewPos[j])] = matrix[i][j]


        self.matrix = result
        self.updateMapping(self.cluster2.getMapping(), ColnewPos, self.cluster1.getMapping(), RownewPos)




    def countSame(self, conceptList1, conceptList2):
        sameConcept = []
        for i in conceptList1:
            for j in conceptList2:
                if i is j: #possible bug here. I want to test whether these two pointers are pointing to same object
                    sameConcept.append(i)
                    break
        return sameConcept

    def updateMapping(self, colOldIDtoName, colReverseMapping, rowOldIDtoName, rowReverseMapping):
        for key, value in colReverseMapping.iteritems():
            if key in colOldIDtoName:
                self.cluster2Mapping[value] = colOldIDtoName[key]
            else:
                self.cluster2Mapping[value] = value
        for key, value in rowReverseMapping.iteritems():
            if key in rowOldIDtoName:
                self.cluster1Mapping[value] = rowOldIDtoName[key]
            else:
                self.cluster1Mapping[value] = value


    def createCSV(self):

        # for key, value in colReverseMapping.iteritems():
        #     if key in colOldeIDtoName:
        #         self.cluster2Mapping[value] = colOldeIDtoName[key]
        #     else:
        #         self.cluster2Mapping[value] = value
        # for key, value in rowReverseMapping.iteritems():
        #     if key in rowOldIDtoName:
        #         self.cluster1Mapping[value] = rowOldIDtoName[key]
        #     else:
        #         self.cluster1Mapping[value] = value

        with open('HMPlot.csv', 'wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            colHeading = [" "]
            for col in range(len(self.matrix[0])):
                colHeading.append(self.cluster2Mapping[col])
            filewriter.writerow(colHeading)
            for row in range(len(self.matrix)):
                curRow = [len(i) for i in self.matrix[row]]
                y_name = self.cluster1Mapping[row]
                curRow.insert(0, y_name)
                filewriter.writerow(curRow)

        with open('OverView.csv', 'wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            colHeading = [self.cluster1.getType(), self.cluster2.getType(), "Concept Name", "Concpet Description"]
            filewriter.writerow(colHeading)
            for key, value in self.ClusterPair.iteritems():
                row = []
                row.append(self.cluster1Mapping[key[0]])
                row.append(self.cluster2Mapping[key[1]])
                for concept in value:
                    newRow = list(row)
                    newRow.append(concept.conceptName())
                    newRow.append(concept.getDescription())
                    filewriter.writerow(newRow)

if __name__ == '__main__':

    from concept.ConceptManager import ConceptManager as CM
    from dataset.datautils import datapath
    from clusterer.randomcluster import RandomCluster
    from clusterer.humancluster import HumanCluster
    cm = CM(filename=datapath("DesInv",1))

    rand = RandomCluster(cm, 7)
    hum = HumanCluster(cm)

    a = ClusterCompare(rand, hum)
    a.compare()
    a.createCSV()