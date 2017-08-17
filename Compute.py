import heapq
import csv

class compareableNode():
    def __init__(self, y_pos, x_pos, value):
        self.x = x_pos
        self.y = y_pos
        self.val = value

    def __cmp__(self, other):
        return cmp(other.val, self.val)

class ClusterCompare:

    def __init__(self, cluster1, cluster2):
        self.cluster1 = cluster1
        self.cluster2 = cluster2

    def compare(self):
        len1 = len(self.cluster1.getCluster())
        len2 = len(self.cluster2.getCluster())

        if len1 <= len2:
            smallerCluster = self.cluster1.getCluster()
            lagerCluster = self.cluster2.getCluster()
            smallerMapping = self.cluster1.getMapping()
            largerMapping = self.cluster2.getMapping()
        else:
            smallerCluster = self.cluster2.getCluster()
            lagerCluster = self.cluster1.getCluster()
            smallerMapping = self.cluster2.getMapping()
            largerMapping = self.cluster1.getMapping()

        smallLen = len(smallerCluster)
        largerLen = len(lagerCluster)
        matrix = [[0 for x in range(smallLen)] for y in range(largerLen)]
        pq = []

        # Build the matrix mappings
        for i in range(largerLen):
            for j in range(smallLen):
                matrix[i][j] = self.countSame(lagerCluster[i], smallerCluster[j])
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
            if curNode.y not in RownewPos:
                RownewPos[curNode.y] = defaultY
                defaultY += 1
        result= [[0 for x in range(smallLen)] for y in range(largerLen)]

        # Map the transfered matrix into result
        for i in range(largerLen):
            for j in range(smallLen):
                result[RownewPos[i]][ColnewPos[j]] = matrix[i][j]

        self.createCSV(result, smallerMapping, ColnewPos, largerMapping, RownewPos)




    def countSame(self, conceptList1, conceptList2):
        count = 0
        for i in conceptList1:
            for j in conceptList2:
                if i is j: #possible bug here. I want to test whether these two pointers are pointing to same object
                    count += 1
                    break
        return count

    def createCSV(self, matrix, colOldeIDtoName, colReverseMapping, rowOldIDtoName, rowReverseMapping):
        colNewIDtoName = {}
        rowNewIDtoName = {}
        for key, value in colReverseMapping.iteritems():
            if key in colOldeIDtoName:
                colNewIDtoName[value] = colOldeIDtoName[key]
            else:
                colNewIDtoName[value] = value
        for key, value in rowReverseMapping.iteritems():
            if key in rowOldIDtoName:
                rowNewIDtoName[value] = rowOldIDtoName[key]
            else:
                rowNewIDtoName[value] = value

        with open('clusterMapping.csv', 'wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            colHeading = [" "]
            for col in range(len(matrix[0])):
                colHeading.append(colNewIDtoName[col])
            filewriter.writerow(colHeading)
            for row in range(len(matrix)):
                curRow = matrix[row]
                y_name = rowNewIDtoName[row]
                curRow.insert(0, y_name)
                filewriter.writerow(curRow)



if __name__ == '__main__':

    from concept.ConceptManager import ConceptManager as CM
    from dataset.datautils import datapath
    from clusterer.randomcluster import RandomCluster
    from clusterer.humancluster import HumanCluster
    cm = CM(filename=datapath("DesInv",1))

    rand = RandomCluster(cm, 7)
    hum = HumanCluster(cm)

    print(ClusterCompare(rand, hum).compare())