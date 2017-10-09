from Compute import ClusterCompare
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def plotBubble(clusterCompareObject, offset=0.25):
    coff = 20 / 9

    curMatrix = clusterCompareObject.getMatrix()
    cnt = [[len(curMatrix[i][j]) for j in range(len(curMatrix[0]))] for i in range(len(curMatrix))]
    nx = len(cnt[0])
    ny = len(cnt)
    xticklabels = [clusterCompareObject.cluster2Mapping[i] for i in range(nx)]
    yticklabels = [clusterCompareObject.cluster1Mapping[i] for i in range(ny)]

    sns.set()
    sns.set_style("ticks")
    plt.figure(figsize=(9 * coff, 9 * coff))  # in inches

    for x in range(nx):
        for y in range(ny):
            # idx = [i for i, pos in enumerate(zip(xindices, yindices)) if pos[0] == x and pos[1] == y]
            # labels2draw = [labels[i] for i in idx]
            # angles = np.linspace(-np.pi / 2, np.pi / 2, num=len(labels2draw) + 1)

            if cnt[y][x] != 0:
                plt.scatter(x, y, s=cnt[y][x] * 400 * coff * coff, c=sns.xkcd_rgb["dark grey"])

                # if len(labels2draw)<2: r = 0
                # else:                  r = offset

                # for l, a in zip(labels2draw, angles):
                # 	# import pdb; pdb.set_trace()
                # 	x_ = x + np.cos(a)*r
                # 	y_ = y + np.sin(a)*r
                # 	plt.scatter(x_, y_,c=sns.xkcd_rgb["pale red"])
                # 	plt.annotate(l,
                # 		 xy=(x_, y_),
                # 		 xytext=(5, 2),
                # 		 textcoords='offset points',
                # 		 ha='right',
                # 		 va='bottom')

    ax = plt.gca()
    ax.set_xticks(range(nx))
    ax.set_yticks(range(ny))

    fig = plt.gcf()
    fig.subplots_adjust(left=0.25, bottom=0.25)

    if xticklabels is not None:
        xticklabels = [x.lower() for x in xticklabels]
        ax.set_xticklabels(xticklabels, rotation=90, ha='right', size=10 * coff)
    else:
        ax.set_xticklabels([""] * nx)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, size=10 * coff)
    else:
        ax.set_yticklabels([""] * ny)

    # plt.grid()
    plt.axes().set_aspect('equal')
    plt.show()

def IndicesCalculation(clusterCompareObject):
    curMatrix = clusterCompareObject.getMatrix()
    row = len(curMatrix)
    col = len(curMatrix[0])
    total = 0.0
    congruency = 0.0
    rowIndices = [0.0 for i in range(row)]
    colIndices = [0.0 for i in range(col)]
    for i in range(row):
        for j in range(col):
            total += len(curMatrix[i][j])
            if (i == j): congruency += len(curMatrix[i][j])
            if (len(curMatrix[i][j]) != 0):
                rowIndices[i] += 1
                colIndices[j] += 1


    congruency /= total
    rowIndices = [i / col for i in rowIndices]
    colIndices = [i / row for i in rowIndices]
    print(total)
    print(congruency)
    print(rowIndices)
    print(colIndices)
if __name__ == '__main__':

    from concept.ConceptManager import ConceptManager as CM
    from dataset.datautils import datapath
    from clusterer.randomcluster import RandomCluster
    from clusterer.humancluster import HumanCluster
    from clusterer.word2veccluster import Word2VecCluster
    cm = CM(filename=datapath("DesInv",1))

    rand = Word2VecCluster(cm, 17)
    hum = HumanCluster(cm)

    a = ClusterCompare(rand, hum)
    a.compare()
    plotBubble(a)
    IndicesCalculation(clusterCompareObject=a)
