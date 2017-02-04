import numpy as np
import matplotlib.pyplot as plt

# xindices: N-list of x indices
# yindices: N-list of y indices
# labels: N-list of point labels
# offset: how much you want to off from the exact location (to avoid overlap)
def plotHM(xindices, yindices, labels, xticklabels=None, yticklabels=None, offset=0.15):
    assert len(xindices) == len(yindices) == len(labels)
    
    plt.figure(figsize=(18, 18))  # in inches
    for i in range(len(labels)):
        a = np.random.rand(1)*np.pi*2
        x = xindices[i] + np.cos(a)*offset
        y = yindices[i] + np.sin(a)*offset
        plt.scatter(x, y)
        plt.annotate(labels[i],
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    if xticklabels is not None:
        plt.xticks(range(len(xticklabels)), xticklabels)
    if yticklabels is not None:
        plt.yticks(range(len(yticklabels)), yticklabels)

    plt.grid()
    plt.axes().set_aspect('equal')
    plt.show()


## usage example
def main():
    wordlist = ['charging', 'interaction', 'sleep']
    xind = [0, 0, 2]
    yind = [0, 1, 2]
    plotHM(xind, yind, wordlist, xticklabels=['a','b','c'], offset=0)

if __name__== "__main__":
    main()
