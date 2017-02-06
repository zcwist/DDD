import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# xindices: N-list of x indices
# yindices: N-list of y indices
# labels: N-list of point labels
# offset: how much you want to off from the exact location (to avoid overlap)
def plotHM(xindices, yindices, labels, xticklabels=None, yticklabels=None, offset=0.25, sort=False):
    assert len(xindices) == len(yindices) == len(labels)

    if sort:
        convx, convy = get_sorted_idx(xindices, yindices)
        xindices = [convx[x] for x in xindices]
        yindices = [convy[y] for y in yindices]

        if xticklabels is not None:
            xticklabels_ = xticklabels[:]
            for i,tl in enumerate(xticklabels): xticklabels_[convx[i]] = tl
            xticklabels = xticklabels_
        if yticklabels is not None:
            yticklabels_ = yticklabels[:]
            for i,tl in enumerate(yticklabels): yticklabels_[convy[i]] = tl
            yticklabels = yticklabels_

    sns.set()
    sns.set_style("darkgrid")
    plt.figure(figsize=(9, 9))  # in inches

    # count first for better visualization
    nx = max(xindices)+1
    ny = max(yindices)+1
    # cnt = np.zeros([ny, nx])
    # for x, y in zip(xindices, yindices): cnt[y,x] += 1

    for x in range(nx):
        for y in range(ny):
            idx = [i for i,pos in enumerate(zip(xindices, yindices)) if pos[0]==x and pos[1]==y]
            labels2draw = [labels[i] for i in idx]
            angles = np.linspace(-np.pi/2, np.pi/2, num=len(labels2draw)+1)

            if len(labels2draw)<2: r = 0
            else:                  r = offset

            for l, a in zip(labels2draw, angles):
                # import pdb; pdb.set_trace()
                x_ = x + np.cos(a)*r
                y_ = y + np.sin(a)*r
                plt.scatter(x_, y_)
                plt.annotate(l,
                     xy=(x_, y_),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    ax = plt.gca()
    ax.set_xticks(range(nx))
    ax.set_yticks(range(ny))
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=20, ha='right')
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    # plt.grid()
    plt.axes().set_aspect('equal')
    plt.show()

def get_sorted_idx(xindices, yindices):
    nx = max(xindices)+1
    ny = max(yindices)+1

    cnt = np.zeros([ny, nx])
    for x, y in zip(xindices, yindices):
            cnt[y,x] += 1
    chk = cnt.copy()

    oldx = list(range(nx))
    oldy = list(range(ny))
    newx = []
    newy = []
    while len(oldx)*len(oldy) > 1:
        r, c = np.unravel_index(np.argmax(chk), chk.shape)
        if len(oldx)>1: 
            newx.append(oldx.pop(c))
            chk = np.delete(chk, c, axis=1)
        if len(oldy)>1: 
            newy.append(oldy.pop(r))
            chk = np.delete(chk, r, axis=0)
    newx += oldx
    newy += oldy
    
    converterx = {old:newx.index(old) for old in range(nx)}
    convertery = {old:newy.index(old) for old in range(ny)}
    return converterx, convertery


## usage example
def main():
    xind = np.random.randint(6, size=40).tolist()+[2,2,2,2,1]
    yind = np.random.randint(8, size=40).tolist()+[3,3,3,3,2]
    labels = [str(x)+str(y) for x, y in zip(xind, yind)]
    xtl = list(range(6))
    ytl = list(range(8))
    plotHM(xind, yind, labels, xtl, ytl, offset=0.15)
    plotHM(xind, yind, labels, xtl, ytl, offset=0.15, sort=True)

## data sample
def data_sample():
    from swsc import SWSC
    from ConceptManager import ConceptManager as CM
    cm = CM(80)
    swsc = SWSC(cm)
    m_labels = swsc.hierachical_clustering()
    h_labels = cm.category_index_list
    concept_list = cm.concept_name_list

    plotHM(h_labels,m_labels,concept_list,xticklabels=cm.categoryList,sort=True)
    

if __name__== "__main__":
    data_sample()
