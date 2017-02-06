import numpy as np
import matplotlib.pyplot as plt

# xindices: N-list of x indices
# yindices: N-list of y indices
# labels: N-list of point labels
# offset: how much you want to off from the exact location (to avoid overlap)
def plotHM(xindices, yindices, labels, xticklabels=None, yticklabels=None, offset=0.15, sort=False):
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
    

if __name__== "__main__":
    main()
