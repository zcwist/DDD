import gensim
import numpy as np
import GensimEmbedding as emb
import matplotlib.pyplot as plt
# import tensorflow as tf

def plot_with_labels(low_dim_embs, labels, connectivity=None, filename=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    if connectivity is not None:
        for i in range(connectivity.shape[0]-1):
            for j in range(i+1, connectivity.shape[1]):
                if connectivity[i, j]:
                    plt.plot([low_dim_embs[i,0], low_dim_embs[j,0]], [low_dim_embs[i,1], low_dim_embs[j,1]])

    plt.show()

    if filename is not None:
        plt.savefig(filename)


def get_dist_matrix2(wordlist1, wordlist2):
    dist_matrix = np.ndarray(shape=(len(wordlist1),len(wordlist2)))
    for i in range(len(wordlist1)):
        for j in range(len(wordlist2)):
            dist_matrix[i][j] = emb.model.similarity(wordlist1[i], wordlist2[j])
    return dist_matrix


def show_graph(xy, labels, connectivity):

    for i in range(xy.shape[0]):
        x, y = xy[i, :]
        plt.scatter(x, y)
        plt.annotate(labels[i],
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    for i in range(connectivity.shape[0]-1):
        for j in range(i+1, connectivity.shape[1]):
            if connectivity[i, j]:
                plt.plot([xy[i,0], xy[j,0]], [xy[i,1], xy[j,1]])

    plt.show()

def dist_from_humancate():
    wordlist1 = [
    'steering',
    'phone',
    'voice',
    'gps',
    'charging',
    'interaction',
    'sleep',
    'sensor',
    'navigation',  
    'parking',
    'command',
    'keyboard',
    'wireless',
    'connectivity']

    human_assign = [
    'control', 
    'control', 
    'control', 
    'control', 
    'control', 
    'connectivity', 
    'comfort', 
    'navigation', 
    'navigation', 
    'parking', 
    'control', 
    'control', 
    'connectivity', 
    'connectivity']

    humanlist = list(set(human_assign))

    d = get_dist_matrix2(humanlist, wordlist1)
    machine_idx = np.argmin(d, axis=0)

    cate_to_idx = {w:i for i, w in enumerate(humanlist)}
    human_idx = [cate_to_idx[w] for w in human_assign]

    plt.figure(figsize=(18, 18))  # in inches
    for i in range(len(human_idx)):
        a = np.random.rand(1)*np.pi*2
        x = human_idx[i] + np.cos(a)*0.15
        y = machine_idx[i] + np.sin(a)*0.15
        plt.scatter(x, y)
        plt.annotate(wordlist1[i],
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    
    plt.xticks(range(len(humanlist)), humanlist)
    plt.grid()
    plt.axes().set_aspect('equal')
    plt.show()
    import  pdb; pdb.set_trace()






def draw_with_neighbors():
    wordlist = [
    'steering',
    'phone',
    'voice',
    'gps',
    'charging',
    'interaction',
    'sleeping',
    'sensor',
    'navigation',  
    'parking',
    'command',
    'keyboard',
    'wireless',
    'connectivity']

    nneigh = 4

    X = emb.embeddings
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=nneigh).fit(X)
    neighbors = []
    for w in wordlist:
        distances, indices = nbrs.kneighbors(X[emb.dictionary[w],None,:])
        similar_words = [emb.model.wv.index2word[k] for k in indices.tolist()[0]]
        neighbors.append(similar_words)

    total_words = [k for w in neighbors for k in w]

    C = np.zeros([len(total_words), len(total_words)], dtype=bool)
    for i in range(0, len(total_words), nneigh):
        # C[i:i+nneigh, i:i+nneigh] = True
        C[i, i:i+nneigh] = True
        C[i:i+nneigh, i] = True


    from sklearn.manifold import TSNE

    word2vis = total_words

    wordlist_idx = [emb.dictionary[w] for w in word2vis]
    show = X[wordlist_idx,:]

    tsne = TSNE(perplexity=2, n_components=2, n_iter=50000)

    # import pdb; pdb.set_trace()

    norm = np.sqrt((show**2).sum(axis=1))
    unit_emb = show / norm.reshape(-1,1)
    low_dim_embs = tsne.fit_transform(unit_emb)

    plt.figure(figsize=(18, 18))  # in inches
    plot_with_labels(low_dim_embs, word2vis, C)






def emb_tester():
    

    wordlist = [
    'steering',
    'phone',
    'voice',
    'gps',
    'charging',
    'interaction',
    'sleeping',
    'sensor',
    'navigation',
    'parking',
    'command',
    'keyboard',
    'wireless',
    'connectivity']

    human_category = [
    'control', 
    'control', 
    'control', 
    'control', 
    'control', 
    'connectivity', 
    'comfort', 
    'navigation', 
    'navigation', 
    'parking', 
    'control', 
    'control', 
    'connectivity', 
    'connectivity']

    





    



    from sklearn.manifold import TSNE

    word2vis = wordlist+list(set(human_category))

    wordlist_idx = [emb.dictionary[w] for w in word2vis]
    show = X[wordlist_idx,:]

    tsne = TSNE(perplexity=2, n_components=2, n_iter=5000)

    # import pdb; pdb.set_trace()

    norm = np.sqrt((show**2).sum(axis=1))
    unit_emb = show / norm.reshape(-1,1)
    low_dim_embs = tsne.fit_transform(unit_emb)

    plt.figure(figsize=(18, 18))  # in inches
    plot_with_labels(low_dim_embs, word2vis)

    import pdb; pdb.set_trace()
    C = np.ones((len(word2vis), len(word2vis)), dtype=bool)
    plot_with_labels(low_dim_embs, word2vis, C)




    # dist_matrix = get_dist_matrix(wordlist)
    # tsne2 = TSNE(metric="precomputed")
    # low2 = tsne2.fit_transform(dist_matrix+1)
    # plot_with_labels(low2, wordlist)
    





    
    # for i, concept in enumerate(self.concept_list):
        # concept.setLowEmb(low_dim_embs[i])
    # Plot(self.cm).drawWithTag()

    # human_category = ['control', 'connectivity', 'comfort', 'navigation', 'parking', 'entry', 'signal', 'leisure', 'display', 'cloud', 'communication', 'voice', 'safety', 'warning']

    # import matplotlib.pyplot as plt

    # dist_matrix = np.ndarray(shape=(len(wordlist),len(wordlist)))
    # for i in range(len(wordlist)):
    #     for j in range(len(wordlist)):
    #         dist_matrix[i][j] = model.similarity(wordlist[i],wordlist[j])


# def dendrogram():
#     from scipy.cluster.hierarchy import dendrogram, linkage
#     Z = linkage(dist_matrix,'average',metric='cosine')

#     plt.figure(figsize=(9,9))
#     dendrogram(Z,labels=wordlist)
#     plt.show()
# dendrogram()

if __name__== "__main__":

    dist_from_humancate()
    emb_tester()
    draw_with_neighbors()


