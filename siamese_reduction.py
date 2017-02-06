import gensim
import numpy as np
import GensimEmbedding as emb
import matplotlib.pyplot as plt
import PlotHM as hm

import tensorflow as tf
import inference

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

    if filename is not None:
        plt.savefig(filename)


def get_dist_matrix2(wordlist1, wordlist2):
    dist_matrix = np.ndarray(shape=(len(wordlist1),len(wordlist2)))
    for i in range(len(wordlist1)):
        for j in range(len(wordlist2)):
            dist_matrix[i][j] = emb.model.similarity(wordlist1[i], wordlist2[j])
    return dist_matrix

def get_dist_matrix(wordlist1, wordlist2, opt="cosine"):
    assert opt in ["cosine", "euclidean"]

    X = emb.embeddings
    if opt is "cosine":
        norm = np.sqrt((X**2).sum(axis=1))
        X = X / norm.reshape(-1,1)
        
    dist_matrix = np.ndarray(shape=(len(wordlist1),len(wordlist2)))
    for i in range(len(wordlist1)):
        for j in range(len(wordlist2)):
            dist_matrix[i][j] = np.sqrt(np.sum((X[emb.dictionary[wordlist1[i]],:]-X[emb.dictionary[wordlist2[j]],:])**2))
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


def sampledata():
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

    # connectivity
    C = np.zeros([len(total_words), len(total_words)], dtype=bool)
    for i in range(0, len(total_words), nneigh):
        # C[i:i+nneigh, i:i+nneigh] = True
        C[i, i:i+nneigh] = True
        C[i:i+nneigh, i] = True

    return total_words, C


    # from sklearn.manifold import TSNE
    # word2vis = total_words
    # wordlist_idx = [emb.dictionary[w] for w in word2vis]
    # show = X[wordlist_idx,:]


    # # TSNE 1
    # tsne = TSNE(perplexity=2, n_components=2, n_iter=50000)
    # norm = np.sqrt((show**2).sum(axis=1))
    # unit_emb = show / norm.reshape(-1,1)
    # low_dim_embs = tsne.fit_transform(unit_emb)

    # # TSNE 2
    # # D = get_dist_matrix2(total_words, total_words)
    # plt.imshow(D); plt.show()
    # tsne2 = TSNE(metric="precomputed")
    # low_dim_embs2 = tsne2.fit_transform(D)
    
    # plt.figure(figsize=(18, 18))  # in inches
    # plot_with_labels(low_dim_embs2, word2vis, C)

    
if __name__== "__main__":


    words, connectivity = sampledata()
    word_idxs = [emb.dictionary[w] for w in words]

    X = emb.embeddings
    data = X[word_idxs,:]
    ndata = data.shape[0]
    D = get_dist_matrix(words, words, opt="cosine")


    norm = np.sqrt((data**2).sum(axis=1))
    data = data / norm.reshape(-1,1)

    # setup siamese network

    with tf.Graph().as_default(), tf.Session() as sess:
        siamese = inference.Siamese()

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
          0.001,                 # Base learning rate.
          global_step,   # Current index into the dataset.
          2000,                 # Decay step.
          0.9,                 # Decay rate.
          staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(siamese.loss)
        # optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(siamese.loss)
        saver = tf.train.Saver()
        
        init = tf.global_variables_initializer()
        sess.run(init)

        batch_size = 30

        for step in range(100000):
            s1 = np.random.randint(ndata, size=batch_size)
            s2 = np.random.randint(ndata, size=batch_size)
            x1 = data[s1,:]
            x2 = data[s2,:]
            l = np.array([D[i1, i2] for i1, i2 in zip(s1, s2)])

            # import pdb; pdb.set_trace()
            _, loss_v = sess.run([optimizer, siamese.loss], feed_dict={
                                siamese.x1: x1, 
                                siamese.x2: x2, 
                                siamese.y: l,
                                siamese.keep_prob: 0.5})

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()

            if step % 10 == 0:
                print ('step %d: loss %.3f' % (step, loss_v))

            if step % 1000 == 0:
                low_dim_embs = sess.run(siamese.o1, 
                    feed_dict={siamese.x1:data, siamese.keep_prob: 1})
                plot_with_labels(low_dim_embs, words, connectivity, filename="save/v%05d.png" % step)
                plt.close()







