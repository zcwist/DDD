import gensim, logging
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

documents = gensim.models.doc2vec.TaggedLineDocument('dataset/data4gensim.txt')

# for doc in documents:
# 	print doc.tags
# 	print doc.words

model = gensim.models.doc2vec.Doc2Vec(size=100,window=5,min_count=1,dbow_words=0)
# # print model[""]
# model.train_words=False
# model.train_lbls=False
model.build_vocab(documents)
model.train(documents)

print model.infer_vector(['only', 'you', 'can', 'prevent', 'forrest', 'fires'])

# print model.similar_by_word("steering")

# for vec in model.docvecs:
# 	print vec



# coordMat = np.zeros((len(model.docvecs),100))
# for i,vec in enumerate(model.docvecs):
# 	coordMat[i] = vec

# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# low_dim_embs = tsne.fit_transform(coordMat)
# # print low_dim_embs

# plt.figure(figsize=(9,9))
# for i, vec in enumerate(low_dim_embs):
# 	x, y = vec
# 	plt.scatter(x,y,s=400,linewidths=0,alpha=0.8,edgecolors='face')
# 	plt.annotate(i,
# 				xy=(x,y),
# 				xytext=(5,2),
# 				textcoords='offset points',
# 				ha='right',
# 				va='bottom')
# plt.show()

