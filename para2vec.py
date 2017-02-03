import GensimEmbedding as emb
from ConceptManager import ConceptManager as CM
import matplotlib.pyplot as plt
from Plot import Plot

import collections
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer("para_embedding_size", 20, "The embedding dimension size of paragraph vector")
flags.DEFINE_integer("batch_size", 5,
					 "Number of training paragraph examples processed per step "
					 "(size of a minibatch).")
flags.DEFINE_integer("window_size", 3,
					 "Size of sampling window")
flags.DEFINE_integer("cluster_size", 13,
					 "Size of cluster for k means")
flags.DEFINE_integer("num_steps",10000, "The number of training times")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
					 "Negative samples per training example.")
flags.DEFINE_bool("random_order", True,"random order of data set")

FLAGS = flags.FLAGS

class Options(object):
	"""docstring for Option"""
	def __init__(self):
		self.emb_dim = FLAGS.embedding_size
		self.para_emb_dim = FLAGS.para_embedding_size
		self.batch_size = FLAGS.batch_size
		self.window_size = FLAGS.window_size
		self.num_steps = FLAGS.num_steps
		self.learning_rate = FLAGS.learning_rate
		self.num_neg_samples = FLAGS.num_neg_samples
		self.cluster_size = FLAGS.cluster_size
		self.random = FLAGS.random_order

class Para2vec(object):
	"""docstring for Para2vec"""
	def __init__(self, conceptManager, options, session):
		super(Para2vec, self).__init__()
		self.cm = conceptManager
		self.concept_list = self.cm.conceptList
		self._para_size = self.cm.concept_size
		self._options = options
		self._session = session
		self._load_word_embeddings()
		self.para_index = 0
		self.word_index = 0
		self.graph = tf.Graph()
		self.build_graph()

	def _load_word_embeddings(self):
		self.word_embeddings = emb.embeddings
		self.word_dictionary = emb.dictionary

	 
	def generate_batch(self,batch_size, window_size):
		"""Generate batch

		Returns:
		para_examples, word_examples, labels
		para_examples:[para_id]
		word_examples:[word_id*(window_size-1)]
		labels: word_id
		""" 

		#para_examples: [para_id]
		para_examples = np.ndarray(shape=(batch_size,1), dtype=np.int32)

		#word_examples: [word_id*(window_size-1)]
		word_examples = np.ndarray(shape=(batch_size,window_size - 1), dtype=np.int32)
		labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
		paragraph = self.concept_list[self.para_index].fullConcept()
		for i in range(batch_size):
			# if there is enough words for this sample
			if ((self.word_index + window_size) > len(paragraph)):
				self.para_index = (self.para_index + 1) % len(self.concept_list)
				self.word_index = 0
				paragraph = self.concept_list[self.para_index].fullConcept()

			para_examples[i][0] = self.para_index

			for j in range(window_size - 1):
				# print self.word_dictionary[paragraph[self.word_index+j].lower()]
				# print Embedding.wordVec(paragraph[self.word_index+j].lower())

				# word_examples[i][j] = self.word_dictionary[paragraph[self.word_index+j].lower()]
				word_examples[i][j] = emb.wordIndex(paragraph[self.word_index+j].lower())
			# labels[i] = self.word_dictionary[paragraph[self.word_index+window_size-1].lower()]
			try:
				labels[i] = emb.wordIndex(paragraph[self.word_index+window_size-1].lower())
			except Exception as e:
				print ("i",i)
				print ("paragraph",paragraph)
				print ("word index", self.word_index+window_size-1)
				raise e
			self.word_index = self.word_index + 1

		return para_examples, word_examples, labels

	def build_graph(self):
		opts = self._options

		# Input data
		self.para_examples = tf.placeholder(tf.int32, shape=[opts.batch_size,1], name="para_examples")
		self.word_examples = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.window_size-1], name="word_examples")
		self.labels = tf.placeholder(tf.int32, shape=[opts.batch_size,1], name="labels")

		

		#Para Embedding: [para_size, emb_dim]
		para_emb = tf.Variable(
			tf.random_uniform(
				[self._para_size,
				opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
			trainable = True,
			name="w_para")
		self._para_emb = para_emb

		#Word Embedding: [vocab_size, emb_dim]
		word_emb = tf.Variable(self.word_embeddings, trainable = False, name="w_word")
		self._word_emb = word_emb


		#Embedding for examples calculation

		para_embed = tf.nn.embedding_lookup(para_emb,self.para_examples) #[[[emb_dim]]*batch_size]
		para_embed = tf.reduce_sum(para_embed,1) #[[emb_dim]*batch_size]
		# self.para_embed = para_embed

		words_embed = tf.nn.embedding_lookup(word_emb,self.word_examples) # sum of embeddings of word in examples [[[emb_dim]*(window_size-1)]*batch_size]
		words_embed = tf.reduce_sum(words_embed,1) #[[emb_dim]*batch_size]
		# self.word_embed = word_embed
		
		# Embeddings for examples: [batch_size, emb_dim]
		embed = tf.add(para_embed, words_embed) # sum of embedding of words and para [[emb_dim]*batch_size]
		# embed = tf.divide(embed, opts.window_size)

		opts.vocab_size = len(self.word_dictionary)

		#Softmax weight: [vocab_size, emb_dim]. Transposed
		w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")
		self._w_out = w_out

		#Softmax bias: [vocab_size]
		b_out = tf.Variable(tf.zeros([opts.vocab_size]), name="b_out")
		self._b_out = b_out

		tf.global_variables_initializer().run()

		loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=w_out, 
				biases=b_out, 
				inputs=embed, 
				labels=self.labels, 
				num_sampled=opts.num_neg_samples, 
				num_classes=opts.vocab_size, 
				name="loss"))
		self.loss = loss

		optimizer = tf.train.GradientDescentOptimizer(opts.learning_rate).minimize(loss)
		self.trainer = optimizer


	def train(self):
		"""Train the model"""
		opts = self._options

		for step in range(opts.num_steps):
			para_examples, word_examples, labels = self.generate_batch(opts.batch_size, opts.window_size)

			feed_dict = {self.para_examples:para_examples, self.word_examples:word_examples, self.labels:labels}
			_, loss_val = self._session.run([self.trainer, self.loss], feed_dict=feed_dict)

			if step%100 == 0:
				print ("loss at step ", step,":", loss_val)
	
	def drawWithTag(self):
		from sklearn.manifold import TSNE
		import matplotlib.pyplot as plt

		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		norm = tf.sqrt(tf.reduce_sum(tf.square(self._para_emb), 1, keep_dims=True))
		normalized_embeddings = self._para_emb / norm
		low_dim_embs = tsne.fit_transform(normalized_embeddings.eval())
		
		for i, concept in enumerate(self.concept_list):
			concept.setLowEmb(low_dim_embs[i])

		Plot(self.cm).drawWithTag()

	def draw(self):
		from sklearn.manifold import TSNE
		import matplotlib.pyplot as plt

		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		norm = tf.sqrt(tf.reduce_sum(tf.square(self._para_emb), 1, keep_dims=True))
		normalized_embeddings = self._para_emb / norm
		low_dim_embs = tsne.fit_transform(normalized_embeddings.eval())
		
		for i, concept in enumerate(self.concept_list):
			concept.setLowEmb(low_dim_embs[i])

		Plot(self.cm).draw()

	def clustering(self):
		opts = self._options

		from sklearn.cluster import KMeans, AgglomerativeClustering
		# clustering = KMeans(n_clusters=opts.cluster_size, random_state=0).fit(self._para_emb.eval())
		clustering = AgglomerativeClustering(n_clusters=14, affinity='cosine',linkage='complete').fit(self._para_emb.eval())

		self.write2csv(clustering.labels_)
		

		#sort the index with size decreasing
		cluster_dic = dict() #{cluster_index:num_of_concepts}
		cate_dict = dict() #{cate_index:num_of_concepts}

		for i in range(len(self.concept_list)):
			if clustering.labels_[i] not in cluster_dic.keys():
				cluster_dic[clustering.labels_[i]] = 0
			if self.concept_list[i].getCategory() not in cate_dict.keys():
				cate_dict[self.concept_list[i].getCategory()] = 0
			cluster_dic[clustering.labels_[i]] = cluster_dic[clustering.labels_[i]] + 1
			cate_dict[self.concept_list[i].getCategory()] = cate_dict[self.concept_list[i].getCategory()] + 1

		import operator

		cluster_dict_sorted = sorted(cluster_dic.items(),key=operator.itemgetter(1),reverse=True)
		cate_dict_sorted = sorted(cate_dict.items(), key=operator.itemgetter(1),reverse=True)

		cluster_map = dict() #{old_cluster_index:new_cluster_index}
		cate_map = dict() #{old_cate_index:new_cate_index}

		for i,cluster_index in enumerate(cluster_dict_sorted):
			cluster_map[cluster_index[0]] = i 
		for i,cate_index in enumerate(cate_dict_sorted):
			cate_map[cate_index[0]] = i

		bubble_data = dict() #{(x,y):freq}
		points = list()
		for i in range(len(self.concept_list)):
			x = cate_map[self.concept_list[i].getCategory()]
			y = cluster_map[clustering.labels_[i]]
			if (x,y) not in points:
				points.append((x,y))
				bubble_data[(x,y)] = 0
			bubble_data[(x,y)] = bubble_data[(x,y)] + 1

		self.draw_bubble(bubble_data)

		# cate_index_list = list()
		# bubble_data = dict() #{(x,y):freq}
		# points = list()
		# for i in range(len(self.concept_list)):
		# 	x = self.cm.getCateIndex(self.concept_list[i].getCategory())
		# 	y = kmeans.labels_[i]
		# 	if (x,y) not in points:
		# 		points.append((x,y))
		# 		bubble_data[(x,y)] = 0
		# 	bubble_data[(x,y)] = bubble_data[(x,y)] + 1

		# self.draw_bubble(bubble_data)



		# for i in range(len(self.concept_list)):
		# 	print (self.concept_list[i].conceptName(),self.concept_list[i].getCategory(),kmeans.labels_[i])
		# print kmeans.labels_

	def draw_bubble(self,dataset):
		plt.figure(figsize=(5,5))
		for point in dataset:
			x = point[0]
			y = point[1]
			plt.scatter(x,y,s=50*dataset[point],linewidths=0,alpha=0.5,edgecolors='face')

		plt.show()

	def draw_dendrogram(self):
		from scipy.cluster.hierarchy import dendrogram, linkage
		Z = linkage(self._para_emb.eval(),'single',metric='cosine')

		concept_name = list()
		for concept in self.concept_list:
			concept_name.append(concept.conceptName())



		plt.figure(figsize=(9,9))
		dendrogram(Z,
			labels=concept_name,
			orientation='right',
			count_sort='descendent')
		fig = plt.gcf()
		fig.subplots_adjust(left=0.25)
		plt.show()

	def write2csv(self,labels_):
		import csv
		with open('output_csv/para2vec.csv','wb') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for i in range(len(self.concept_list)):
				spamwriter.writerow([self.concept_list[i].conceptName(),self.concept_list[i].getCategory(),labels_[i]])

class Para2VecConc(Para2vec):
	"""docstring for Para2VecConc"""
	def __init__(self, conceptManager, options, session):
		super(Para2VecConc, self).__init__(conceptManager, options, session)

	def build_graph(self):
		opts = self._options

		# Input data
		self.para_examples = tf.placeholder(tf.int32, shape=[opts.batch_size,1], name="para_examples")
		self.word_examples = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.window_size-1], name="word_examples")
		self.labels = tf.placeholder(tf.int32, shape=[opts.batch_size,1], name="labels")
	

		#Para Embedding: [para_size, emb_dim]
		para_emb = tf.Variable(
			tf.random_uniform(
				[self._para_size,
				opts.para_emb_dim], -0.5 / opts.para_emb_dim, 0.5 / opts.para_emb_dim),
			trainable = True,
			name="w_para")
		self._para_emb = para_emb

		#Word Embedding: [vocab_size, emb_dim]
		word_emb = tf.Variable(self.word_embeddings, trainable = False, name="w_word")
		self._word_emb = word_emb


		#Embedding for examples calculation
		para_embed = tf.nn.embedding_lookup(para_emb,self.para_examples) #[[[emb_dim]]*batch_size]
		para_embed = tf.reduce_sum(para_embed,1) #[[emb_dim]*batch_size]

		words_embed = tf.nn.embedding_lookup(word_emb,self.word_examples) # sum of embeddings of word in examples [[[emb_dim]*(window_size-1)]*batch_size]
		words_embed = tf.reshape(words_embed,[opts.batch_size,opts.emb_dim*(opts.window_size-1)])

		# Embeddings for examples: [batch_size, emb_dim]
		embed = tf.concat(1,[para_embed,words_embed])

		opts.vocab_size = len(self.word_dictionary)

		#Softmax weight: [vocab_size, emb_dim]. Transposed
		w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim*(opts.window_size-1)+opts.para_emb_dim]), name="w_out")
		self._w_out = w_out

		#Softmax bias: [vocab_size]
		b_out = tf.Variable(tf.zeros([opts.vocab_size]), name="b_out")
		self._b_out = b_out

		tf.global_variables_initializer().run()

		loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=w_out, 
				biases=b_out, 
				inputs=embed, 
				labels=self.labels, 
				num_sampled=opts.num_neg_samples, 
				num_classes=opts.vocab_size, 
				name="loss"))
		self.loss = loss

		optimizer = tf.train.GradientDescentOptimizer(opts.learning_rate).minimize(loss)
		self.trainer = optimizer

class Para2VecPVDBOW(Para2vec):
	"""docstring for Para2Vec using PV-DBOW model"""
	def __init__(self, conceptManager, options, session):
		super(Para2VecPVDBOW, self).__init__(conceptManager, options, session)

	def generate_batch(self, batch_size, window_size):
		"""Generate batch for PV-DBOW
		sample a text window
		sample a random word from the text window and 
		form a classification task given the Paragraph Vector.

		Returns:
		para_examples, word_examples, labels
		para_examples:[para_id]
		labels: [word_id]
		"""
		import random

		#para_examples: [para_id]
		para_examples = np.ndarray(shape=(batch_size,1), dtype=np.int32)

		#labels: [word_id]
		labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)

		paragraph = self.concept_list[self.para_index].fullConcept()
		for i in range(batch_size):
			# if there is enough words for this sample
			if ((self.word_index + window_size) > len(paragraph)):
				self.para_index = (self.para_index + 1) % len(self.concept_list)
				self.word_index = 0
				paragraph = self.concept_list[self.para_index].fullConcept()

			para_examples[i][0] = self.para_index

			label_pos = random.randint(0, window_size-1) # get a word randomly from the window
			try:
				labels[i] = emb.wordIndex(paragraph[self.word_index+label_pos].lower())
			except Exception as e:
				print ("i",i)
				print ("paragraph",paragraph)
				print ("word index", self.word_index+window_size-1)
				raise e
			self.word_index = self.word_index + 1
		return para_examples, labels

	def build_graph(self):
		opts = self._options

		#Input data
		self.para_examples = tf.placeholder(tf.int32, shape=[opts.batch_size,1], name="para_examples")
		self.labels = tf.placeholder(tf.int32, shape=[opts.batch_size,1], name="labels")

		#Para Embedding: [para_size, para_embedding_size]
		para_emb = tf.Variable(
			tf.random_uniform(
				[self._para_size,
				opts.para_emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
			trainable = True,
			name="w_para")
		self._para_emb = para_emb

		#Embedding for examples calculation
		para_embed = tf.nn.embedding_lookup(para_emb,self.para_examples) #[[[para_embedding_size]]*batch_size]
		para_embed = tf.reduce_sum(para_embed,1) #[[para_embedding_size]*batch_size]
		# Embeddings for examples: [batch_size, para_embedding_size]
		embed = para_embed

		opts.vocab_size = len(self.word_dictionary)

		#Softmax weight: [vocab_size, emb_dim]. Transposed
		w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.para_emb_dim]), name="w_out")
		self._w_out = w_out

		#Softmax bias: [vocab_size]
		b_out = tf.Variable(tf.zeros([opts.vocab_size]), name="b_out")
		self._b_out = b_out

		tf.global_variables_initializer().run()

		loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=w_out, 
				biases=b_out, 
				inputs=embed, 
				labels=self.labels, 
				num_sampled=opts.num_neg_samples, 
				num_classes=opts.vocab_size, 
				name="loss"))
		self.loss = loss

		optimizer = tf.train.GradientDescentOptimizer(opts.learning_rate).minimize(loss)
		self.trainer = optimizer

	def train(self):
		"""Train the model"""

		opts = self._options

		for step in range(opts.num_steps):
			para_examples, labels = self.generate_batch(opts.batch_size, opts.window_size)

			feed_dict = {self.para_examples:para_examples,self.labels:labels}
			_, loss_val = self._session.run([self.trainer, self.loss], feed_dict=feed_dict)

			if step%100 == 0:
				print ("loss at step ", step,":", loss_val)
	

def testTeam1WithSum(size):
	"""For ConceptTeam1.csv"""
	opts = Options()
	with tf.Graph().as_default(), tf.Session() as session:
		model = Para2vec(CM(size),opts,session)
		model.train()
		# model.clustering()
		# model.drawWithTag()
		model.draw_dendrogram()	

def testAllWithSum():
	"""For AllConcepts.csv"""
	opts = Options()
	with tf.Graph().as_default(), tf.Session() as session:
		model = Para2vec(CM(1121,filename="dataset/AllConcepts.csv"),opts,session)
		model.train()
		model.clustering()
		model.draw()	

def testTeam1WithConc(size):
	"""For ConceptTeam1.csv with Para2vecConc""" 
	opts = Options()
	with tf.Graph().as_default(), tf.Session() as session:
		model = Para2VecConc(CM(size),opts,session)
		model.train()
		# model.clustering()
		# model.drawWithTag()
		model.draw_dendrogram()	

def testAllWithConc(size):
	"""For AllConcepts.csv"""
	opts = Options()
	with tf.Graph().as_default(), tf.Session() as session:
		model = Para2vec(CM(size,filename="dataset/AllConcepts.csv"),opts,session)
		model.train()
		model.clustering()
		model.draw()	

def testTeam1WithPVDBWO(size):
	"""For team 1, using pv-dbow model"""
	opts = Options()
	with tf.Graph().as_default(), tf.Session() as session:
		model = Para2VecPVDBOW(CM(size),opts,session)
		model.train()
		# model.clustering()
		model.draw_dendrogram()


if __name__ == '__main__':
	# testTeam1WithSum(20)
	testTeam1WithConc(20)
	
	# testAllWithSum(1121)
	# testAllWithConc(1121)

	# testTeam1WithPVDBWO(20)


		