import Embedding
from ConceptManager import ConceptManager as CM
from Plot import Plot

import collections
import numpy as np
import tensorflow as tf
import math

flags = tf.app.flags

flags.DEFINE_integer("embedding_size", 128, "The embedding dimension size.")
flags.DEFINE_integer("paragraph_embedding_size", 20, "The embedding dimension size.")
flags.DEFINE_integer("batch_size", 5,
                     "Number of training paragraph examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("window_size", 3,
                     "Size of sampling window")
flags.DEFINE_integer("num_steps",20000, "The number of training times")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")

FLAGS = flags.FLAGS

class Options(object):
    """docstring for Option"""
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.window_size = FLAGS.window_size
        self.num_steps = FLAGS.num_steps
        self.learning_rate = FLAGS.learning_rate
        self.num_neg_samples = FLAGS.num_neg_samples
        self.wrd_dim = FLAGS.embedding_size
        self.phr_dim = FLAGS.paragraph_embedding_size



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
        self.word_embeddings = Embedding.embeddings
        self.word_dictionary = Embedding.dictionary

     
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
            while (self.word_index + window_size) > len(paragraph):
                self.para_index = (self.para_index + 1) % len(self.concept_list)
                self.word_index = 0
                paragraph = self.concept_list[self.para_index].fullConcept()
             
            para_examples[i][0] = self.para_index

            for j in range(window_size - 1):
                # print self.word_dictionary[paragraph[self.word_index+j].lower()]
                # print Embedding.wordVec(paragraph[self.word_index+j].lower())

                # word_examples[i][j] = self.word_dictionary[paragraph[self.word_index+j].lower()]
                word_examples[i][j] = Embedding.wordIndex(paragraph[self.word_index+j].lower())
            # labels[i] = self.word_dictionary[paragraph[self.word_index+window_size-1].lower()]
            try:
                labels[i] = Embedding.wordIndex(paragraph[self.word_index+window_size-1].lower())
            except:
                import pdb; pdb.set_trace()
            self.word_index = self.word_index + 1

        return para_examples, word_examples, labels

    # def para_with_word_emb(self, para_id, word_ids):
    #   """sum of para emb and words emb"""

    #   #Paragraph embedding: [1,emb_size]
    #   para_emb = tf.nn.embedding_lookup(self._para_emb,[para_id])
    #   #Word embedding: [window_size, emb_size]
    #   word_emb = tf.nn.embedding_lookup(self._word_emb,words_ids)

    #   #Sum of word_emb: [1, emb_size]
    #   words_emb = tf.reduce_sum(word_emb,0)

    #   #Para + words embedding: [1, emb_size]
    #   return tf.add(para_emb, words_emb)


    def build_graph(self):
        opts = self._options

        # Input data
        self.para_examples = tf.placeholder(tf.int32, shape=[opts.batch_size, 1], name="para_examples")
        self.word_examples = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.window_size-1], name="word_examples")
        self.labels = tf.placeholder(tf.int32, shape=[opts.batch_size, 1], name="labels")

        
        # concaternation
        emb_dim = opts.phr_dim + opts.wrd_dim*(opts.window_size-1)

        #Para Embedding: [para_size, phr_dim]
        para_emb = tf.Variable(
            tf.random_uniform(
                [self._para_size, opts.phr_dim], 
                -0.5 / opts.phr_dim, 0.5 / opts.phr_dim),
                trainable = True,
                name="w_para")
        self._para_emb = para_emb

        #Word Embedding: [vocab_size, emb_dim]
        word_emb = tf.Variable(self.word_embeddings, trainable = False, name="w_word")
        self._word_emb = word_emb

        #Embedding for examples calculation
        para_embed = tf.nn.embedding_lookup(para_emb, self.para_examples) #[[[emb_dim]]*batch_size]
        para_embed = tf.reshape(para_embed, [opts.batch_size, -1])
        # para_embed = tf.reduce_sum(para_embed,1) #[[emb_dim]*batch_size]
        # self.para_embed = para_embed

        words_embed = tf.nn.embedding_lookup(word_emb,self.word_examples) # sum of embeddings of word in examples [[[emb_dim]*(window_size-1)]*batch_size]
        words_embed = tf.reshape(words_embed, [opts.batch_size, -1])
        # words_embed = tf.reduce_sum(words_embed,1) #[[emb_dim]*batch_size]
        # self.word_embed = word_embed
        embed = tf.concat(1, [para_embed, words_embed], name="concat_embed")

        
        # Embeddings for examples: [batch_size, emb_dim]
        # embed = tf.add(para_embed, words_embed) # sum of embedding of words and para [[emb_dim]*batch_size]

        opts.vocab_size = len(self.word_dictionary)

        #Softmax weight: [vocab_size, emb_dim]. Transposed
        # w_out = tf.Variable(tf.zeros([opts.vocab_size, emb_dim]), name="w_out")
        w_out = tf.Variable(tf.truncated_normal([opts.vocab_size, emb_dim], stddev=1.0 / math.sqrt(emb_dim)), name="w_out")   
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
                num_classes=opts.vocab_size), name="loss")

        self.loss = loss
        self.trainer = tf.train.GradientDescentOptimizer(opts.learning_rate).minimize(loss)

    def train(self):
        """Train the model"""
        opts = self._options

        for step in range(opts.num_steps):

            # import pdb; pdb.set_trace()
            para_examples, word_examples, labels = self.generate_batch(opts.batch_size, opts.window_size)


            feed_dict = {self.para_examples:para_examples, self.word_examples:word_examples, self.labels:labels}
            _, loss_val = self._session.run([self.trainer, self.loss], feed_dict=feed_dict)

            if step%100 == 0:
                print ("loss at step ", step,":", loss_val)
    
    def draw(self):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        norm = tf.sqrt(tf.reduce_sum(tf.square(self._para_emb), 1, keep_dims=True))
        normalized_embeddings = self._para_emb / norm
        low_dim_embs = tsne.fit_transform(normalized_embeddings.eval())

        for i, concept in enumerate(self.concept_list):
            concept.setLowEmb(low_dim_embs[i])

        Plot(self.cm).drawWithTag()

        
        # plt.figure(figsize=(18,18))
        # for i in range(len(self.concept_list)):
        #   x,y = low_dim_embs[i,:]
        #   plt.scatter(x,y)
        #   plt.annotate(self.concept_list[i].conceptName(),
        #              xy=(x, y),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom')
        # plt.show()
         

def main():
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = Para2vec(CM(40),opts,session)
        model.train()
        model.draw()    

if __name__ == '__main__':
    main()


        