import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

model = gensim.models.Word2Vec.load('GensimModel/model')
print model['car']