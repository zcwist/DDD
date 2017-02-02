import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

remodel = False
if remodel:
	text8 = gensim.models.word2vec.Text8Corpus('text8')
	model = gensim.models.Word2Vec(text8,size=200,min_count=10)
	model.save('model')
else:
	model = gensim.models.Word2Vec.load('model')

# print len(model['word'])
print (model.wv.index2word[47133])


