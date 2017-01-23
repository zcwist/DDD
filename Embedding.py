import gensim

model = gensim.models.Word2Vec.load('GensimModel/model')

def wordVec(word):
	try:
		return model[word]
	except Exception as e:
		raise e

if __name__ == '__main__':
	print (wordVec('word'))
	print (wordVec('letter')) 

		