import pickle

with open('LearntModel/final_embeddings','rb') as f:
	embeddings = pickle.load(f)

with open('LearntModel/dictionary','rb') as f:
	dictionary = pickle.load(f)

with open('LearntModel/reverse_dictionary','rb') as f:
	reverse_dictionary = pickle.load(f)

def wordVec(word):
	try:
		return embeddings[dictionary[word]]
	except Exception as e:
		return embeddings[dictionary["UNK"]]

def wordIndex(word):
	try:
		return dictionary[word]
	except Exception as e:
		return dictionary["UNK"]
		

if __name__ == '__main__':
	# print (wordVec('wordd'))
	print (dictionary)

		