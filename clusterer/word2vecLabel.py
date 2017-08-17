from label import Label
class Word2VecLabel(Label):
	"""docstring for Word2VecLabel"""
	def __init__(self, cluster):
		super(Word2VecLabel, self).__init__(cluster)

	def label_cluster(self):
		pass

	def label_a_cluster(self,a_cluster):

		contri_word_freq = dict()

		def freq_counter(word):
			if word not in contri_word_freq.keys():
				contri_word_freq[word] = 0
			contri_word_freq[word] += 1


