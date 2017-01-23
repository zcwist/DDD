import logging

import numpy as np

from gensim import utils, matutils
from gensim.models import doc2vec

class DocsText8(object):
	"""docstring for DocsText8"""
	def __init__(self, string_tags=False):
		super(DocsText8, self).__init__()
		self.string_tags = string_tags
		
	def _tag(self, i):
		return i if not self.string_tags else '_*d' % i

	def __iter__(self):
		with open('text8') as f:
			for i, line in enumerate(f):
				yield doc2vec.TaggedDocument(utils.simple_preprocess(line),[self._tag[i]])

print DocsText8()[1]

