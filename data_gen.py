# -*- coding: utf-8 -*-
import time
import re
from random import shuffle
import string

import numpy as np

printable = set(string.printable)

class Corpus(object):
	"""
	"""

	def __init__(self,
		train_file,
		val_file,
		data_sample,
		batch_size=64,
		max_len=25,
		skip=2):
		"""
		"""

		self.train_file = train_file
		self.val_file = val_file
		self.data_sample = data_sample
		self.batch_size= batch_size
		self.max_len = max_len
		self.skip = skip

		self.read_data()


	def read_data(self):
		"""
		"""
		with open(self.train_file) as f:
			self.train_data = f.read()
			self.train_data = filter(lambda x: x in printable, self.train_data)
			self.train_data = re.split('(?<=</page>)\n+',self.train_data)
			shuffle(self.train_data)
			self.train_data = self.train_data[:int(len(self.train_data) * self.data_sample)]
	

		with open(self.val_file) as f:
			self.val_data = f.read()
			self.val_data = filter(lambda x: x in printable, self.val_data)
			self.val_data = re.split('(?<=</page>)\n+',self.val_data)
			shuffle(self.val_data)
			self.val_data = self.val_data[:int(len(self.val_data) * self.data_sample)]

		self.build_vocabulary(self.train_data)
		self.find_size()

		return


	def find_size(self):
		"""
		"""
		num_train_batches = 0
		for item in self.train_data:
			batches = [item[i:i+self.batch_size] for i in range(0, len(item), self.batch_size)]
			num_train_batches += len(batches)

		self.train_size = num_train_batches

		num_val_batches = 0
		for item in self.val_data:
			batches = [item[i:i+self.batch_size] for i in range(0, len(item), self.batch_size)]
			num_val_batches += len(batches)

		self.val_size = num_val_batches

		return


	def build_vocabulary(self,
		data):
		"""
		"""

		char_vocab = []
		for item in data:
			chars = list(set(item))
			for char in chars:
				if char not in char_vocab:
					char_vocab.append(char)


		char_vocab = sorted(char_vocab)
		self.vocab_size = len(char_vocab)

		self.char2id = dict((c,i) for i,c in enumerate(char_vocab))
		self.id2char = dict((i,c) for i,c in enumerate(char_vocab))
		print('Found %s unique tokens' % self.vocab_size)


		return


	def get_train(self):
		"""
		"""

		while True:
			for item in self.train_data:

				batches = [item[i:i+self.batch_size] for i in range(0, len(item), self.batch_size)]
				for batch in batches:
					sections = []
					next_chars = []
					for i in range(0,len(batch)-self.max_len,self.skip):
						sections.append(batch[i: i + self.max_len])
						next_chars.append(batch[i + self.max_len])

					if len(sections) == 0:
						continue

					train_batch = np.zeros((len(sections),self.max_len,self.vocab_size))
					train_targets = np.zeros((len(sections),self.vocab_size))

					for i,section in enumerate(sections):
						for j,char in enumerate(section):
							train_batch[i,j,self.char2id[char]] = 1.

						train_targets[i,self.char2id[next_chars[i]]] = 1.

					yield train_batch, train_targets

			self.shuffle_examples('train')


	def get_val(self):
		"""
		"""

		while True:
			for item in self.val_data:

				batches = [item[i:i+self.batch_size] for i in range(0, len(item), self.batch_size)]

				for batch in batches:
					sections = []
					next_chars = []
					for i in range(0,len(batch)-self.max_len,self.skip):
						sections.append(batch[i: i + self.max_len])
						next_chars.append(batch[i + self.max_len])

					if len(sections) == 0:
						continue

					val_batch = np.zeros((len(sections),self.max_len,self.vocab_size))
					val_targets = np.zeros((len(sections),self.vocab_size))

					for i,section in enumerate(sections):
						for j,char in enumerate(section):
							try:
								val_batch[i,j,self.char2id[char]] = 1.
							except:
								pass
						try:
							val_targets[i,self.char2id[next_chars[i]]] = 1.
						except:
							pass
							
					yield val_batch, val_targets

			self.shuffle_examples('val')


	def shuffle_examples(self,
		flag):
		"""
		"""

		if flag == 'train':
			print 'Shuffling training set'
			shuffle(self.train_data)
		else:
			print 'Shuffling validation set'
			shuffle(self.val_data)

		return




