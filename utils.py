# -*- coding: utf-8 -*-

import re
from random import shuffle


data_file = 'data/enwik8'


def split_data(data_file,
	val_size=0.2):
	"""
	"""

	with open(data_file) as f:
		data = f.read()
		data = re.split('(?<=</page>)\n+',data)

	shuffle(data)

	val_data = data[:int(val_size*len(data))]
	train_data = data[int(val_size*len(data)):]

	with open('data/train_set.txt', 'w') as of:
		for d in train_data:
			of.write(d)
			of.write('\n')

	with open('data/val_set.txt', 'w') as of:
		for d in val_data:
			of.write(d)
			of.write('\n')

	return 


if __name__ == '__main__':
	split_data(data_file)
