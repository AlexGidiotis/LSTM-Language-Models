# -*- coding: utf-8 -*-

import re
from random import shuffle

import numpy as np


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


def sample(prediction,
	temperature):
	"""
	Sample the model predictions with a given temperature.

	Args:
		prediction: The predictions outputed by the model.
		temperature: The temperature used for sampling.
			High temperature(close to 1.) leads to high confidence.
			Low temperature leads to higher diversity. 

	Returns:
		probabilities: The output probabilities sampled.
	"""

	prediction = np.asarray(prediction).astype('float64')
	prediction = np.log(prediction) / temperature
	exp_prediction = np.exp(prediction)
	prediction = exp_prediction / np.sum(exp_prediction)
	probabilities = np.random.multinomial(1, prediction, 1)
	
	return probabilities


if __name__ == '__main__':
	split_data(data_file)
