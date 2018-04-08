import json
import sys

import numpy as np

from keras.models import model_from_json


def load_model(STAMP):

	json_file = open(STAMP+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights(STAMP+'.hdf5')
	print("Loaded model from disk")

	model.summary()

	return model


def sample(prediction,
	temperature):
	"""
	"""

	prediction = np.asarray(prediction).astype('float64')
	prediction = np.log(prediction) / temperature
	exp_prediction = np.exp(prediction)
	prediction = exp_prediction / np.sum(exp_prediction)
	probabilities = np.random.multinomial(1, prediction, 1)
	
	return probabilities


def text_generator(model,
	max_len,
	vocab_size,
	char2id,
	id2char,
	temperature):
	"""
	"""		

	starting_text = 'WikiCorpus language model'

	test_generated = ''
	test_generated += starting_text
	sys.stdout.write(test_generated)

	for i in range(1000):

		x = np.zeros((1, max_len, vocab_size))
		for t, char in enumerate(starting_text):
			x[0, t, char2id[char]] = 1.


		preds = model.predict(x, verbose=0)[0]


		next_char_one_hot = sample(preds,temperature=temperature)
		next_char = id2char[np.argmax(next_char_one_hot)]


		test_generated += next_char


		sys.stdout.write(next_char)
		sys.stdout.flush()

	print()

	return


if __name__ == '__main__':

	max_len = 25
	STAMP = 'language_model'

	with open('char2id.json', 'r') as fp:
		json_string = fp.read()
		char2id = json.loads(json_string)
	kv = char2id.items()
	kv = [(tup[0],int(tup[1])) for tup in kv]
	char2id = dict(kv)

	vocab_size = len(char2id)

	with open('id2char.json', 'r') as fp:
		json_string = fp.read()
		id2char = json.loads(json_string)
	kv = id2char.items()
	kv = [(int(tup[0]),tup[1]) for tup in kv]
	id2char = dict(kv)

	model = load_model(STAMP)


	text_generator(model,max_len,vocab_size,char2id,id2char,temperature=0.9)