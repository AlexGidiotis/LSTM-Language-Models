import numpy as np
import random
import time
import sys

from keras.layers import Dense, Input, LSTM, Dropout, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import callbacks
from data_gen import Corpus



train_data = 'data/train_set.txt'
val_data = 'data/val_set.txt'


class GenerateText(callbacks.Callback):
	"""
	"""

	def on_epoch_end(self, epoch, logs={}):
		"""
		"""		

		starting_text = 'WikiCorpus language model'

		if epoch % 50 == 0:
			test_generated = ''
			test_generated += starting_text
			sys.stdout.write(test_generated)

			# Generate 1000 characters.
			for i in range(1000):
				# First vectorize the starting text.
				x = np.zeros((1, max_len, data_set.vocab_size))

				for t, char in enumerate(starting_text):
					x[0, t, data_set.char2id[char]] = 1.

				# Predict using the previously generated text as input.
				preds = model.predict(x, verbose=0)[0]
				# Sample and decode the prediction.
				next_char_one_hot = sample(preds,0.6)
				next_char = data_set.id2char[np.argmax(next_char_one_hot)]
				# Append prediction to the generated text.
				test_generated += next_char
				# Write the output one char at a time.
				sys.stdout.write(next_char)
				sys.stdout.flush()

			print()

		return


class Checkpointer(callbacks.Callback):
	"""
	"""

	def on_epoch_end(self, epoch, logs={}):
		"""
		"""		

		if epoch % 10 == 0:
			model.save_weights(STAMP + '.hdf5', True)

		return
 

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


def build_model(max_len,
	vocab_size):
	"""
	"""

	input_layer = Input(shape=(max_len,vocab_size))

	lstm1 = LSTM(100,
		activation='tanh', 
		recurrent_activation='hard_sigmoid', 
		recurrent_dropout=0.0,
		dropout=0.2, 
		return_sequences=True)(input_layer)
	lstm1 = BatchNormalization()(lstm1)

	lstm2 = LSTM(100, 
		activation='tanh', 
		recurrent_activation='hard_sigmoid', 
		recurrent_dropout=0.0,
		dropout=0.4, 
		return_sequences=False)(lstm1)
	lstm2 = BatchNormalization()(lstm2)

	dropout = Dropout(0.4)(lstm2)
	predictions = Dense(vocab_size,
		activation='softmax')(dropout)

	model = Model(inputs=input_layer, outputs=predictions)

	model.compile(loss='categorical_crossentropy',
			optimizer='rmsprop')
	model.summary()

	return model


if __name__ == '__main__':

	epochs = 2000
	max_len = 25
	batch_size = 512
	data_sample = 0.1
	STAMP = 'language_model'

	data_set = Corpus(train_data,val_data,max_len=max_len,batch_size=batch_size,data_sample=data_sample)

	model = build_model(max_len,data_set.vocab_size)

	print(STAMP)

	model_json = model.to_json()
	with open(STAMP + ".json", "w") as json_file:
		json_file.write(model_json)

	generate_text = GenerateText()
	checkpointer = Checkpointer()
	
	hist = model.fit_generator(data_set.get_train(),
		steps_per_epoch=(data_set.train_size),
		epochs=epochs,
		validation_data=data_set.get_val(),
		validation_steps=(data_set.val_size),
		callbacks=[generate_text,checkpointer],
		verbose=1)


		

