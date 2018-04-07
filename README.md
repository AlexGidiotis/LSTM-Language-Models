# LSTM-Language-Models
Experiments with LSTM networks for language modeling. The language models implemented here can easily be used with any type of corpus.
In my experiments I am using the [Hutter Prize](http://prize.hutter1.net) 100 MB dataset. I have split the dataset into training and validation set (80%/20%). 

The language model implemented here is a two layer LSTM network that outputs the following character for a sequence of characters. The trained model can then be used to generate new text character by character that is similar to the original training corpus.



## Requirements

I highly suggest that you use the **GPU** version of **Tensorflow** because this project requires some heavy computations. In order to
use the GPU version of **Tensorflow** you will first need to install [CUDA](http://docs.nvidia.com/cuda).

- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Keras](https://github.com/fchollet/keras)

Run `pip install -r requirements.txt` to install the requirements.