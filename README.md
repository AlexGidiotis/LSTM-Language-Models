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

## Usage 

### Data 

In order to train your own language models you will need a corpus of data. The pre-processing required is minimal. In my experiments I just split the raw text data into training and validation set. In order to split the Wiki corpus you can use `util.py`. If you want to use your own data you simply need to split the dataset in a `train_set.txt` and `val_set.txt` set and put the data files into a `data/` directory. 


### Training 

Once that data are ready you can start training your own language model. In my experiments I am splitting each set into wiki pages (</page>\n as the delimiter) and each wiki page into batches of 512 characters. Then these batches are fed into the model one at a time for training. 


### Generation

Once the model has converged, you can use it to generate new text samples. The generation requires an initial text feed to start the sampling chain and then the generation can be controlled using the temperature parameter.