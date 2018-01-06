# Mahabharatha Chatbot
ChatBot Trained using Seq2Seq Model on Mahabharatha Subtitles

## Model Information
* RNN Encoder-Decoder Model(Seq2Seq)
* MultiRNNCell GRU Cell
* Attention Decoder and Embedding Layer with Buckets
* Sampled SoftMax Loss and SGD Optimizer
* Beam Search Response Construction

#### Dependencies
* [Python 3.6](https://www.python.org)
* [Tensorflow > 1.1](https://www.tensorflow.org/)
* [nltk](https://pypi.python.org/pypi/nltk)
* [flask](http://flask.pocoo.org/)

## Dataset Preparation

The Data has been prepared from Mahabharatha Subtitles.
All pre-processing and post-processing steps are available in prep_data.py file.

MBT SUBS: https://www.opensubtitles.org/en/ssearch/sublanguageid-all/idmovie-63130 

It can be used to train on any set of subtitles within the same file.

## Instructions
* Download pre trained Model and extract in the checkpoints folder  
* Run with 'python3 mbt_bot.py --m=chat'  
* There are 3 modes, train, chat and api(for rest api using flask)  