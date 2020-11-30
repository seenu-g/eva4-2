
[Examples of how to to do sentiment analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)
* [1 - Simple Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb).
nlp1.ipynb: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop.
* [2 - Updated Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb).
nlp2.ipynb: 1 + using packed padded sequences, loading and using pre-trained word embeddings, different optimizers, different RNN architectures, bi-directional RNNs
* [3 - Faster Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb).
nlp3.ipynb: 2 ~ different approach that does not use RNNs
* [4 - Convolutional Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb).
nlp4.ipynb: convolutional neural networks (CNNs) for sentiment analysis
 [5- Multi-class Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/5%20-%20Multi-class%20Sentiment%20Analysis.ipynb)
 nlp5.ipynb over the case where we have more than 2 classes, as is common in NLP
 [6- Transformers for Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb)
nlp6.ipynb using the BERT (Bidirectional Encoder Representations from Transformers) model. we are going to use the transformers library to get pre-trained transformers and use them as our embedding layers. We will freeze (not train) the transformer and only train the remainder of the model which learns from the representations produced by the transformer.

pip install torchtext
python -m spacy download en
pip install transformers


References
[1.BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
[2.Attention Is All You Need](https://arxiv.org/abs/1706.03762)