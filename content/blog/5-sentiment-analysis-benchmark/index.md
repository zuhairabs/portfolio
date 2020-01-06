---
title: Overview and benchmark of traditional and deep learning models in text classification 📝
date: 2018-06-12 10:10:00 # YYYY-MM-DD - H:M:S
author: Ahmed BESBES
tags: ['nlp', 'sentiment analysis', 'cnn', 'rnn', 'gru', 'transfer learning', 'deep learning', 'keras', 'neural networks', 'twitter', 'glove', 'bag of words', 'word ngrams', 'character ngrams']
excerpt: How do deeo learning models based on convoutions (CNNs) and recurrents cells (RNNs) compare to Bag of Words models in the case of a classification problem
slug: benchmarking-sentiment-analysis-models
folder: /blog/5-sentiment-analysis-benchmark
ogimage: images/cover-benchmark.png
---

This article is an extension of a <a href="https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html" style="text-decoration: none">previous one</a> I wrote when I was experimenting sentiment analysis on twitter data. Back in the time, I explored a simple model: a two-layer feed-forward neural network trained on **keras**. The input tweets were represented as document vectors resulting from a weighted average of the embeddings of the words composing the tweet.

The embedding I used was a word2vec model I trained from scratch on the corpus using **gensim**. The task was a binary classification and I was able with this setting to achieve 79% accuracy.

<img src="./images/cover_resized.png">

The goal of this post is to explore other NLP models trained on the same dataset and then benchmark their respective performance on a given test set.

We'll go through different models: from simple ones relying on a bag-of-word representation to a heavy machinery deploying convolutional/recurrent networks: We'll see if we'll score more than 79% accuracy! 

<img src="./images/rnn_unrolled.png" width="100%">

I will start from simple models and add up complexity progressively. The goal is also to show that **simple models work well too**.

So I'm going to try out these:

- Logistic regression with word ngrams
- Logistic regression with  character ngrams
- Logistic regression with word and character ngrams
- Recurrent neural network (bidirectional GRU) without pre-trained embeddings
- Recurrent neural network (bidirectional GRU) with GloVe pre-trained embeddings
- Multi channel Convolutional Neural Network 
- RNN (Bidirectional GRU) + CNN model 

By the end of this post, you will have a boilerplate code for each of these NLP techniques. It'll help you kickstart your NLP project and eventually achieve state-of-the art results (some of these models are really powerful).

We'll also provide a comprehensive benchmark from which we'll tell which model is best suited for predicting the sentiment of a tweet.

In the related git repo, I will publish the different models, their predictions, as well as the test set. You can try them yourself and get confident about the results

Let's get started!


```python
import os
import re

import warnings
warnings.simplefilter("ignore", UserWarning)
from matplotlib import pyplot as plt
%matplotlib inline


import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np 
from string import punctuation

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.externals import joblib

import scipy
from scipy.sparse import hstack
```


<style>
div.prompt {display:none}
</style>


## 0 - Data pre-processing

The dataset can be downloaded from this <a href="http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/">link</a>.

We'll load it and restrict ourselves to the variables we need (Sentiment and SentimentText).

It contains 1578614 classified tweets, each row is marked as 1 for positive sentiment and 0 for negative sentiment.

The author recommend using 1/10 for testing the algorithm and the rest for training.


```python
data = pd.read_csv('./data/tweets.csv', encoding='latin1', usecols=['Sentiment', 'SentimentText'])
data.columns = ['sentiment', 'text']
data = data.sample(frac=1, random_state=42)
print(data.shape)
# (1578614, 2)

for row in data.head(10).iterrows():
    print(row[1]['sentiment'], row[1]['text']) 
```

    1 http://www.popsugar.com/2999655 keep voting for robert pattinson in the popsugar100 as well!! 
    1 @GamrothTaylor I am starting to worry about you, only I have Navy Seal type sleep hours. 
    0 sunburned...no sunbaked!    ow.  it hurts to sit.
    1 Celebrating my 50th birthday by doing exactly the same as I do every other day - working on our websites.  It's just another day.   
    1 Leah and Aiden Gosselin are the cutest kids on the face of the Earth 
    1 @MissHell23 Oh. I didn't even notice.  
    0 WTF is wrong with me?!!! I'm completely miserable. I need to snap out of this 
    0 Was having the best time in the gym until I got to the car and had messages waiting for me... back to the down stage! 
    1 @JENTSYY oh what happened?? 
    0 @catawu Ghod forbid he should feel responsible for anything! 


Tweets are noisy, let's clean them by removing urls, hashtags and user mentions.


```python
def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tokens
```

Once the data is clean, we save it on disk.

```python
data['tokens'] = data.text.progress_map(tokenize)
data['cleaned_text'] = data['tokens'].map(lambda tokens: ' '.join(tokens))
data[['sentiment', 'cleaned_text']].to_csv('./data/cleaned_text.csv')

data = pd.read_csv('./data/cleaned_text.csv')
print(data.shape)
# (1575026, 2)

data.head()
```

<div style="overflow-x: scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe" style="border-collapse: collapse">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>cleaned_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>playing with my routers looks like i might hav...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>sleeeep agh im so tired and they wrote gay on ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>alan ignored me during the concert boo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>really want some mini eggs why are they only a...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>thanks guys sorry i had to miss your show at m...</td>
    </tr>
  </tbody>
</table>
</div>



Now that the dataset is cleaned, let's prepare a train/test split to build our models. 

We'll use this split throughout all the notebook.


```python
x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'], 
                                                    data['sentiment'], 
                                                    test_size=0.1, 
                                                    random_state=42,
                                                    stratify=data['sentiment'])

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (1417523,) (157503,) (1417523,) (157503,)
```

I'm saving the test labels on disk for later use.


```python
pd.DataFrame(y_test).to_csv('./predictions/y_true.csv', index=False, encoding='utf-8')
```

Let's start applying some machine learning now:

## 1 - Bag of word model based on word ngrams

So. What's an **n-gram** ?

 ![](./images/ngrams.png)

As we see it in this figure, n-grams are simply all combinations of adjacent words (in this case) of length n that you can find in your source text. 

In our model, we are going to use unigrams (n=1) and bigrams (n=2) as features. 

The dataset will therefore be represented as a matrix where each row refers to a tweet and each column refers to a feature (unigram or bigram) extracted from the text (after tokenization and cleaning). Each cell will be the **tf-idf** score. (we could also use a simple count but tf-idf is more used in general and usually works better). We call this matrix the **document-term matrix**. 

As you can imagine, the number of unique unigrams and bigrams of 1.5 million-tweet corpus is huge. In practice, we will set this number to a fixed value, for a computational reason. You can determine this value by using a cross-validation.

This is what the corpus should look like after vectorization.

![](./images/tfidf.jpg)

> I like pizza a lot

    
Let's say we want to feed this sentence to a predictive model using the features described above.

Given that we're using unigrams and bigrams, the model will extract the following features:

> i, like, pizza, a, lot, i like, like pizza, pizza a, a lot


Therefore, the sentence will be formed by a vector of size N (= total number of tokens) containing lots of zeros and the tf-idf scores of these ngrams.
So you can clearly see that we will be dealing with large and **sparse** vectors.

When dealing with large and sparse data, linear models generally perform quite well. Besides, they are faster to train than other types of models (e.g. tree-based models).

I can tell from past experience that **logistic regression** works well on top of sparse tfidf matrices.


```python
vectorizer_word = TfidfVectorizer(max_features=40000,
                             min_df=5, 
                             max_df=0.5, 
                             analyzer='word', 
                             stop_words='english', 
                             ngram_range=(1, 2))

vectorizer_word.fit(x_train, leave=False)

tfidf_matrix_word_train = vectorizer_word.transform(x_train)
tfidf_matrix_word_test = vectorizer_word.transform(x_test)
```

After generating  tfidf matrices for both train and test sets, we can build our first model an test it.

The tifidf matrices are the features of the logistic regression.


```python
lr_word = LogisticRegression(solver='sag', verbose=2)
lr_word.fit(tfidf_matrix_word_train, y_train)
```

Once the model is trained, we apply it on the test data to get predictions. Then we save these values as well as the model on disk.


```python
joblib.dump(lr_word, './models/lr_word_ngram.pkl')

y_pred_word = lr_word.predict(tfidf_matrix_word_test)
pd.DataFrame(y_pred_word, columns=['y_pred']).to_csv('./predictions/lr_word_ngram.csv', index=False)
```

Let's see what accuracy score we have:


```python
y_pred_word = pd.read_csv('./predictions/lr_word_ngram.csv')
print(accuracy_score(y_test, y_pred_word))
```

> 0.782042246814


**78.2% accuracy** for a first model ! Quite not bad. Let's move to the next model.

# 2 - Bag of word model based on character ngrams

We never said that ngrams were for words only. We can apply them at a character level as well.

<img src='./images/ngrams_char.jpg' width="50%">
 

You see it coming, right? We're going to apply the same code above to character ngrams instead, and we will go up to 4-grams.

This basically means that a sentence like "I like this movie" will have these features:

> I, l, i, k, e, ..., I li,  lik, like, ..., this, ... , is m, s mo, movi, ...

Character ngrams are surprisingly very effective. They can even outperform word tokens in modeling a language task. For example <a href="http://www.icsd.aegean.gr/lecturers/stamatatos/papers/ijait-spam.pdf">spam filters</a> or <a href="http://www.martijnwieling.nl/files/groningen_power.pdf">Native Language Identification</a> heavily rely on character ngrams.

Unlike the previous model which learns combinations of words, this model learns combinations of letters, which can handle the morphological makeup of a word.

One of the advantages of the character-based representation is the better handling of **misspelled words**.

Let's run the same pipeline :


```python
vectorizer_char = TfidfVectorizer(max_features=40000,
                             min_df=5, 
                             max_df=0.5, 
                             analyzer='char', 
                             ngram_range=(1, 4))

vectorizer_char.fit(tqdm_notebook(x_train, leave=False));

tfidf_matrix_char_train = vectorizer_char.transform(x_train)
tfidf_matrix_char_test = vectorizer_char.transform(x_test)

lr_char = LogisticRegression(solver='sag', verbose=2)
lr_char.fit(tfidf_matrix_char_train, y_train)

y_pred_char = lr_char.predict(tfidf_matrix_char_test)
joblib.dump(lr_char, './models/lr_char_ngram.pkl')

pd.DataFrame(y_pred_char, columns=['y_pred']).to_csv('./predictions/lr_char_ngram.csv', index=False)

y_pred_char = pd.read_csv('./predictions/lr_char_ngram.csv')
print(accuracy_score(y_test, y_pred_char))
```

> 0.80420055491


**80.4% accuracy !** Character-ngrams perfom better that word-ngrams.

## 3 - Bag of word model based on word and character ngrams

Character ngram features seem to provide a better accuracy than word ngrams. But what about the combination of the two: word + character ngrams?

Let's concatenate the two tfidf matrices we generated and build a new, *hybrid* tfidf matrix.

This model will help us learn the indentity of a word and its possible neighbors as well as its morphological structure.

These properties will be combined.


```python
tfidf_matrix_word_char_train =  hstack((tfidf_matrix_word_train, tfidf_matrix_char_train))
tfidf_matrix_word_char_test =  hstack((tfidf_matrix_word_test, tfidf_matrix_char_test))

lr_word_char = LogisticRegression(solver='sag', verbose=2)
lr_word_char.fit(tfidf_matrix_word_char_train, y_train)

y_pred_word_char = lr_word_char.predict(tfidf_matrix_word_char_test)
joblib.dump(lr_word_char, './models/lr_word_char_ngram.pkl')

pd.DataFrame(y_pred_word_char, columns=['y_pred']).to_csv('./predictions/lr_word_char_ngram.csv', index=False)

y_pred_word_char = pd.read_csv('./predictions/lr_word_char_ngram.csv')
print(accuracy_score(y_test, y_pred_word_char))
```

> 0.81423845895


Awesome: **81.4% accuracy**. We just increased by one whole unit and outperformed the two previous settings.

### What can we say about bag-of-word models before we move on?


- **Pros**: They can be surprisingly powerful given their simplicity, they are fast to train, and easy to understand and interpret.  
- **Cons**: Even though ngrams bring some context between words, bag of word models fail in modeling long-term dependencies between words in a sequence. They increase the dimensionality of the problem.


Now we're going to dive into deep learning models. The reason deep learning outperform bag of word models is the ability to capture the sequencial dependency between words in a sentence. This has been possible thanks to the invention of special neural network architectures called **Recurrent Neural Networks**.

I won't cover the theoretical foundations of RNNs, but here's a <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">link</a> I find worth reading. It's from Cristopher Olah's blog. It details LSTM: Long Short Term Memory. A special kind of RNN. 

Before starting, we have to setup a deep learning dedicated environment that uses Keras on top of Tensorflow. I honestly tried to run everything on my personal laptop but given the important size of the dataset and the complexity of RNN architectures, this has not been practical. At all.

One good option is AWS. I generally use this <a href="https://aws.amazon.com/marketplace/pp/B077GCH38C?qid=1527197041958&sr=0-1&ref_=srh_res_product_title">deep learning AMI</a> on an EC2 <a href="https://aws.amazon.com/ec2/instance-types/p2/">p2.xlarge</a> instance. Amazon AMI are pre-configured VM images, where all the packages (Tensorflow, PyTocrh, Keras, etc. ) are installed. I highly recommend this one which I have been using for a while.


```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import Sequential

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model
```

## 4 - Recurrent Neural Network without pre-trained embedding

RNNs may look scary. Although they're complex to understand, they're quite interesting. They encapsulate a very beautiful design that overcomes traditional neural networks' shortcomings that rise when dealing with sequence data: text, time series, videos, DNA sequences, etc.

An RNN is a sequence of neural network blocks that are linked to each others like a chain. Each one is passing a message to a successor.

Again if you want to dive into the internal mechanics, I highly recommend Colah's <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">blog</a> out of which the diagram below is taken.

<img src="./images/rnn_unrolled.png" width="100%">

We will process text data, which is a sequence type. The order of words is very important to the meaning. Hopefully RNNs take care of this and can capture long-term dependencies.

To use Keras on text data, we firt have to preprocess it. For this, we can use Keras' Tokenizer class. This object takes as argument **num_words** which is the maximum number of words kept after tokenization based on their word frequency.


```python
MAX_NB_WORDS = 80000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(data['cleaned_text'])
```

Once the tokenizer is fitted on the data, we can use it to convert text strings to sequences of numbers.

These numbers represent the position of each word in the dictionary (think of it as mapping). 

Let's see an example:


```python
x_train[15]

# 'breakfast time happy time'
```

Here's how the tokenizer turns it into a sequence of digits. 


```python
tokenizer.texts_to_sequences([x_train[15]])

# [[530, 50, 119, 50]]
```

Let's now apply this tokenizer on the train and test sequences:


```python
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)
```

Now the tweets are mapped to lists of integers. However, we still cannot stack them together in a matrix since they have different lengths.
Hopefully Keras allows to **pad** sequences with **0s** to a maximum length. We'll set this length to 35. (which is the maximum numbers of tokens in the tweets).


```python
MAX_LENGTH = 35
padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)
padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)
```


```python
padded_train_sequences
#  array([[    0,     0,     0, ...,  2383,   284,     9],
#         [    0,     0,     0, ...,    13,    30,    76],
#         [    0,     0,     0, ...,    19,    37, 45231],
#         ..., 
#         [    0,     0,     0, ...,    43,   502,  1653],
#         [    0,     0,     0, ...,     5,  1045,   890],
#         [    0,     0,     0, ..., 13748, 38750,   154]])
    
print(padded_train_sequences.shape)
# (1417523, 35)
```




    



Now the data is ready to be fed to an RNN.

Here are some elements of the architecture I'll be using:

- An embedding dimension of 300. This means that each word from the 80000 that we'll be using is mapped to a 300-dimension dense vector (of float numbers). The mapping will adjust throughout the training.

- A spatial dropout is applied on the embedding layer to reduce overfitting: it basically looks at batches of 35x300 matrices and randomly drop (set to 0) word vectors (i.e rows) in each matrix. This helps not to focus on specific words in an attempt to generalize well.

- A bidirectional Gated Recurrent Unit (GRU): this is the recurrent network part. It's a faster variant of the LSTM architecture. Think of it as a combination of two recurrent networks that scan the text sequence in both directions: from left to right and from right to left. This allows the network, when reading a given word, to understand it by using the context from both past and future information. The GRU takes as parameter a number of units which is the dimension of the output h_t of each network block. We will set this number to 100. And since we are using a bidirectional version of the GRU, the final output per RNN block will be of dimension 200.
 
The output of the bidirectional GRU has the dimension (batch_size, timesteps, units). This means that if we use a typical batch size of 256, this dimension will be (256, 35, 200)

- On top of every batch, we apply a global average pooling that consists in averaging the output vectors corresponding to the each time step (i.e the words)

- We apply the same operation with max pooling. 

- We concatenate the outputs of the two previous operations.


```python
def get_simple_rnn_model():
    embedding_dim = 300
    embedding_matrix = np.random.random((MAX_NB_WORDS, embedding_dim))
    
    inp = Input(shape=(MAX_LENGTH, ))
    x = Embedding(input_dim=MAX_NB_WORDS, output_dim=embedding_dim, input_length=MAX_LENGTH, 
                  weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

rnn_simple_model = get_simple_rnn_model()
```

Let's look at the different layers of this model:


```python
plot_model(rnn_simple_model, 
           to_file='./images/rnn_simple_model.png', 
           show_shapes=True, 
           show_layer_names=True)
```

<img src="./images/rnn_simple_model.png" width="100%">

During the training, model checkpoint is used. It allows to automatically save (on disk) the best models (w.r.t accuracy measure) at the end of each epoch.


```python
filepath="./models/rnn_no_embeddings/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 2

history = rnn_simple_model.fit(x=padded_train_sequences, 
                    y=y_train, 
                    validation_data=(padded_test_sequences, y_test), 
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)

best_rnn_simple_model = load_model('./models/rnn_no_embeddings/weights-improvement-01-0.8262.hdf5')

y_pred_rnn_simple = best_rnn_simple_model.predict(padded_test_sequences, verbose=1, batch_size=2048)

y_pred_rnn_simple = pd.DataFrame(y_pred_rnn_simple, columns=['prediction'])
y_pred_rnn_simple['prediction'] = y_pred_rnn_simple['prediction'].map(lambda p: 1 if p >= 0.5 else 0)
y_pred_rnn_simple.to_csv('./predictions/y_pred_rnn_simple.csv', index=False)


y_pred_rnn_simple = pd.read_csv('./predictions/y_pred_rnn_simple.csv')
print(accuracy_score(y_test, y_pred_rnn_simple))
```

> 0.826219183127


**82.6% accuracy !** Quite not bad ! We are now performing better than the previous bag-of-word models because we are taking into account the sequence nature of the text. 

Can we do better?

## 5 - Recurrent Neural Network with GloVe pre-trained embeddings

In the last model, the embedding matrix was initialized randomly. What if we could use pre-trained word embeddings to intialize it instead?

Let's take an example: imagine that you have the word *pizza* in your corpus. Following the previous architecture, you would initialize it to a 300 dimension vector of random float values. This is perfectly fine. You can do that, and this embedding will adjust an evolve throughout training. However, what you could do instead of randomly picking a vector for pizza is using an embedding for this word that has been learnt from another model on a very large corpus. This is a special kind of **transfer learning**.

Using the knowledge from an external embedding can enhance the precision of your RNN because it integrates new information (lexical and semantic) about the words, an information that has been trained and distilled on a very large corpus of data. 

The pre-trained embedding we'll be using is <a href="https://nlp.stanford.edu/projects/glove/">GloVe</a>.

Official documentation: *GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.* 

The GloVe embeddings I'll be using are trained on a **very large** common internet crawl that includes:

- 840 Billion tokens, 
- 2.2 million size vocab

The zipped file is 2.03 GB download. Beware, this file cannot be easily loaded on a standard laptop.

The dimension of GloVe embeddings is 300. 

GloVe embeddings come in raw text data, where each line contains a word and 300 floats (the corresponding embedding). So the first thing to do is convert this structure to a python dictionary.


```python
def get_coefs(word, *arr):
    try:
        return word, np.asarray(arr, dtype='float32')
    except:
        return None, None
    
embeddings_index = dict(get_coefs(*o.strip().split()) for o in tqdm_notebook(open('./embeddings/glove.840B.300d.txt')))

embed_size=300
for k in tqdm_notebook(list(embeddings_index.keys())):
    v = embeddings_index[k]
    try:
        if v.shape != (embed_size, ):
            embeddings_index.pop(k)
    except:
        pass
            
embeddings_index.pop(None)
```

Once the embedding index in created, we extract all the vectors, we stack them together and compute their mean and standard deviation. 


```python
values = list(embeddings_index.values())
all_embs = np.stack(values)

emb_mean, emb_std = all_embs.mean(), all_embs.std()
```

Now we generate the embedding matrix. We will initialize it following a normal distribution of mean=emb_mean and std=emb_std.

Then we go through the 80000 words of our corpus. For each word, if it is contained in GloVe, we pick its embedding. 

Otherwise, we pass.


```python
word_index = tokenizer.word_index
nb_words = MAX_NB_WORDS
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

oov = 0
for word, i in tqdm_notebook(word_index.items()):
    if i >= MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        oov += 1

print(oov)

def get_rnn_model_with_glove_embeddings():
    embedding_dim = 300
    inp = Input(shape=(MAX_LENGTH, ))
    x = Embedding(MAX_NB_WORDS, embedding_dim, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

rnn_model_with_embeddings = get_rnn_model_with_glove_embeddings()

filepath="./models/rnn_with_embeddings/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 4

history = rnn_model_with_embeddings.fit(x=padded_train_sequences, 
                    y=y_train, 
                    validation_data=(padded_test_sequences, y_test), 
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)

best_rnn_model_with_glove_embeddings = load_model('./models/rnn_with_embeddings/weights-improvement-03-0.8372.hdf5')

y_pred_rnn_with_glove_embeddings = best_rnn_model_with_glove_embeddings.predict(
    padded_test_sequences, verbose=1, batch_size=2048)

y_pred_rnn_with_glove_embeddings = pd.DataFrame(y_pred_rnn_with_glove_embeddings, columns=['prediction'])
y_pred_rnn_with_glove_embeddings['prediction'] = y_pred_rnn_with_glove_embeddings['prediction'].map(lambda p: 
                                                                                                    1 if p >= 0.5 else 0)
y_pred_rnn_with_glove_embeddings.to_csv('./predictions/y_pred_rnn_with_glove_embeddings.csv', index=False)

y_pred_rnn_with_glove_embeddings = pd.read_csv('./predictions/y_pred_rnn_with_glove_embeddings.csv')
print(accuracy_score(y_test, y_pred_rnn_with_glove_embeddings))
```

>  0.837203100893


**83.7% accuracy !** Transfer learning from external word embeddings works! For the rest of the tutorial, I'll be using GloVe embeddings in the embedding matrix.

## 6 - Multi-channel Convolutional Neural Network

In this section, I'm experimenting a convolutional neural network architecture I read about <a href="http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/">here</a>. CNNs are generally used in computer vision. However, we've recently started applying them to NLP tasks and the results were promising. 

Let's briefly see what happens when we use convnets on text data. To explain this, I'm borrowing this famous diagram (below) from wildm.com (a very good blog!)

Let's consider the example it used: I like this movie very much ! (7 tokens)

- The embedding dimension of each word is 5. Therefore this sentence is represented by a matrix of dimension (7,5). You can think of it as an "image" (~ a matrix of digits/floats). 
- 6 Filters, 2 of size (2, 5) (3, 5) and (4, 5) are applied on this matrix. The particularity of these filters is that they are not square matrices and their width is equal to the embedding matrix's width. So the result of each convolution will be a column vector.

- Each column vector resulting from the convolution is subsampled using a maxpooling operation.

- The results of the maxpooling operations are concatenated in a final vector that is passed to a softmax function for classification.


> What is the intuition behind ?

The result of each convolution will fire when a special pattern is detected. By varying the size of the kernels and concatenating their outputs, you're allowing yourself to detect patterns of multiples sizes (2, 3, or 5 adjacent words).

Patterns could be expressions (word ngrams?) like "I hate", "very good" and therefore CNNs can identify them in the sentence regardless of their position.

<img src="./images/cnn_text.png" width="75%">


```python
def get_cnn_model():
    embedding_dim = 300
    
    filter_sizes = [2, 3, 5]
    num_filters = 256
    drop = 0.3

    inputs = Input(shape=(MAX_LENGTH,), dtype='int32')
    embedding = Embedding(input_dim=MAX_NB_WORDS,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=MAX_LENGTH,
                                trainable=True)(inputs)

    reshape = Reshape((MAX_LENGTH, embedding_dim, 1))(embedding)
    conv_0 = Conv2D(num_filters, 
                    kernel_size=(filter_sizes[0], embedding_dim), 
                    padding='valid', kernel_initializer='normal', 
                    activation='relu')(reshape)

    conv_1 = Conv2D(num_filters, 
                    kernel_size=(filter_sizes[1], embedding_dim), 
                    padding='valid', kernel_initializer='normal', 
                    activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, 
                    kernel_size=(filter_sizes[2], embedding_dim), 
                    padding='valid', kernel_initializer='normal', 
                    activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[0] + 1, 1), 
                          strides=(1,1), padding='valid')(conv_0)

    maxpool_1 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[1] + 1, 1), 
                          strides=(1,1), padding='valid')(conv_1)

    maxpool_2 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[2] + 1, 1), 
                          strides=(1,1), padding='valid')(conv_2)
    concatenated_tensor = Concatenate(axis=1)(
        [maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=1, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

cnn_model_multi_channel = get_cnn_model()

plot_model(cnn_model_multi_channel, 
           to_file='./images/cnn_model_multi_channel.png', 
           show_shapes=True, 
           show_layer_names=True)
```

<img src="./images/cnn_model_multi_channel.png" width="100%">


```python
filepath="./models/cnn_multi_channel/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 4

history = cnn_model_multi_channel.fit(x=padded_train_sequences, 
                    y=y_train, 
                    validation_data=(padded_test_sequences, y_test), 
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)

best_cnn_model = load_model('./models/cnn_multi_channel/weights-improvement-04-0.8264.hdf5')

y_pred_cnn_multi_channel = best_cnn_model.predict(padded_test_sequences, verbose=1, batch_size=2048)

y_pred_cnn_multi_channel = pd.DataFrame(y_pred_cnn_multi_channel, columns=['prediction'])
y_pred_cnn_multi_channel['prediction'] = y_pred_cnn_multi_channel['prediction'].map(lambda p: 1 if p >= 0.5 else 0)
y_pred_cnn_multi_channel.to_csv('./predictions/y_pred_cnn_multi_channel.csv', index=False)
```


```python
y_pred_cnn_multi_channel = pd.read_csv('./predictions/y_pred_cnn_multi_channel.csv')
print(accuracy_score(y_test, y_pred_cnn_multi_channel))
```

> 0.826409655689


**82.6% accuracy**, we're less precise than RNNs but still better than BOW models. Maybe an investigation of the hyperparameters (number of filters, and size) gives an edge?

## 7 - Recurrent + Convolutional neural network

RNNs are powerful. Howerer, some people found out that they could maek them more robust by adding a convolutional layer on top of the reccurrent layer.

The rational behind is that RNNs allow you to embed the information about the sequence and previous words and CNN takes this embedding and extract local features from it. Having these two layers working together is a winning combination.

More about this <a href="http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/">here</a>.


```python
def get_rnn_cnn_model():
    embedding_dim = 300
    inp = Input(shape=(MAX_LENGTH, ))
    x = Embedding(MAX_NB_WORDS, embedding_dim, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

rnn_cnn_model = get_rnn_cnn_model()

plot_model(rnn_cnn_model, to_file='./images/rnn_cnn_model.png', show_shapes=True, show_layer_names=True)
```

<img src="./images/rnn_cnn_model.png" width="100%">


```python
filepath="./models/rnn_cnn/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 4

history = rnn_cnn_model.fit(x=padded_train_sequences, 
                    y=y_train, 
                    validation_data=(padded_test_sequences, y_test), 
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)

best_rnn_cnn_model = load_model('./models/rnn_cnn/weights-improvement-03-0.8379.hdf5')

y_pred_rnn_cnn = best_rnn_cnn_model.predict(padded_test_sequences, verbose=1, batch_size=2048)

y_pred_rnn_cnn = pd.DataFrame(y_pred_rnn_cnn, columns=['prediction'])
y_pred_rnn_cnn['prediction'] = y_pred_rnn_cnn['prediction'].map(lambda p: 1 if p >= 0.5 else 0)
y_pred_rnn_cnn.to_csv('./predictions/y_pred_rnn_cnn.csv', index=False)

y_pred_rnn_cnn = pd.read_csv('./predictions/y_pred_rnn_cnn.csv')
print(accuracy_score(y_test, y_pred_rnn_cnn))
```

    0.837882453033


**83.8% accuracy .** Best model so far.

## 8 -  Summary

We've run seven different models. Let's see how they compare:


```python
import seaborn as sns
from sklearn.metrics import roc_auc_score
sns.set_style("whitegrid")
sns.set_palette("pastel")

predictions_files = os.listdir('./predictions/')

predictions_dfs = []
for f in predictions_files:
    aux = pd.read_csv('./predictions/{0}'.format(f))
    aux.columns = [f.strip('.csv')]
    predictions_dfs.append(aux)

predictions = pd.concat(predictions_dfs, axis=1)

scores = {}

for column in tqdm_notebook(predictions.columns, leave=False):
    if column != 'y_true':
        s = accuracy_score(predictions['y_true'].values, predictions[column].values)
        scores[column] = s

scores = pd.DataFrame([scores], index=['accuracy'])

mapping_name = dict(zip(list(scores.columns), 
                        ['Char ngram + LR', '(Word + Char ngram) + LR', 
                           'Word ngram + LR', 'CNN (multi channel)',
                           'RNN + CNN', 'RNN no embd.', 'RNN + GloVe embds.']))

scores = scores.rename(columns=mapping_name)
scores = scores[['Word ngram + LR', 'Char ngram + LR', '(Word + Char ngram) + LR',
                'RNN no embd.', 'RNN + GloVe embds.', 'CNN (multi channel)',
                'RNN + CNN']]

scores = scores.T

ax = scores['accuracy'].plot(kind='bar', 
                             figsize=(16, 5), 
                             ylim=(scores.accuracy.min()*0.97, scores.accuracy.max() * 1.01), 
                             color='red', 
                             alpha=0.75, 
                             rot=45, 
                             fontsize=13)
ax.set_title('Comparative accuracy of the different models')

for i in ax.patches:
    ax.annotate(str(round(i.get_height(), 3)), 
                (i.get_x() + 0.1, i.get_height() * 1.002), color='dimgrey', fontsize=14)
```

 ![](./images/benchmark.png)

Let's quickly check the correlations between the predictions of the models.


```python
fig = plt.figure(figsize=(10, 5))
sns.heatmap(predictions.drop('y_true', axis=1).corr(method='kendall'), cmap="Blues", annot=True);
```

 ![](./images/heatmap.png)

## Conclusions

Here are rapid findings I think are worth sharing:

- Bag of word models using character ngrams can be very efficient. **Do not underestimate them!**. They are relatively cheap to compute, and also easy to interpret.

- RNNs are powerful. However, you can sometimes pump them with external pre-trained embeddings like GloVe. You can also use other popular embeddings such as word2vec and FastText.

- CNNs can be applied to text. They have the main advantage of being very fast to train. Besides, their ability to extract local features out of text is particularly interesting to nlp tasks.

- RNNs and CNNs can be stacked together to take advantags of both architectures.

This post was quite long, I hope you've enjoyed it. Don't hesitate to comment if you have any question or recommendation.

## Where to go from here?

Here are great resources I used when writing this post:

- http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- http://wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
