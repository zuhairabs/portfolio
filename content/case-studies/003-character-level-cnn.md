---
title: Character Level CNN
excerpt: You'll find here a PyTorch implementation of a character level CNN for text classification by Zhang and Lecun (2015) and a video tutorial (by me) accompanying it.
iframe: //github.com/zuhairabs/portfolio/blob/master/content/case-studies/images/character-level-cnn/character_cnn.png
demo: //
src: //github.com/zuhairabs/
badgeUrl: "https://ghbtns.com/github-btn.html?user=zuhairabs&repo=&type=star&count=true" 

info:
  idea: This is a PyTorch implementation of a character-level convolutional neural network for text classification.
  tech: [PyTorch]
  links:
    - [Original paper, https://arxiv.org/pdf/1509.01626.pdf]
---

Character Level Convolutional Neural Network

I've been for a quite long time been interested in character level NLP models for text classification and this model really caught my attention.

<img src="./images/character-level-cnn/character_cnn.png">

So I decided to implement it for these various reasons:

- Based on the paper, it's powerful in text classification (see benchmark) even though it doesn't have any notion of semantics
- You don't need to apply any text preprocessing (tokenization, lemmatization, stemming ...) while using it
- Its handles misspelled words and out-of-vocabulary tokes, **by desing**
- It's fast to train
- It doesn't require storing a large word embedding matrix. Hence, it's lightweight and you can deploy it in production easily

When I shared my implementation on Twitter, it quickly went viral: 


<blockquote class="twitter-tweet tw-align-center"><p lang="en" dir="ltr">My <a href="https://twitter.com/PyTorch?ref_src=twsrc%5Etfw">@PyTorch</a> implementation of Character Based ConvNets for text classification published by <a href="https://twitter.com/ylecun?ref_src=twsrc%5Etfw">@ylecun</a> in 2015 is now open-source on <a href="https://twitter.com/github?ref_src=twsrc%5Etfw">@github</a> . Many training features and hacks are implemented. Feel free to check and contribute! <a href="https://t.co/XBtaFQIUhy">https://t.co/XBtaFQIUhy</a><a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/NLP?src=hash&amp;ref_src=twsrc%5Etfw">#NLP</a> <a href="https://t.co/GM8NzZ7GOg">pic.twitter.com/GM8NzZ7GOg</a></p>&mdash; Ahmed Besbes (@ahmed_besbes_) <a href="https://twitter.com/ahmed_besbes_/status/1090903275010998272?ref_src=twsrc%5Etfw">January 31, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

## Dependencies

- numpy
- pandas
- sklearn
- PyTorch 0.4.1
- tensorboardX
- Tensorflow (to be able to run TensorboardX)

## Structure of the code

At the root of the project, you will have:

- train.py: used for training a model
- predict.py: used for the testing and inference
- config.json: a configuration file for storing model parameters (number of filters, neurons)
- src: a folder that contains:
    * cnn_model.py: the actual CNN model (model initialization and forward method)
    * data_loader.py: the script responsible of passing the data to the training after processing it
    * utils.py: a set of utility functions for text preprocessing (url/hashtag/user_mention removal)

## How to use the code

### Training

Launch train.py with the following `arguments`:

* `data_path`: path of the data. Data should be in csv format with at least a column for text and a column for the label
- `validation_split`: the ratio of validation data. default to 0.2
- `label_column`: column name of the labels
- `text_column`: column name of the texts 
- `max_rows`: the maximum number of rows to load from the dataset. (I mainly use this for testing to go faster)
- `chunksize`: size of the chunks when loading the data using pandas. default to 500000
- `encoding`: default to utf-8
- `steps`: text preprocessing steps to include on the text like hashtag or url removal
- `group_labels`: whether or not to group labels. Default to None.
- `use_sampler`: whether or not to use a weighted sampler to overcome class imbalance
- `alphabet`: default to "abcdefghijklmnopqrstuvwxyz
0123456789,;.!?:'\"/\\|_@#$%^&*~\`+-=<>()[]{}" (normally you should not modify it)
- `number_of_characters`: default 70
- `extra_characters`: additional characters that you'd add to the alphabet. For example uppercase letters or accented characters
- `max_length`: the maximum length to fix for all the documents. default to 150 but should be adapted to your data
- `epochs`: number of epochs 
- `batch_size`: batch size, default to 128.
- `optimizer`: adam or sgd, default to sgd
- `learning_rate`: default to 0.01
- `class_weights`: whether or not to use class weights in the cross entropy loss
- `focal_loss`: whether or not to use the focal loss
- `gamma`: gamma parameter of the focal loss. default to 2 
- `alpha`: alpha parameter of the focal loss. default to 0.25
- `schedule`: number of epochs by which the learning rate decreases by half (learning rate scheduling works only for sgd), default to 3. set it to 0 to disable it
- `patience`: maximum number of epochs to wait without improvement of the validation loss, default to 3
- `early_stopping`: to choose whether or not to early stop the training. default to 0. set to 1 to enable it.
- `checkpoint`: to choose to save the model on disk or not. default to 1, set to 0 to disable model checkpoint
- `workers`: number of workers in PyTorch DataLoader, default to 1
- `log_path`: path of tensorboard log file
- `output`: path of the folder where models are saved
- `model_name`: prefix name of saved models

Here's an example:

```shell
python train.py --data_path=/data/tweets.csv --max_rows=200000
```

### Prediction

Launch predict.py with the following arguments:

- `model`: path of the pre-trained model
- `text`: input text
- `steps`: list of preprocessing steps, default to lower
- `alphabet`: default to "abcdefghijklmnopqrstuvwxyz
0123456789-,;.!?:\'"\\/|_@#$%^&*~`+-=<>()[]{}\n"
- `number_of_characters`: default to 70
- `extra_characters`: additional characters that you'd add to the alphabet. For example uppercase letters or accented characters
- `max_length`: the maximum length to fix for all the documents. default to 150 but should be adapted to your data

Example usage:

```bash
python predict.py ./models/pretrained_model.pth --text="I love pizza !" --max_length=150
```


## Results

I have tested this model on a set of french labeled customer reviews (of over 3 millions rows). I reported the metrics in TensorboardX. 

I got the following results

||F1 score|Accuracy|
|-|-|-|
|train|0.965|0.9366|
|test|0.945|0.915|

<img src="./images/character-level-cnn/training_metrics.png">


## Download pretrained models

Sentiment analysis model on French customer reviews (3M documents): [download link](https://drive.google.com/file/d/1pmzeac-Vx07ScBL0S-xJ5EqRJYGdtWvh/view?usp=sharing)

When using it set `max_length` to 300 and `extra_characters` to "éàèùâêîôûçëïü" (accented letters)


## Contributions - PR are welcome:

Here's a non-exhaustive list of potential future features to add:

- Adapt the loss for multi-class classification 
- Log training and validation metrics for each epoch to a text file
- Provide notebook tutorials

If you feel like adding a feature or impproving something do not hesitate to submit a [pull request](https://github.com/ahmedbesbes/character-based-cnn/pulls)
