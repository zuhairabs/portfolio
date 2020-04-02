---
title: Neural Networks from Scratch
excerpt: Learn how to build and train a neural network from scratch. In pure Python only with no frameworks involved. This script helps you start this project.

iframe: //http://github.com/zuhairabs/portfolio/blob/master/content/case-studies/images/neural-networks-from-scratch/backprop.gif
demo: //
src: //github.com/zuhairabs/
badgeUrl: "https://ghbtns.com/github-btn.html?user=zuhairabs&repo=&type=star&count=true" 
info:
  idea: The main idea is to understand how back-propagation works and how to implement it, for scratch, to train neural networks. This script is here to help you.
  tech: [Pyton, Numpy]
  links:
---



## Learn backpropagtion the hard way

<p align="center">
  <img src="./images/neural-networks-from-scratch/backprop.gif" width="100%">
</p>

In this repository, I will show you how to build a neural network from scratch (yes, by using plain python code with no framework involved) that trains by mini-batches using gradient descent. Check **nn.py** for the code.

In the related notebook **Neural\_Network\_from\_scratch\_with\_Numpy.ipynb** we will test nn.py on a set of non-linear classification problems

- We'll train the neural network for some number of epochs and some hyperparameters
- Plot a live/interactive decision boundary 
- Plot the train and validation metrics such as the loss and the accuracies


## Example: Noisy Moons (Check the notebook for other kinds of problems)

> Decision boundary (you'll get to this graph animated during training)

![Decision boundary](images/neural-networks-from-scratch/decision_boundary.png)

> Loss and accuracy monitoring on train and validation sets 

![Loss/Accuracy monitoring on train/val](images/neural-networks-from-scratch/loss_acc.png)


## Where to go from here?
nn.py is a toy neural network that is meant for educational purposes only. So there's room for a lot of improvement if you want to pimp it. Here are some guidelines:

- Implement a different loss function such as the Binary Cross Entropy loss. For a classification problem, this loss works better than a Mean Square Error. 
- Make the code generic regarding the activation functions so that we can choose any function we want: ReLU, Sigmoid, Tanh, etc.
- Try to code another optimizers: SGD is good but it has some limitations: sometimes it can be stuck in local minima. Look into Adam or RMSProp.
- Play with the hyperparameters and check the validation metrics

