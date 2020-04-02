---
title: "Deep learning for knee injury diagnosis"
excerpt: "This repository contains an implementation of a convolutional neural network that classifies specific knee injuries from MRI exams. Check it if you want to learn more or to adapt the code to another medical imaging problem."
slug: mrnet network

iframe: //https://stanfordmlgroup.github.io/projects/mrnet/img/fig3.png
demo: //
src: //github.com/zuhairabs/
badgeUrl: "https://ghbtns.com/github-btn.html?user=zuhairabs&repo=&type=star&count=true" 


info:
  idea: Learn how to implement a convolutional neural network that classifies knee injuries from MRI exams
  tech: [PyTorch]
  links: 
    - [Stanford ML Group, https://stanfordmlgroup.github.io/competitions/mrnet/]
    - ["Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet", https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002699&type=printable]
---

[![MIT](https://img.shields.io/badge/license-MIT-5eba00.svg)](https://github.com/ahmedbesbes/character-based-cnn/blob/master/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ahmedbesbes/mrnet)
[![Twitter](https://img.shields.io/twitter/follow/ahmed_besbes_.svg?label=Follow&style=social)](https://twitter.com/ahmed_besbes_)
[![Stars](https://img.shields.io/github/stars/ahmedbesbes/character-based-cnn.svg?style=social)](https://github.com/ahmedbesbes/mrnet/stargazers)


This repository contains an implementation of a convolutional neural network that classifies specific knee injuries from MRI exams.

It also contains the matieral of a series of posts I wrote on <a href="http://ahmedbesbes.com"> my blog</a>.

## Dataset: MRNet 

The data comes from Stanford ML Group research lab. It consits of 1,370 knee MRI exams performed at Stanford University Medical Center to study the presence of Anterior Cruciate Ligament (ACL) tears.

For more information about the ACL tear problem and the MRNet data please refer to my blog post where you can investigate the data and build the following data visualization in jupyter notebook:


<p align="center">
    <img src="./images/knee-injury-detection/mri.gif"  width="100%">
</p>

To learn more about the data and how to realize this visualization widget, read <a href="https://ahmedbesbes.com/automate-the-diagnosis-of-knee-injuries-with-deep-learning-part-1-an-overview-of-the-mrnet-dataset.html">my first post.</a>

## Code structure:

This charts summarizes the architecture of the project:

<img src="./images/knee-injury-detection/pipeline.png">

For more details about the code, please refer to my second <a href="https://ahmedbesbes.com/automate-the-diagnosis-of-knee-injuries-with-deep-learning-part-2-building-an-acl-tear-classifier.html">blog post </a>.

## How to use the code:

If you want to retrain the network on your own you have to ask for the data from Stanford via this <a href="https://stanfordmlgroup.github.io/competitions/mrnet/">link</a>.

Once you download the data, create a `data` folder and place it at the root of the project. You should have two folders inside: `train` and `valid` as well as a bunch of csv files.

To run the script you can execute it with the following arguments:

```python
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, required=True,
                    choices=['abnormal', 'acl', 'meniscus'])
parser.add_argument('-p', '--plane', type=str, required=True,
                    choices=['sagittal', 'coronal', 'axial'])
parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
parser.add_argument('--lr_scheduler', type=int, choices=[0, 1], default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
parser.add_argument('--patience', type=int, choices=[0, 1], default=5)
```

example to train a model to detect acl tears on the sagittal plane for a 20 epochs:

```bash
python -t acl -p sagittal --epochs=20
```

Note: Before running the script, add the following (empty) folders at the root of the project:
- models
- logs


## Results:

I trained an ACL tear classifier on a sagittal plane and got the following AUC scores:

- on train: 0.8669
- on validation: 0.8850

Logs on Tensorboard:

<img src="./images/knee-injury-detection/sagittal_tensorboard.png">


## Contributions - PR are welcome:

If you feel that some functionalities or improvements could be added to the project, don't hesitate to submit a pull request.

