---
title: Image Dataset Builder
excerpt: A script to help you quickly build custom computer vision datasets for object classification, detection or segmentation.

iframe: //www.youtube.com/embed/qXLvMr9mrP4/?modestbranding=1&showinfo=0&autohide=1&rel=0
demo: //www.youtube.com/embed/qXLvMr9mrP4/?modestbranding=1&showinfo=0&autohide=1&rel=0
src: //github.com/ahmedbesbes/dataset-builder

info:
  idea: The main idea is to provide a script for quickly building custom computer vision datasets for classification, detection or segmentation
  tech: [Pyton, Selenium, makesense.ai, Scraping]
  links:
---

This script is meant to help you quickly build custom computer vision datasets for classification, detection or segmentation: it doesn't do the labeling for you. But it takes care of the steps beforehand:

- Define your set of classes
- Scrape the data for each class
- Rename the files
- Organize the folder structure
- Upload the data to [makesense.ai](http://makesense.ai) for the annotation (objection detection or segmentation)


If you opt for the detection task, the script uploads the downloaded images with the corresponding labels to http://makesense.ai (or locally to http://localhost:3000) so that all you have to do in annotate yourself.

Once the annotation is done, your labels can be exported and you'll be ready to train your awesome models.

## Requirements

- google\_images\_download 
```shell 
pip install google_images_download
```
- Selenium 
```shell
pip install -U selenium
```
- ChromeDriver 77.0.3865.40

If you wish to run Make Sense locally: 

```shell
# clone repository
git clone https://github.com/SkalskiP/make-sense.git

# navigate to main dir
cd make-sense

# install dependencies
npm install

# serve with hot reload at localhost:3000
npm start
```

## How to use the code

When you run the script, you can specify the following arguments:

- `output_directory`: the root folder when images are downloaded
- `limit`: the maximum number of downloaded images per category
- `delete_history`: whether you choose to erase previous downloads or not
- `task`: classification, detection or segmentation
- `driver`: path to chrome driver
- `run_local` : whether or not to use makesense locally

```shell
python dataset_builder.py --limit 20 --delete_history yes
```

Once the script runs, you'll be asked to define your classes (or queries)

<img src="./images/dataset-builder/class_names.png">

Here's what the output looks like after the download:

<img src="./images/dataset-builder/downloaded_files.png" width="50%">


## Object Detection

This only works if you choose a detection or segmentation `task`.

Make Sense is an awesome open source webapp that lets you easily label your image dataset for tasks such as localization.

You can check it out here: https://www.makesense.ai/ You can also clone it and run it locally (for better performance): https://github.com/SkalskiP/make-sense

In order to use this tool, I'll be running it locally and interface with it using Selenium: Once the dataset is downloaded, Selenium opens up a Chrome browser, upload the images to the app and fill in the label list: this ultimately allows you to annotate.

## To Do Later 😁

Please feel free to contribute ! Report any bugs in the issue section, or request any feature you'd like to see shipped:

- [ ] Accelerate the download of images via multiprocessing
- [ ] Apply a quality check on the images
- [ ] Integrate automatic tagging using pre-trained networks