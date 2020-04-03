---
title: "End to end machine learning: from data collection to deployment 🚀"
excerpt: "Learn how to build and deploy a machine learning application from scratch: an end-to-end tutorial to learn scraping, training a character level CNN for text classification, buidling an interactive responsive web app with Dash and Docker and deploying to AWS. You're in for a treat !"
slug: end-to-end-ml-app

iframe: //miro.medium.com/max/1400/1*lY9P1H0d5vUKmdjriE05Ng.png
demo: //
src: //github.com/MarwanDebbiche/post-tuto-deployment
badgeUrl: "https://ghbtns.com/github-btn.html?user=MarwanDebbiche&repo=post-tuto-deployment&type=star&count=true" 


info:
  idea: Learn how to implement a convolutional neural network that classifies knee injuries from MRI exams
  tech: [Scrapy, Selenium, PyTorch, Dash, Docker, aws]
  links: 
    - [Blog Post, https://zuhair.netlify.com/blog/end-to-end-machine-learning]
---

<p align="center">
    <img src="miro.medium.com/max/1400/1*lY9P1H0d5vUKmdjriE05Ng.png"  width="100%">
</p>

In this job, I collaborated with <a href="https://github.com/MarwanDebbiche">Marwan Debbiche</a>

You may read about it <a href="https://zuhairabs.netlify.com/blog/end-to-end-machine-learning">here</a> and <a href="https://marwandebbiche.com/posts/e2e-ml/">here</a>.

In this post, we'll go through the necessary steps to build and deploy a machine learning application. This starts from data collection to deployment; and the journey, you'll see, is exciting and fun. 😀

Before we begin, let's have a look at [the app](https://www.reviews.ai2prod.com/) we'll build:

<p align="center">
    <video height="400" autoplay="autoplay" controls loop>
    <source src="https://s3.eu-west-3.amazonaws.com/ahmedbesbes.com-assets/app.mp4" type="video/mp4">
    </video>
</p>

As you see, this web app allows a user to evaluate random brands by writing reviews. While writing, the user will see the sentiment score of his input updating in real-time, alongside a proposed 1 to 5 rating.

The user can then change the rating in case the suggested one does not reflect his views, and submit.

You can think of this as a crowd sourcing app of brand reviews, with a sentiment analysis model that suggests ratings that the user can tweak and adapt afterwards.

To build this application, we'll follow these steps:

- Collecting and scraping customer reviews data using `Selenium` and `Scrapy`
- Training a deep learning sentiment classifier on this data using `PyTorch`
- Building an interactive web app using `Dash`
- Setting a `REST API` and a `Postgres` database
- Dockerizing the app using `Docker Compose`
- Deploying to `AWS`

<hr>

## Run the app locally


To run this project locally using `Docker Compose` `run`: 

```
docker-compose build
docker-compose up
```
You can then access the dash app at [http://localhost:8050](http://localhost:8050)

## Development

If you want to contribute to this project and run each service independently:

### Launch API

In order to launch the API, you will first need to run a local `postgres` db using `Docker`:

```
docker run --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres -p 5432:5432 -d postgres
```

Then you'll have to type the following commands:

```shell
cd src/api/
python app.py
```

### Launch Dash app

In order to run the `dash` server to visualize the output:

```shell
cd src/dash/
python app.py
```


## How to contribute 😁

Feel free to contribute! Report any bugs in the [issue section](https://github.com/MarwanDebbiche/post-tuto-deployment/issues).

Here are the few things we noticed, and wanted to add.

- [ ] Add server-side pagination for Admin Page and `GET /api/reviews` route.
- [ ] Protect admin page with authentication.
- [ ] Either use [Kubernetes](https://kubernetes.io) or [Amazon ECS](https://aws.amazon.com/ecs) to deploy the app on a cluster of containers, instead of on one single EC2 instance.
- [ ] Use continuous deployment with [Travis CI](https://travis-ci.org)
- [ ] Use a managed service such as [RDD](https://aws.amazon.com/rds/) for the database


## Licence

MIT
