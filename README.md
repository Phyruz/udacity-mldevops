# Udacity ML DevOps Engineer Nanodegree

This repositories contains all the project related to the [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) offered by Udacity.

## Structure 

Each folder in this repository contains all the code, directories and files related to each of the program's projects, namely:
1. [Predict Customer Churn with Clean Code](https://github.com/Phyruz/udacity-mldevops/tree/main/predict_churn_with_clean_code)
2. [Build an ML Pipeline for Short-term Rental Prices in NYC](https://github.com/Phyruz/udacity-mldevops/tree/main/build_an_ml_pipeline_for_short_term_rental_prices_in_nyc)
3. Deploying a Machine Learning Model on Heroku with FastAPI (WIP)
4. A Dynamic Risk Assessment System (WIP)

### 1. Predict Customer Churn with Clean Code

In this project, I implemented clean code practices in the context of implementing a customer churn prediction model. The completed project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

### 2. Build an ML Pipeline for Short-term Rental Prices in NYC (WIP)
In this project, I write a Machine Learning Pipeline to solve the following problem: a property management company is renting rooms and properties in New York for short periods on various rental platforms. They need to estimate the typical price for a given property based on the price of similar properties. The company receives new data in bulk every week, so the model needs to be retrained with the same cadence, necessitating a reusable pipeline. The end-to-end pipeline covering data fetching, validation, segregation, train and validation, test, and release. It is run on an initial data sample, and then re-run it on a new data sample simulating a new data delivery.

### 3. Deploying a Machine Learning Model on Heroku with FastAPI
In this project, I deploy a machine learning model on Heroku, using Git and DVC to track the code, data, and model while developing a simple classification model on the Census Income Data Set. After developing the model, it is released into production by checking its performance on slices and writing a model card encapsulating key knowledge about the model. A Continuous Integration and Continuous Deployment framework is enstablished to ensure that the pipeline passes a series of unit tests before deployment. Lastly, an API is written using FastAPI and tested locally. After successful deployment the API is tested live using the requests module.

### 4. A Dynamic Risk Assessment System (WIP)

