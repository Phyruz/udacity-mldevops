# Predict Customer Churn

Project Title:
**Build an ML Pipeline** of ML DevOps Engineer Nanodegree Udacity course.
 
## Project Description
In this project, I am implementing a re-usable end-to-end ML pipeline within the [Machine Learning DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) offered by Udacity. 

The project start from a [Starter Kit](https://github.com/udacity/nd0821-c2-build-model-workflow-starter), which has been forked in my GitHub account.

## Set-up

### Create Environment
Create a new environment using the environment.yml file provided in the root of the depository and activate it:

```
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```
### Get API key for W&B

Make sure you have an account and logged in W&B.
```
> wandb login [your API key]
```
### Cookie cutter

In order to make your job a little easier, you are provided a cookie cutter template that you can use to create stubs for new pipeline components. It is not required that you use this, but it might save you from a bit of boilerplate code. Just run the cookiecutter and enter the required information, and a new component will be created including the conda.yml file, the MLproject file as well as the script. You can then modify these as needed, instead of starting from scratch. For example:
```
> cookiecutter cookie-mlflow-step -o src

step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
parameters [parameter1,parameter2]: parameter1,parameter2,parameter3
```

### Parameter configuration

All parameters controlling the pipeling are defined in the config.yaml file defined in the root of the starter kit. Within this project, we use Hydra to manage this configuration file.