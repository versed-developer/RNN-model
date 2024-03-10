# Create work environment

## Setup virtual environment and activate it
`python -m venv env`

`.\env\Scripts\activate`

## Install required dependencies
`pip install -r requirements.txt`

# Train model and save the best one
Inside virtual env, run below command to train model
`python lstm.py`

After trained, it will ask you to save model or not.
If you type `y` or `Y`, it will save the trained model.


Before deciding to save model or not, you can check accuracy graph by running below command.

`tensorboard --logdir ./logs`

Once the above command run successfully, please open browser and visit `localhost:6006`

# Predict data and write predicted phases to excel files
`python evaluate.py`

# Updated steps of the project

## First step
user is asked to select the training file : ( from the following files) ( down-1, down-5, down-10, down-15, down-20, down-30,down-60)
## Second step
training the model with LSTM and extract the highest accuracy
## Third step
model is saved and write result in a new excel file(e.g. tr1.excel which has one sheet with two columns-G truth and predictions)
## Fourth step 
user is asked to select the testing file : ( form the following files) ( down-1, down-5, down-10, down-15, down-20, down-30,down-60)
## Last step 
Print the accuracy with training and testing file name and write the prediction result in a new excel file (e.g. tr1ts5.excel which has one sheet with two columns-G truth and predictions)
