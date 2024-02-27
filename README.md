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