import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import dill
import numpy as np 
import pandas as pd 
from src.exceptions import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        # loop through each model in models 
        for i in range(len(list(models))):
            # get the model name
            model = list(models.values())[i]
            
            model.fit(X_train, y_train) # train model 

            y_train_pred = model.predict(X_train) # predict on training data

            y_test_pred = model.predict(X_test) # predict on test data

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            # store the model name and its score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

            return report 
        
    except Exception as e:
        raise CustomException(e, sys)
