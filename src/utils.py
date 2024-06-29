import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import dill
import numpy as np 
import pandas as pd 
from src.exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)