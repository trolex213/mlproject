# responsible for fetching data from the source and storing it in the database. 
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd 

from src.exceptions import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# DataIngestion class is responsible for fetching data from the source and storing it in the database. Doing so removes the need to 
# explicitly define the data source and destination in different parts of the code, can simply call this class to fetch the necessary data files. 
@dataclass
class DataIngestionConfig:
    # 'artifact' is the directory where the data files are stored.
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv') 

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:

            data_file_path = os.path.join('notebook', 'data', 'stud.csv')
            df = pd.read_csv(data_file_path)
            logging.info("Data has been successfully read from the source")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split has been initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is now complete.")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)