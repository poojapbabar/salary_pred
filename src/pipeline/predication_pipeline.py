import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj
import pandas as pd

class PredictPipeline: 
    def __init__(self):
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            logging.info("Error occurred in predict function in prediction_pipeline location")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, age: int, educational_num: int, hours_per_week: int, workclass: str, occupation: str):
        self.age = age
        self.educational_num = educational_num
        self.hours_per_week = hours_per_week
        self.workclass = workclass 
        self.occupation = occupation

    def get_data_as_dataframe(self): 
        try: 
            custom_data_input_dict = {
                'age': [self.age], 
                'educational-num': [self.educational_num], 
                'hours-per-week': [self.hours_per_week], 
                'workclass': [self.workclass], 
                'occupation': [self.occupation]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created")
            return df
        except Exception as e:
            logging.info("Error occurred in get_data_as_dataframe function in prediction_pipeline")
            raise CustomException(e, sys) 
