import pandas as pd 
import numpy as np 
import sys 
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging 
from src.exception import CustomException 
from dataclasses import dataclass
from src.utils import save_function 
from src.utils import model_performance 

@dataclass 
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array): 
        try: 
            logging.info("Segregating the dependent and independent variables")
            X_train, y_train, X_test, y_test = (train_array[:, :-1], 
                                                train_array[:, -1], 
                                                test_array[:, :-1], 
                                                test_array[:, -1])
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(), 
                "Lasso": Lasso(), 
                "ElasticNet": ElasticNet(), 
                "DecisionTree": DecisionTreeRegressor()
            }
            model_report: dict = model_performance(X_train, y_train, X_test, y_test, models)

            logging.info("Model evaluation complete")
            logging.info(f"Model Report: {model_report}")

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"The best model is {best_model_name}, with R2 Score: {best_model_score}")
            
            # Save best model
            save_function(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info(f"Best model saved to {self.model_trainer_config.trained_model_file_path}")

        except Exception as e: 
            logging.error("Error occurred during model training", exc_info=True)
            raise CustomException(e, sys)
