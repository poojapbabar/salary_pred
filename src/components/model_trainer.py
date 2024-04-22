import os
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_function
from src.utils import model_performance
from src.pipeline import predication_pipeline

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_training(self, train_data, test_data, preprocessor_path):
        try:
            logging.info('Model training process initiated')

            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            # Define models
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "DecisionTree": DecisionTreeRegressor()
            }

            # Evaluate models
            model_report = model_performance(X_train, y_train, X_test, y_test, models)

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f'The best model is {best_model_name}, with R2 Score: {best_model_score}')

            # Save best model
            save_function(file_path=os.path.join('artifacts', 'model.pkl'), obj=best_model)
            logging.info('Best model saved as model.pkl')

            logging.info('Model training process completed')

        except Exception as e:
            logging.error('Error occurred during model training', exc_info=True)
            raise CustomException(e)
