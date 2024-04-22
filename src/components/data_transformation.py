import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_function

class DataTransformation:
    def __init__(self):
        pass

    def initiate_data_transformation(self, raw_data_path):
        try:
            logging.info('Data transformation initiated')

            # Read raw data
            df = pd.read_csv(raw_data_path)

            # Define columns for transformation
            numerical_cols = ['age', 'educational-num', 'hours-per-week']
            categorical_cols = ['workclass', 'occupation']

            # Define preprocessing pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ])

            # Fit and transform data
            transformed_data = preprocessor.fit_transform(df)

            # Save preprocessing object
            save_function(file_path=os.path.join('artifacts', 'preprocessor.pkl'), obj=preprocessor)
            logging.info('Preprocessing object saved')

            logging.info('Data transformation completed')

            # Return transformed data and preprocessing object path
            return transformed_data, os.path.join('artifacts', 'preprocessor.pkl')

        except Exception as e:
            logging.error('Error occurred during data transformation', exc_info=True)
            raise CustomException(e)
