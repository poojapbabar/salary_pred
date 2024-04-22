import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_function

@dataclass
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation pipeline initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['occupation', 'relationship']
            numerical_cols = ['age', 'educational-num', 'hours-per-week']

            # Define custom categories for ordinal encoding
            workclass_cat = ['Private', 'Local-gov', 'Self-emp-not-inc', 'Federal-gov',
                             'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']
            occupation_cat = ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv',
                              'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical',
                              'Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv',
                              'Transport-moving', 'Handlers-cleaners', 'Armed-Forces']

            # Numerical pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[workclass_cat, occupation_cat])),
                ('scaler', StandardScaler())
            ])

            # Column transformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info('Data transformation pipeline completed')

            return preprocessor

        except Exception as e:
            logging.error("Error occurred in data transformation", exc_info=True)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Data transformation initiated')

            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformation_object()

            # Define columns to remove
            target_column_name = 'income'
            remove_columns = ['workclass', 'capital-gain', 'capital-loss', 'race', 'fnlwgt',
                              'gender', 'marital-status', 'native-country', 'education']

            # Separate input features and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name] + remove_columns)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name] + remove_columns)
            target_feature_test_df = test_df[target_column_name]

            # Transform data using preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features with target and save preprocessor object
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            save_function(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                          obj=preprocessing_obj)
            logging.info('Preprocessor object saved')

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error occurred in data transformation", exc_info=True)
            raise CustomException(e, sys)
