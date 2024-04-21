# import os
# import sys
# import logging
# import pandas as pd
# from src.exception import CustomException
# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass


# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

# logging.basicConfig(level=logging.INFO)  # Configure logging

# @dataclass
# class DataIngestionConfig:
#     train_data_path: str = os.path.join('artifacts', 'train.csv')
#     test_data_path: str = os.path.join('artifacts', 'test.csv')
#     raw_data_path: str = os.path.join('artifacts', 'data.csv')

# class DataIngestion:
#     def __init__(self):
#         self.ingestion_config = DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         logging.info("Entered the data ingestion method or component")

#         try:
#             data = pd.read_csv('notebook/adult.csv')  # Corrected path separator
#             logging.info("Read the dataset as a dataframe")

#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

#             data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
#             logging.info("Train test split initiated")
#             train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

#             train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
#             test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

#             logging.info("Ingestion of the data is completed")

#             return (
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#             )

#         except Exception as e:
#             raise CustomException(e, sys)

# if __name__ == "__main__":  # Moved the block outside the class definition
#     obj = DataIngestion()
#     train_data,test_data = obj.initiate_data_ingestion()


#     data_transformation = DataTransformation()
#     data_transformation.initiate_data_transformation(train_data,test_data)


import sys
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['age', 'educational-num', 'hours-per-week']
            categorical_columns = ['occupation', 'relationship']
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "income"

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Convert y_train and y_test to NumPy arrays before applying multi-dimensional indexing
            y_train_array = y_train.values
            y_test_array = y_test.values

            # Concatenate transformed features with target variable
            train_arr = np.concatenate([X_train_transformed, y_train_array[:, np.newaxis]], axis=1)
            test_arr = np.concatenate([X_test_transformed, y_test_array[:, np.newaxis]], axis=1)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
