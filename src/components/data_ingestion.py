import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion process initiated')
        try:
            # Read the raw data
            df = pd.read_csv(os.path.join(r'notebook\adult.csv'))
            logging.info('Raw dataset read as pandas DataFrame')

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data to file
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f'Raw data saved to {self.ingestion_config.raw_data_path}')

            # Perform train-test split
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save train and test data to files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion process completed')

            # Return paths to train and test data files
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error('Exception occurred during data ingestion process', exc_info=True)
            raise CustomException(e, sys)
