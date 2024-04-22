import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        logging.info('Data ingestion process initiated')
        try:
            # Read the raw data
            df = pd.read_csv(os.path.join(r'notebook\adult.csv'))
            logging.info('Raw dataset read as pandas DataFrame')

            # Save raw data to file
            df.to_csv(os.path.join('artifacts', 'raw.csv'), index=False)
            logging.info('Raw data saved')

            logging.info('Data ingestion process completed')

            # Return path to the raw data file
            return os.path.join('artifacts', 'raw.csv')

        except Exception as e:
            logging.error('Exception occurred during data ingestion process', exc_info=True)
            raise CustomException(e)
