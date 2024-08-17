import os
import sys
from us_visa.exception import CustomException
from us_visa.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from us_visa.configuration.postgrsql_connection import PostgresSQLConnection
from us_visa.constants import *


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join(DATA_INGESTION_DIR_NAME,"train.csv")
    test_data_path: str=os.path.join(DATA_INGESTION_DIR_NAME,"test.csv")
    raw_data_path: str=os.path.join(DATA_INGESTION_DIR_NAME,"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            postgres_connection = PostgresSQLConnection()
            engine = postgres_connection.get_engine()

            df = pd.read_sql_table(table_name, con=engine)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__=="__main__":
#     obj=DataIngestion()
#     obj.initiate_data_ingestion()



