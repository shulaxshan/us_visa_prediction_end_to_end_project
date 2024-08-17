import os
from datetime import date
import urllib.parse

db_username = 'postgres'
db_password = urllib.parse.quote('abc@123')
db_host = 'localhost'
db_port = '5432'
db_name = 'my_learning_db'

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")



""""
Data Ingestion related constant start with DATA_INGESTION VAR NAME

"""
DATA_INGESTION_DIR_NAME: str = 'artifacts/data_ingestion'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
table_name: str = 'us_visa'



"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"



""""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME

"""
DATA_TRANSFORMATION_DIR_NAME: str = 'artifacts/data_transformation'
DATA_INGESTION_TRAIN_PATH: str = 'artifacts/data_ingestion/train.csv'
DATA_INGESTION_TEST_PATH: str = 'artifacts/data_ingestion/test.csv'



""""
Model Trainer related constant start with MODEL_TRAINER VAR NAME

"""
MODEL_TRAINER_DIR_NAME: str = 'artifacts/model_trainer'
DATA_TRANFORMATION_TRAIN_PATH: str = 'artifacts/data_transformation/preprocessor_trained_data.npy'
DATA_TRANFORMATION_TEST_PATH: str = 'artifacts/data_transformation/preprocessor_test_data.npy'



PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"