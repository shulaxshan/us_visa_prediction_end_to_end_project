import sys
from us_visa.exception import CustomException
from us_visa.logger import logging
from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_ingestion import DataIngestionConfig
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.data_transformation import DataTransformationConfig
from us_visa.components.model_trainer import ModelTrainConfig
from us_visa.components.model_trainer import ModelTraining
from us_visa.constants import *

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainConfig()


    def start_data_ingestion(self):
        try:
            logging.info(">>>>>> stage Data Ingestion started <<<<<<<<<\n\nx===========x")
            data_ingestion=DataIngestion()
            data_ingestion.initiate_data_ingestion()
            logging.info("\n\n>>>>>> stage Data Ingestion completed <<<<<<<<<\n\nx===========x")

            return data_ingestion.ingestion_config.train_data_path, data_ingestion.ingestion_config.test_data_path
        except CustomException as e:
            logging.error(e,sys)


    def start_data_transformation(self):
        try:
            logging.info(">>>>>> stage Data Transformation started <<<<<<<<<\n\nx===========x")
            data_transformation=DataTransformation()
            data_transformation.initiate_data_transformation(DATA_INGESTION_TRAIN_PATH,DATA_INGESTION_TEST_PATH)
            logging.info("\n\n>>>>>> stage Data Transformation completed <<<<<<<<<\n\nx===========x")

            return (data_transformation.data_transformation_config.preprocessor_train_file_path,
                    data_transformation.data_transformation_config.preprocessor_train_file_path)
            
        except CustomException as e:
            logging.error(e,sys)


    def start_model_training(self):
        try:
            logging.info(">>>>>> stage model training started <<<<<<<<<\n\nx===========x")
            model_trainer=ModelTraining()
            model_trainer.initiate_model_trainer(DATA_TRANFORMATION_TRAIN_PATH,DATA_TRANFORMATION_TEST_PATH)
            logging.info("\n\n>>>>>> stage model training completed <<<<<<<<<\n\nx===========x")

            return model_trainer.model_trainer_config.prediction_df_file_path
            
        except CustomException as e:
            logging.error(e,sys)
            

    def run_pipeline(self, ) -> None:
        try:
            data_ingestion = self.start_data_ingestion()
            data_transformation = self.start_data_transformation()
            model_tainer = self.start_model_training()
            
        except CustomException as e:
            logging.error(e,sys)
     