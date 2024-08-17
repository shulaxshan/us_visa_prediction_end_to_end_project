import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.logger import logging
from us_visa.exception import CustomException
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.constants import *



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(DATA_TRANSFORMATION_DIR_NAME, "preprocessor.pkl")
    preprocessor_train_file_path = os.path.join(DATA_TRANSFORMATION_DIR_NAME, "preprocessor_trained_data.npy")
    preprocessor_test_file_path = os.path.join(DATA_TRANSFORMATION_DIR_NAME, "preprocessor_test_data.npy")

    


class DataTransformation:
    def __init__(self):
        try:
            self.data_transformation_config = DataTransformationConfig()
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)


    def get_data_transformation_object(self):
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformation_object()
            logging.info("Got the preprocessor object")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #### Training datafram
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df= np.where(target_feature_train_df=='Denied', 1,0)
            logging.info("Got train features and test features of Training dataset")

            input_feature_train_df['company_age'] = CURRENT_YEAR-input_feature_train_df['yr_of_estab']
            logging.info("Added company_age column to the Training dataset")

            drop_cols = self._schema_config['drop_columns']
            logging.info("drop the columns in drop_cols of Training dataset")

            input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)

            #### Testing datafram
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df= np.where(target_feature_test_df=='Denied', 1,0)

            input_feature_test_df['company_age'] = CURRENT_YEAR-input_feature_test_df['yr_of_estab']
            logging.info("Added company_age column to the Test dataset")

            input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)
            logging.info("drop the columns in drop_cols of Test dataset")

            logging.info("Got train features and test features of Testing dataset")

            #### Apply preprocessing objects
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Used the preprocessor object to fit transform the train features")

            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Used the preprocessor object to transform the test features")

            logging.info("Applying SMOTEENN on Training dataset")
            smt = SMOTEENN(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                        input_feature_train_arr, target_feature_train_df)
            logging.info("Applied SMOTEENN on training dataset")

            logging.info("Applying SMOTEENN on testing dataset")
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                        input_feature_test_arr, target_feature_test_df)
            logging.info("Applied SMOTEENN on testing dataset")

            logging.info("Created train array and test array")
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.preprocessor_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.preprocessor_test_file_path, array=test_arr)

            return (self.data_transformation_config.preprocessor_train_file_path, 
                    self.data_transformation_config.preprocessor_test_file_path, 
                    self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)




