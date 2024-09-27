import os
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame

from us_visa.constants import MODEL_FILE_PATH, PREPROCESSING_OBJECT_FILE_PATH
from us_visa.utils.main_utils import load_object
from us_visa.exception import CustomException
from us_visa.logger import logging


class usvisaData:
    def __init__(self,
                continent,
                education_of_employee,
                has_job_experience,
                requires_job_training,
                no_of_employees,
                region_of_employment,
                prevailing_wage,
                unit_of_wage,
                full_time_position,
                company_age
                ):
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age

        except CustomException as e:
            logging.error(e,sys)

    def get_usvisa_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            usvisa_input_dict = self.get_usvisa_data_as_dict()
            return DataFrame(usvisa_input_dict)
        
        except CustomException as e:
            logging.error(e,sys)


    def get_usvisa_data_as_dict(self):
        """
        This function returns a dictionary from USvisaData class input 
        """
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")

        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created usvisa data dict")
            logging.info("Exited get_usvisa_data_as_dict method as USvisaData class")

            return input_data

        except CustomException as e:
            logging.error(e,sys)




class PredictionPipeline:
    def __init__(self):
        pass


    def predict(self, dataframe) -> str:
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            model_file_path: str = MODEL_FILE_PATH
            preprocesser_path: str = PREPROCESSING_OBJECT_FILE_PATH
            model = load_object(file_path=model_file_path)
            preprocessor = load_object(file_path=preprocesser_path)
            data_scaled = preprocessor.transform(dataframe)
            result =  model.predict(data_scaled)
            
            return result
        
        except CustomException as e:
            logging.error(e,sys)
