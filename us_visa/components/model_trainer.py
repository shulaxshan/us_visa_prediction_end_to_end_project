import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,classification_report

from us_visa.logger import logging
from us_visa.exception import CustomException
from us_visa.utils.main_utils import save_pickle, load_numpy_array_data
from us_visa.constants import *
from us_visa.utils.main_utils import evaluate_models

@dataclass
class ModelTrainConfig:
    model_obj_file_path = os.path.join(MODEL_TRAINER_DIR_NAME, "model.pkl")
    prediction_df_file_path = os.path.join(MODEL_TRAINER_DIR_NAME, "final_model_trained_results.txt")


class ModelTraining:
    def __init__(self):
        try:
            self.model_trainer_config = ModelTrainConfig()
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_model_trainer(self,train_array_path,test_array_path):
        try:
            train_array = load_numpy_array_data(train_array_path)
            test_array = load_numpy_array_data(test_array_path)
            logging.info("Sucessfully loaded train and test arrays")

            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1])
            

            models= {
                "AdaBoostClassifier": AdaBoostClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier()
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name}")

            # if best_model_score <0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both traing and testing dataset")

            save_pickle(
                file_path= self.model_trainer_config.model_obj_file_path,
                obj = best_model)
            logging.info("Successfully saved best model's pickle file")

            predicted = best_model.predict(X_test)
            accuracy_ = accuracy_score(y_test, predicted)
            print("Accuracy score: ", accuracy_)

            # Print the classification report
            report = classification_report(y_test, predicted)
            print("Classification Report:\n", report)


                        # Save results to text file
            with open(self.model_trainer_config.prediction_df_file_path, "w") as file:
                file.write(f"Best Model: {best_model_name}\n")
               # file.write(f"Best Model Score: {best_model_score:.4f}\n")
                file.write(f"Final Accuracy: {accuracy_:.4f}\n")
                file.write("Classification Report:\n")
                file.write(report)


            return self.model_trainer_config.prediction_df_file_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
