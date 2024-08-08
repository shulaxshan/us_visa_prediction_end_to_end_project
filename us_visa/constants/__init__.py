import os
from datetime import date
import urllib.parse

db_username = 'postgres'
db_password = urllib.parse.quote('abc@123')
db_host = 'localhost'
db_port = '5432'
db_name = 'my_learning_db'

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"