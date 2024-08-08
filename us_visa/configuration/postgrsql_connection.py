import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import sys
from us_visa.logger import logging
from us_visa.exception import CustomException
from us_visa.constants import *
# import certifi


# ca = certifi.where()
class PostgresSQLConnection:
    def __init__(self, database_name=db_name,db_username=db_username,db_password= db_password,db_host= db_host,db_port=db_port) -> None:
        try:
            self.connection_string = f'postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{database_name}'
            self.engine = create_engine(self.connection_string)
            print("PostgreSQL connection succesfull")
            logging.info("PostgreSQL connection succesfull")
            # table_name = 'us_visa'
            # df1 = pd.read_sql_table(table_name, con=engine)
        except Exception as e:
            raise CustomException(e,sys)
        

    def get_engine(self):
        return self.engine