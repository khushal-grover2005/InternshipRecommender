from src.logger import logging
from src.exception import CustomException
from src.components.dataingestion import DataIngestion
import sys

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data=DataIngestion()
        train_path,test_path=data.initiate_data_ingestion()

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)