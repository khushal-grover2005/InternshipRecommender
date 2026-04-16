from src.logger import logging
from src.exception import CustomException
from src.components.dataingestion import DataIngestion
from src.components.datatransformation import DataTransformation
import sys

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data=DataIngestion()
        train_path,test_path=data.initiate_data_ingestion()

        transform=DataTransformation()

        train_arr, test_arr, preprocessor_path=transform.initiate_data_transformation(train_path=train_path,test_path=test_path)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)