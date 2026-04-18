import sys
from src.components.dataingestion import DataIngestion
from src.components.datatransformation import DataTransformation
from src.components.modeltrainer import ModelTrainer
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # 1. Data Ingestion
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()

            # 2. Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            # 3. Model Training
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer(train_arr, test_arr)

            print("Training Pipeline Completed Successfully!")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()