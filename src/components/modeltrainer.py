import os
import sys
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        
        try:
            logging.info("Model Trainer: Received vectorized features")

            # We calculate self-similarity to ensure the vectors are well-formed
            # This is essentially the 'evaluation' step for a recommender
            scores = cosine_similarity(train_array)
            
            logging.info(f"Similarity matrix calculated with shape: {scores.shape}")

            # We save the train_array (the numerical representations of the internships)
            # because that is what we need to compare against during prediction.
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=train_array
            )
            
            logging.info("Model (Feature Matrix) saved successfully.")
            
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)