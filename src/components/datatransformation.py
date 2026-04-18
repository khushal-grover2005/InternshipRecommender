import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    transformed_train_path: str = os.path.join('artifacts', "model.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            # Use 'skills' or 'cleaned_text' column - ensure this matches your CSV
            target_column = 'skills' 
            
            # Fill NaN values to avoid errors
            train_df[target_column] = train_df[target_column].fillna('')
            test_df[target_column] = test_df[target_column].fillna('')

            # 1000 features is plenty for internship matching and saves HUGE RAM
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

            logging.info("Applying TF-IDF transformation")
            
            # Transform to Sparse Matrix (Crucial for Render)
            train_feature_matrix = tfidf.fit_transform(train_df[target_column])
            
            logging.info(f"Sparse matrix shape: {train_feature_matrix.shape}")

            # Save the TF-IDF object (The Preprocessor)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=tfidf
            )

            # Save the Sparse Matrix (The Model/Database)
            save_object(
                file_path=self.data_transformation_config.transformed_train_path,
                obj=train_feature_matrix
            )

            return (
                train_feature_matrix,
                tfidf
            )

        except Exception as e:
            raise CustomException(e, sys)