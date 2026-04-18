import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 
from sklearn.metrics.pairwise import cosine_similarity

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            
            data_scaled = preprocessor.transform([features])

            
            similarity_scores = cosine_similarity(data_scaled, model)

            
            top_indices = similarity_scores[0].argsort()[-5:][::-1]

            return top_indices, similarity_scores[0][top_indices]

        except Exception as e:
            raise CustomException(e, sys)