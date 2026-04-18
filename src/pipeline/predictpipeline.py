import sys
import os
import pandas as pd
import gc  # Garbage Collector to save RAM
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

            # 1. Load the objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # 2. Transform input
            data_scaled = preprocessor.transform([features])

            # 3. Calculate Similarity
            similarity_scores = cosine_similarity(data_scaled, model)

            # 4. Get Top 5
            top_indices = similarity_scores[0].argsort()[-5:][::-1]
            scores = similarity_scores[0][top_indices]

            # 5. CRITICAL: Manual Memory Cleanup for Render Free Tier
            del model
            del preprocessor
            gc.collect()

            return top_indices, scores

        except Exception as e:
            raise CustomException(e, sys)