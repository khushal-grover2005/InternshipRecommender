import sys
import os
from dataclasses import dataclass
import pandas as pd
import spacy
import re
from src.exception import CustomException
from src.logger import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import save_object

# Load the model here once
nlp = spacy.load("en_core_web_sm")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text=text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        doc = nlp(text)
        
        cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

        
        return " ".join(cleaned_tokens)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
           train_df = pd.read_csv(train_path)
           test_df = pd.read_csv(test_path)

           logging.info('Read train and test data completed')

           
           text_columns = ['career_objective', 
                            'skills', 
                            'degree_names', 
                            'major_field_of_studies', 
                            'responsibilities', 
                            'positions', 
                            'certification_skills',
                            'educationaL_requirements']

            
           for df in [train_df, test_df]:
               df['Content'] = df[text_columns].fillna('').agg(' '.join, axis=1)

           logging.info('Combined columns into "Content" column')

           
           train_df['Content'] = train_df['Content'].apply(self.clean_text)
           test_df['Content'] = test_df['Content'].apply(self.clean_text)

           logging.info('Data cleaned') 
            
           tfidf = TfidfVectorizer(max_features=1000)

            
           train_feature_arr = tfidf.fit_transform(train_df['Content']).toarray()
           test_feature_arr = tfidf.transform(test_df['Content']).toarray()

           save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=tfidf

            )

           logging.info(f"Saved preprocessing object.")
            

           return (
                train_feature_arr,
                test_feature_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

           
        except Exception as e:
            raise CustomException(e,sys)