import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import PyPDF2
from flask import Flask, request, render_template

from src.pipeline.predictpipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

DATA_PATH = os.path.join("artifacts", "train.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if request.method == 'POST':
            user_input = ""
            
            # 1. Check if a File was uploaded
            if 'resume_file' in request.files and request.files['resume_file'].filename != '':
                file = request.files['resume_file']
                if file.filename.endswith('.pdf'):
                    logging.info("PDF Resume detected, extracting text...")
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        user_input += page.extract_text()
                else:
                    return render_template('index.html', error="Currently, only PDF files are supported.")
            
            # 2. If no file, fall back to Textarea
            else:
                user_input = request.form.get('user_input')

            # Validation
            if not user_input or len(user_input.strip()) < 10:
                return render_template('index.html', error="Please upload a resume or paste your skills.")

            # 3. Prediction Pipeline
            predict_pipeline = PredictPipeline()
            indices, scores = predict_pipeline.predict(user_input)
            
            df = pd.read_csv(DATA_PATH)
            
            results = []
            for i, idx in enumerate(indices):
                match_percentage = int(float(scores[i]) * 100)
                results.append({
                    "position": df.iloc[idx]['positions'],
                    "skills": df.iloc[idx]['skills'],
                    "score": f"{match_percentage}%" 
                })
                
            return render_template('index.html', results=results, user_text=user_input if not request.files['resume_file'].filename else "Extracted from PDF")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)