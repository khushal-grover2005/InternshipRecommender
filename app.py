import os
import sys
import pandas as pd
import PyPDF2
from flask import Flask, request, render_template

# Fix for Render: Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
            
            # 1. Handle PDF
            if 'resume_file' in request.files and request.files['resume_file'].filename != '':
                file = request.files['resume_file']
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    user_input += page.extract_text()
            else:
                user_input = request.form.get('user_input')

            if not user_input or len(user_input.strip()) < 10:
                return render_template('index.html', error="Input too short.")

            # 2. Run Prediction
            pipeline = PredictPipeline()
            indices, scores = pipeline.predict(user_input)
            
            # 3. Load data for mapping
            df = pd.read_csv(DATA_PATH)
            results = []
            for i, idx in enumerate(indices):
                results.append({
                    "position": df.iloc[idx]['positions'],
                    "skills": df.iloc[idx]['skills'],
                    "score": f"{int(float(scores[i])*100)}%" 
                })
                
            return render_template('index.html', results=results, user_text="Processing complete.")

    except Exception as e:
        # Log error to Render console
        print(f"Error occurred: {str(e)}")
        return render_template('index.html', error="Server busy. Please try with less text.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)