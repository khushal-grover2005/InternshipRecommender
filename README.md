# NexStep - AI Internship Recommender

An intelligent AI-powered system that matches students with suitable internship opportunities based on their skills, profile, and experience using Natural Language Processing (NLP) and cosine similarity algorithms.

## 🌟 Features

- **Smart Profile Analysis**: Parses student profiles and extracts key skills and experiences
- **PDF Resume Support**: Upload PDF resumes for automatic text extraction and analysis
- **AI-Powered Matching**: Uses TF-IDF vectorization and cosine similarity for intelligent matching
- **Real-time Recommendations**: Get instant internship recommendations based on your profile
- **Match Scoring**: Displays match percentage for each recommended position
- **Beautiful UI**: Modern, responsive web interface with smooth animations
- **Flask Backend**: Lightweight and scalable REST API

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework for building REST APIs
- **Python 3.x**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library for TF-IDF vectorization and cosine similarity
- **Spacy**: NLP library for text processing
- **PyPDF2**: PDF file handling and text extraction
- **Dill**: Object serialization for model persistence

### Frontend
- **HTML5**: Semantic markup
- **Bootstrap 5**: Responsive CSS framework
- **Animate.css**: Smooth animations
- **JavaScript**: Interactive elements

### Development
- **Gunicorn**: WSGI HTTP server for production
- **Git**: Version control

## 📁 Project Structure

```
InternshipRecommender/
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Package configuration
├── README.md                       # Project documentation
│
├── src/                           # Source code directory
│   ├── __init__.py
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   ├── utils.py                   # Utility functions
│   │
│   ├── components/                # Core ML components
│   │   ├── __init__.py
│   │   ├── dataingestion.py       # Data loading and splitting
│   │   ├── datatransformation.py  # Data preprocessing and vectorization
│   │   └── modeltrainer.py        # Model training and similarity calculation
│   │
│   └── pipeline/                  # ML pipelines
│       ├── predictpipeline.py     # Inference pipeline
│       └── trainpipeline.py       # Training pipeline
│
├── templates/                      # HTML templates
│   └── index.html                 # Main web interface
│
├── artifacts/                      # Model and data artifacts
│   ├── train.csv                  # Training internship data
│   ├── test.csv                   # Test data
│   ├── data.csv                   # Full dataset
│   └── [model files]              # Trained model artifacts
│
├── Dataset/                        # Source datasets
│   └── data.csv
│
└── logs/                          # Application logs
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd InternshipRecommender
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model** (if not already trained)
   ```bash
   python src/pipeline/trainpipeline.py
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Web Interface**
   Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

## 📖 Usage

### Web Interface

1. **Option 1: Paste Your Skills**
   - Fill in the textarea with your profile information
   - Include skills, projects, achievements, and experience
   - Click "Analyze & Match Now"

2. **Option 2: Upload Resume (PDF)**
   - Upload a PDF resume
   - The system extracts text automatically
   - Get instant recommendations

### Example Input
```
I am a pre-final year CS student at Thapar University. 
I have expertise in Python, Machine Learning with TensorFlow, 
and Data Analysis. I have won SIH 2024 and have 6 months of 
experience in web development with Django and React.
```

### Output
- List of recommended internship positions
- Match percentage for each position
- Key requirements and skills for each internship

## 🧠 How It Works

### Architecture Overview

```
User Input (Text/PDF)
    ↓
Text Preprocessing & NLP
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity Calculation
    ↓
Ranking & Scoring
    ↓
Top N Recommendations
```

### Algorithm Details

1. **Data Ingestion**: Load internship database and split into training/test sets
2. **Data Transformation**:
   - Combine relevant columns into single "Content" field
   - Clean text (lowercase, remove special characters)
   - Apply NLP preprocessing using Spacy
   - Vectorize using TF-IDF (Term Frequency-Inverse Document Frequency)

3. **Model Training**:
   - Create TF-IDF feature matrix from training data
   - Calculate cosine similarity matrix between all internships

4. **Prediction**:
   - Vectorize user input using trained TF-IDF model
   - Compute cosine similarity with all internship positions
   - Return top-N matches with similarity scores

## 🔌 API Endpoints

### `GET /`
Returns the main web interface

**Response**: HTML page with input form

### `POST /recommend`
Processes user input and returns internship recommendations

**Request**:
```
Form Data:
- user_input (textarea): User's skills and profile text
- resume_file (file, optional): PDF resume file
```

**Response**:
```json
{
  "results": [
    {
      "position": "ML Engineer Internship",
      "skills": "Python, TensorFlow, Machine Learning...",
      "score": "85%"
    },
    ...
  ],
  "user_text": "Submitted profile text"
}
```

## 📊 Dataset

The system uses a CSV-based internship dataset containing:
- **positions**: Job title/position
- **skills**: Required skills and competencies
- **Other metadata**: Company, location, duration, etc.

### Dataset Format
```csv
positions,skills,...
ML Engineer Internship,"Python, TensorFlow, Deep Learning, ...",
Data Analyst,"SQL, Python, Excel, Tableau, ...",
...
```

Training/test split: 80/20

## ⚙️ Configuration

### App Configuration (app.py)
```python
DATA_PATH = "artifacts/train.csv"      # Dataset path
Flask app port: 8080                   # Default port
DEBUG = True                           # Development mode
```

### Logging
Logs are stored in `logs/` directory with timestamps
Format: `[YYYY-MM-DD HH:MM:SS] LINE_NUMBER - MESSAGE`

## 🔧 Development

### Running with Gunicorn (Production)
```bash
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### Training on New Data
```bash
# Update Dataset/data.csv with new internship data
python src/pipeline/trainpipeline.py
```

### Extending the System

1. **Add New Features**: Modify `datatransformation.py`
2. **Improve Matching**: Update `modeltrainer.py`
3. **Custom Preprocessing**: Extend `utils.py`

## 📈 Future Enhancements

- [ ] User authentication and profile management
- [ ] Personalized recommendation history
- [ ] Advanced filtering (location, duration, stipend)
- [ ] Candidate feedback and model improvement
- [ ] Integration with job portals (LinkedIn, Internshala)
- [ ] Multi-language support
- [ ] Advanced NLP models (BERT, GPT-based)
- [ ] Skills gap analysis and recommendations
- [ ] Real-time dataset updates

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**Khushal Grover**
- Email: khushalgrover64@gmail.com
- Project: Submission for Smart India Hackathon (SIH) 2026

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Smart India Hackathon (SIH) 2026 organizers
- Thapar University
- Open-source community for libraries and frameworks

## 📞 Support & Contact

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: khushalgrover64@gmail.com

---

**Note**: This is a submission for Smart India Hackathon 2026. The system uses cosine similarity for matching and can be enhanced with advanced ML models for production use.
