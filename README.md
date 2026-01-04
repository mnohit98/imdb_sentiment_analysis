# IMDb Movie Review Sentiment Analysis Project

## Project Overview
This project aims to build a machine learning model that predicts the sentiment (positive or negative) of IMDb movie reviews using Natural Language Processing (NLP) techniques. The project includes comprehensive text preprocessing, feature engineering, model development, and evaluation.

## Project Structure
```
imdb_sentiment_analysis/
├── 01_Data_Exploration_Preprocessing.ipynb  # Data loading, EDA, and text preprocessing
├── 02_Feature_Engineering.ipynb              # Feature extraction (TF-IDF, Word2Vec, etc.)
├── 03_Model_Development.ipynb                 # Model training (LR, NB, SVM, RF, Neural Networks)
├── 04_Model_Evaluation.ipynb                   # Model evaluation and visualization
├── 05_Predict_New_Reviews.ipynb               # Prediction on new reviews
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── DATASET_INSTRUCTIONS.md                     # Dataset download guide
├── data/                                       # Dataset directory
│   └── imdb_data.csv                          # Input dataset (to be added)
└── models/                                     # Saved models directory
    ├── vectorizer.pkl
    ├── models (various .pkl files)
    └── visualizations
```

## Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Dataset

The dataset should be placed in the `data/` directory as `imdb_data.csv`. 

**Dataset Source:** [Google Sheets Link](https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/edit?gid=1889101679#gid=1889101679)

**To download the dataset:**
1. Open the Google Sheets link
2. Go to File → Download → Comma Separated Values (.csv)
3. Save as `imdb_data.csv` in the `data/` folder

## Usage

### Step 1: Data Exploration and Preprocessing
Run `01_Data_Exploration_Preprocessing.ipynb` to:
- Load and explore the dataset
- Analyze review length, sentiment distribution
- Perform text cleaning (remove HTML tags, special characters)
- Tokenization, stop word removal
- Lemmatization and stemming
- Save preprocessed data

### Step 2: Feature Engineering
Run `02_Feature_Engineering.ipynb` to:
- Extract textual features (word count, character count, etc.)
- Create TF-IDF vectors
- Generate Word2Vec embeddings (optional)
- Prepare features for modeling

### Step 3: Model Development
Run `03_Model_Development.ipynb` to:
- Split data into training and testing sets
- Train multiple models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - Neural Networks (LSTM/BERT - optional)
- Hyperparameter tuning
- Save trained models

### Step 4: Model Evaluation
Run `04_Model_Evaluation.ipynb` to:
- Evaluate model performance with multiple metrics
- Visualize confusion matrices
- Plot ROC curves and Precision-Recall curves
- Analyze feature importance
- Generate word clouds
- Generate insights and recommendations

### Step 5: Predict New Reviews
Run `05_Predict_New_Reviews.ipynb` to:
- Load trained model and vectorizer
- Preprocess new review text
- Make sentiment predictions
- Get prediction probabilities

## Features

The dataset contains:
- **review**: Text of the movie review
- **sentiment**: Label (positive/negative)

## Models Implemented

1. **Logistic Regression**: Baseline linear model
2. **Naive Bayes**: Probabilistic classifier (good for text)
3. **Support Vector Machine (SVM)**: Effective for high-dimensional text data
4. **Random Forest**: Ensemble tree-based model
5. **Neural Networks**: LSTM/Deep Learning models (optional)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## Expected Results

- Model accuracy: 80-90%
- F1-Score: 0.80-0.90
- ROC-AUC: 0.85-0.95
- Key insights on words/phrases affecting sentiment

## Key Insights

Based on the analysis, common factors affecting sentiment include:
- Review length (longer reviews may be more detailed)
- Specific words and phrases
- Review structure and tone
- Use of positive/negative language patterns

## Business Applications

1. **Movie Producers**: Understand audience reactions
2. **Streaming Platforms**: Recommend content based on sentiment
3. **Marketing Teams**: Identify what resonates with audiences
4. **Content Creators**: Learn from audience feedback

## Author
Student Project - IMDb Sentiment Analysis

## License
This project is for educational purposes.

## Notes
- Make sure to run notebooks in sequence (01 → 02 → 03 → 04 → 05)
- Models and vectorizers are saved automatically after training
- Preprocessed data is saved for reuse between notebooks
- Text preprocessing may take time for large datasets
