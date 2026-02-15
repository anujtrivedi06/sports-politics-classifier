# Sports vs Politics Text Classifier

A machine learning project that classifies text documents as either Sports or Politics using multiple classification algorithms and feature extraction techniques.

## 📋 Overview

This project implements a text classification system that can distinguish between sports-related and politics-related documents. It compares multiple machine learning algorithms and feature representation methods to identify the best performing approach.

## 🎯 Problem Statement

Design a classifier that reads a text document and classifies it as Sport or Politics using machine learning techniques. The project explores:
- Multiple ML algorithms (Naive Bayes, Logistic Regression, Linear SVM)
- Different feature representations (TF-IDF, Bag of Words)
- Comparative analysis of model performance
- System limitations and future improvements

## 🚀 Features

- **Multiple ML Models**: Naive Bayes, Logistic Regression, Linear SVM
- **Feature Extraction**: TF-IDF and Count Vectorizer (Bag of Words)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Cross-validation
- **Visualizations**: Performance comparison charts and confusion matrices
- **Production-Ready**: Saved models and vectorizers for deployment

## 📊 Results Summary

### Best Performing Model: Naive Bayes with TF-IDF
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1-Score**: 100%
- **Cross-Validation Score**: 89.33% (±9.47%)

### Model Comparison

| Model | Feature Type | Accuracy | Precision | Recall | F1-Score |
|-------|-------------|----------|-----------|--------|----------|
| Naive Bayes | TF-IDF | 100.0% | 100.0% | 100.0% | 100.0% |
| Naive Bayes | Count | 91.67% | 100.0% | 83.33% | 90.91% |
| Logistic Regression | TF-IDF | 91.67% | 100.0% | 83.33% | 90.91% |
| Logistic Regression | Count | 91.67% | 100.0% | 83.33% | 90.91% |
| Linear SVM | TF-IDF | 91.67% | 100.0% | 83.33% | 90.91% |
| Linear SVM | Count | 91.67% | 100.0% | 83.33% | 90.91% |

## 📁 Project Structure

```
sports_politics_classifier/
│
├── classifier.py              # Main classification script
├── best_model.pkl            # Saved best performing model
├── vectorizers.pkl           # Saved feature vectorizers
├── report_data.json          # Detailed results in JSON format
├── model_comparison.png      # Performance comparison charts
├── confusion_matrices.png    # Confusion matrices for all models
└── README.md                 # This file
```

## 🛠️ Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/sports-politics-classifier.git
cd sports-politics-classifier

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training the Models

```bash
python classifier.py
```

This will:
1. Create/load the dataset
2. Preprocess the data
3. Extract features using TF-IDF and Count Vectorizer
4. Train all models
5. Generate performance visualizations
6. Save the best model

### Using the Classifier

```python
import pickle

# Load the saved model and vectorizer
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizers.pkl', 'rb') as f:
    vectorizers = pickle.load(f)

# Classify new text
text = ["The basketball team won the championship game"]
features = vectorizers['tfidf'].transform(text)
prediction = model.predict(features)
print(f"Prediction: {prediction[0]}")
```

## 📈 Dataset

The current implementation uses a sample dataset with:
- **Total Samples**: 60 (30 sports, 30 politics)
- **Training Set**: 48 samples (80%)
- **Testing Set**: 12 samples (20%)

### Dataset Characteristics:
- Balanced classes (equal sports and politics examples)
- Real-world document structure
- Diverse vocabulary covering various sports and political topics

## 🔬 Methodology

### 1. Data Collection
Sample dataset created with representative texts from both categories.

### 2. Preprocessing
- Train-test split (80-20)
- Stratified sampling to maintain class balance

### 3. Feature Extraction
- **TF-IDF**: Captures term importance with n-grams (1-2)
- **Bag of Words**: Simple count-based representation with n-grams (1-2)

### 4. Model Training
Three algorithms tested:
- **Naive Bayes**: Probabilistic classifier
- **Logistic Regression**: Linear classification
- **Linear SVM**: Maximum margin classifier

### 5. Evaluation
- 5-fold cross-validation
- Multiple metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis

## ⚠️ Limitations

1. **Small Dataset**: Current implementation uses 60 samples
2. **Domain Coverage**: Limited to general sports and politics topics
3. **Language**: English-only support
4. **Ambiguity**: May struggle with documents covering both topics
5. **Context Dependency**: No deep semantic understanding

## 🔮 Future Improvements

1. **Larger Dataset**: Collect 10,000+ documents from news sources
2. **Deep Learning**: Implement BERT, RoBERTa for better understanding
3. **Multi-class**: Extend to more categories (economy, technology, etc.)
4. **Real-time API**: Deploy as web service
5. **Active Learning**: Continuous improvement with user feedback
6. **Multilingual Support**: Extend to other languages

## 📝 License

This project is open source and available under the MIT License.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating text classification techniques. For production use, consider using larger datasets and more sophisticated models.
