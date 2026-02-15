"""
Demo Script: Using the Sports vs Politics Classifier
This script demonstrates how to use the trained model to classify new documents.
"""

import pickle
import sys
import os

# Get current directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    """Load the trained model and vectorizers"""
    try:
        model_path = os.path.join(SCRIPT_DIR, 'best_model.pkl')
        vec_path = os.path.join(SCRIPT_DIR, 'vectorizers.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vec_path, 'rb') as f:
            vectorizers = pickle.load(f)
        
        return model, vectorizers
    except FileNotFoundError as e:
        print(f"Error: Model files not found in {SCRIPT_DIR}")
        print(f"Missing file: {e.filename}")
        print("\nPlease run classifier.py first to train the model.")
        print("Make sure both files exist:")
        print(f"  - {os.path.join(SCRIPT_DIR, 'best_model.pkl')}")
        print(f"  - {os.path.join(SCRIPT_DIR, 'vectorizers.pkl')}")
        sys.exit(1)

def classify_text(text, model, vectorizer):
    """Classify a single text document"""
    # Transform text using TF-IDF vectorizer
    features = vectorizer.transform([text])
    
    # Get prediction
    prediction = model.predict(features)
    
    # Get probability scores if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)
        confidence = max(probabilities[0])
        return prediction[0], confidence
    
    return prediction[0], None

def main():
    """Main demo function"""
    print("="*70)
    print("SPORTS VS POLITICS CLASSIFIER - DEMO")
    print("="*70)
    
    # Load model
    print("\nLoading trained model...")
    model, vectorizers = load_model()
    tfidf_vectorizer = vectorizers['tfidf']
    print("✓ Model loaded successfully\n")
    
    # Test examples
    test_documents = [
        "The basketball team won the championship after a thrilling overtime game.",
        "The president signed the new healthcare bill into law yesterday.",
        "The striker scored a hat trick in the football match against their rivals.",
        "Congress debated the infrastructure spending proposal for several hours.",
        "The tennis star retired from professional play after winning twenty grand slams.",
        "The governor announced a new policy to address climate change concerns.",
        "The baseball pitcher threw a no-hitter in the playoff game.",
        "The senate will vote on the amendment to the constitution next week.",
    ]
    
    print("Classifying sample documents:\n")
    print("-" * 70)
    
    for i, doc in enumerate(test_documents, 1):
        prediction, confidence = classify_text(doc, model, tfidf_vectorizer)
        
        # Truncate document for display
        display_doc = doc if len(doc) <= 60 else doc[:57] + "..."
        
        print(f"\n{i}. Document: \"{display_doc}\"")
        print(f"   Prediction: {prediction.upper()}")
        if confidence:
            print(f"   Confidence: {confidence*100:.2f}%")
    
    print("\n" + "-" * 70)
    
    # Interactive mode
    print("\n\nInteractive Mode - Enter your own text to classify")
    print("(Type 'quit' to exit)\n")
    
    while True:
        user_input = input("Enter text to classify: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nThank you for using the classifier!")
            break
        
        if not user_input:
            print("Please enter some text.\n")
            continue
        
        prediction, confidence = classify_text(user_input, model, tfidf_vectorizer)
        
        print(f"\n  → Prediction: {prediction.upper()}")
        if confidence:
            print(f"  → Confidence: {confidence*100:.2f}%\n")
        else:
            print()

if __name__ == "__main__":
    main()
