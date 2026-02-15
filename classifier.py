"""
Sports vs Politics Text Classifier
Author: AI Classification System
Date: February 2026

This script implements a comprehensive text classification system to distinguish
between sports and politics documents using multiple ML techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import pickle
import json
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Get current directory for cross-platform compatibility
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class SportsVsPoliticsClassifier:
    """Main classifier class for Sports vs Politics text classification"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def create_sample_dataset(self):
        """Create a sample dataset with sports and politics texts"""
        
        sports_texts = [
            "The Lakers won the championship game last night with an incredible performance by their star player.",
            "Manchester United signed a new striker for a record transfer fee this summer.",
            "The tennis champion served 15 aces during the final match at Wimbledon.",
            "The quarterback threw three touchdown passes in the fourth quarter to win the game.",
            "Olympic athletes are training intensively for the upcoming summer games in Paris.",
            "The baseball team hit a grand slam in the bottom of the ninth inning.",
            "The soccer match ended in a penalty shootout after extra time.",
            "The basketball player scored 50 points breaking the franchise record.",
            "The formula one driver won the race despite starting from the back of the grid.",
            "The hockey team advanced to the playoffs after a thrilling overtime victory.",
            "The golf tournament was decided by a single stroke on the final hole.",
            "The marathon runner set a new world record finishing in under two hours.",
            "The cricket team won the test series after a dominant bowling performance.",
            "The NFL draft saw several quarterbacks selected in the first round.",
            "The swimming champion broke three world records at the world championships.",
            "The boxing match ended with a knockout in the third round.",
            "The volleyball team won the gold medal at the international tournament.",
            "The track and field athlete won multiple medals at the Olympics.",
            "The rugby team scored a try in the final minutes to secure victory.",
            "The badminton player won the championship without dropping a single set.",
            "The football coach was fired after a series of disappointing losses.",
            "The basketball dynasty won their fifth championship in seven years.",
            "The baseball pitcher threw a perfect game striking out 15 batters.",
            "The soccer league announced new rules to improve player safety.",
            "The tennis star retired from professional competition after a legendary career.",
            "The cricket captain led his team to an unexpected victory in the final.",
            "The gymnastics team performed flawlessly to win the team competition.",
            "The cycling race was won by a breakaway group in the mountain stage.",
            "The hockey goalie made 45 saves in the championship game.",
            "The figure skater landed a quadruple jump to win the gold medal.",
        ]
        
        politics_texts = [
            "The president announced a new policy initiative addressing climate change during the press conference.",
            "Congress passed the infrastructure bill after months of heated debate and negotiation.",
            "The senator delivered a speech criticizing the administration's foreign policy decisions.",
            "Voters went to the polls today to elect representatives in the midterm elections.",
            "The prime minister met with world leaders at the international summit to discuss trade agreements.",
            "The supreme court ruled on a landmark case affecting constitutional rights.",
            "The governor signed legislation aimed at reforming the state's healthcare system.",
            "Political analysts predict a close race in the upcoming presidential election.",
            "The parliament voted to approve the controversial budget proposal.",
            "The mayor announced plans to increase funding for public education.",
            "The candidate launched their campaign promising economic reform and job creation.",
            "The legislative session ended with several bills still pending in committee.",
            "The ambassador addressed concerns about diplomatic relations between the two nations.",
            "The political party held its national convention to nominate candidates.",
            "The cabinet minister resigned amid controversy over policy implementation.",
            "The referendum results showed strong public support for the constitutional amendment.",
            "The opposition leader challenged the government's handling of the economic crisis.",
            "The electoral commission certified the results of the disputed election.",
            "The congressman introduced a bill to reform immigration policy.",
            "The political debate focused on healthcare, education, and economic policy.",
            "The senate hearing examined allegations of corruption in government agencies.",
            "The president vetoed the bill citing concerns about fiscal responsibility.",
            "The coalition government collapsed after losing a vote of no confidence.",
            "The federal court ruled the executive order unconstitutional.",
            "The secretary of state traveled abroad for diplomatic negotiations.",
            "The political scandal led to calls for impeachment proceedings.",
            "The legislative committee held hearings on proposed tax reforms.",
            "The gubernatorial race attracted national attention and significant campaign spending.",
            "The parliamentary system underwent reforms to increase transparency.",
            "The international treaty was ratified by the senate after lengthy deliberation.",
        ]
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': sports_texts + politics_texts,
            'label': ['sports'] * len(sports_texts) + ['politics'] * len(politics_texts)
        })
        
        return df
    
    def preprocess_data(self, df):
        """Split data into training and testing sets"""
        X = df['text']
        y = df['label']
        
        # 80-20 train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        print(f"Class distribution - Training: {self.y_train.value_counts().to_dict()}")
        print(f"Class distribution - Testing: {self.y_test.value_counts().to_dict()}")
        
    def create_features(self):
        """Create TF-IDF and Bag of Words features"""
        
        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
        X_train_tfidf = tfidf.fit_transform(self.X_train)
        X_test_tfidf = tfidf.transform(self.X_test)
        self.vectorizers['tfidf'] = tfidf
        
        # Count Vectorizer (Bag of Words)
        count_vec = CountVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
        X_train_count = count_vec.fit_transform(self.X_train)
        X_test_count = count_vec.transform(self.X_test)
        self.vectorizers['count'] = count_vec
        
        return {
            'tfidf': (X_train_tfidf, X_test_tfidf),
            'count': (X_train_count, X_test_count)
        }
    
    def train_models(self, features):
        """Train multiple ML models"""
        
        models_to_train = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
            'Linear SVM': LinearSVC(random_state=RANDOM_SEED, max_iter=1000),
        }
        
        for feature_type, (X_train_feat, X_test_feat) in features.items():
            print(f"\n{'='*60}")
            print(f"Training models with {feature_type.upper()} features")
            print(f"{'='*60}")
            
            for model_name, model in models_to_train.items():
                print(f"\nTraining {model_name}...")
                
                # Train model
                model.fit(X_train_feat, self.y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_feat)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, pos_label='sports')
                recall = recall_score(self.y_test, y_pred, pos_label='sports')
                f1 = f1_score(self.y_test, y_pred, pos_label='sports')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_feat, self.y_train, cv=5)
                
                # Store results
                key = f"{model_name}_{feature_type}"
                self.models[key] = model
                self.results[key] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred)
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def generate_visualizations(self):
        """Generate comparison charts and visualizations"""
        
        # Prepare data for visualization
        results_df = pd.DataFrame([
            {
                'Model': key.split('_')[0] + ' ' + key.split('_')[1],
                'Feature': key.split('_')[-1].upper(),
                'Accuracy': vals['accuracy'],
                'Precision': vals['precision'],
                'Recall': vals['recall'],
                'F1-Score': vals['f1_score']
            }
            for key, vals in self.results.items()
        ])
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create grouped bar chart
            pivot_data = results_df.pivot(index='Model', columns='Feature', values=metric)
            pivot_data.plot(kind='bar', ax=ax, rot=45, width=0.8)
            
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.legend(title='Feature Type')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("\n✓ Model comparison chart saved")
        
        # Confusion matrices
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        for idx, (key, vals) in enumerate(self.results.items()):
            ax = axes[idx // 3, idx % 3]
            
            cm = vals['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Politics', 'Sports'],
                       yticklabels=['Politics', 'Sports'])
            
            model_name = key.replace('_', ' ').title()
            ax.set_title(f"{model_name}", fontsize=10)
            ax.set_ylabel('Actual', fontsize=9)
            ax.set_xlabel('Predicted', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices saved")
        
    def save_models(self):
        """Save trained models and vectorizers"""
        
        # Save best model based on F1-score
        best_model_key = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_model = self.models[best_model_key]
        
        model_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save vectorizers
        vec_path = os.path.join(OUTPUT_DIR, 'vectorizers.pkl')
        with open(vec_path, 'wb') as f:
            pickle.dump(self.vectorizers, f)
        
        print(f"\n✓ Best model saved: {best_model_key}")
        
    def generate_report_data(self):
        """Generate comprehensive report data"""
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_samples': len(self.X_train) + len(self.X_test),
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'classes': ['sports', 'politics']
            },
            'models': []
        }
        
        for key, vals in self.results.items():
            model_info = {
                'name': key,
                'accuracy': float(vals['accuracy']),
                'precision': float(vals['precision']),
                'recall': float(vals['recall']),
                'f1_score': float(vals['f1_score']),
                'cv_mean': float(vals['cv_mean']),
                'cv_std': float(vals['cv_std'])
            }
            report['models'].append(model_info)
        
        # Save report
        report_path = os.path.join(OUTPUT_DIR, 'report_data.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("✓ Report data saved")
        
        return report


def main():
    """Main execution function"""
    
    print("="*70)
    print("SPORTS VS POLITICS TEXT CLASSIFIER")
    print("="*70)
    
    # Initialize classifier
    classifier = SportsVsPoliticsClassifier()
    
    # Create dataset
    print("\n1. Creating sample dataset...")
    df = classifier.create_sample_dataset()
    print(f"   Dataset created with {len(df)} samples")
    
    # Preprocess
    print("\n2. Preprocessing data...")
    classifier.preprocess_data(df)
    
    # Feature extraction
    print("\n3. Extracting features...")
    features = classifier.create_features()
    print("   Features extracted using TF-IDF and Count Vectorizer")
    
    # Train models
    print("\n4. Training models...")
    classifier.train_models(features)
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    classifier.generate_visualizations()
    
    # Save models
    print("\n6. Saving models...")
    classifier.save_models()
    
    # Generate report
    print("\n7. Generating report data...")
    report = classifier.generate_report_data()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Print summary
    print("\nBest performing models:")
    sorted_results = sorted(classifier.results.items(), 
                          key=lambda x: x[1]['f1_score'], 
                          reverse=True)[:3]
    
    for idx, (model_name, results) in enumerate(sorted_results, 1):
        print(f"\n{idx}. {model_name}")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   F1-Score: {results['f1_score']:.4f}")


if __name__ == "__main__":
    main()
