import joblib
import pandas as pd
from sklearn.metrics import classification_report
from .data_processor import DataProcessor
from .model import get_base_models, EnsembleModel

def train_model(train_data_path):
    # Load and process data
    df = pd.read_csv(train_data_path)
    processor = DataProcessor()
    df = processor.process_features(df)
    
    X = df.drop('Credit_Score', axis=1)
    y = df['Credit_Score']
    
    # Initialize and train model
    base_models = get_base_models()
    ensemble = EnsembleModel(base_models)
    print("Training ensemble model...")
    ensemble.train(X, y)
    
    # Evaluate
    y_pred = ensemble.predict(X) 
    print("\nTraining Results:")
    print(classification_report(y, y_pred))
    
    # Save artifacts
    model_artifacts = {
        'processor': processor,
        'models': ensemble.models,
        'weights': ensemble.weights
    }
    joblib.dump(model_artifacts, 'credit_score_ensemble.joblib')
    print("\nModel artifacts saved successfully!")