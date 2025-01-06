import joblib
import pandas as pd
from .model import EnsembleModel

def predict_credit_score(df):
    # Load artifacts
    artifacts = joblib.load('credit_score_ensemble.joblib')
    processor = artifacts['processor']
    
    # Process features
    X = processor.process_features(df.copy())
    
    # Initialize ensemble and predict
    ensemble = EnsembleModel(artifacts['models'])
    ensemble.weights = artifacts['weights']
    
    predictions = ensemble.predict(X)
    probabilities = ensemble.predict_proba(X)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Predicted_Score': predictions,
        'Good_Prob': probabilities[:,0],
        'Standard_Prob': probabilities[:,1],
        'Poor_Prob': probabilities[:,2]
    })
    
    return results_df, predictions, probabilities