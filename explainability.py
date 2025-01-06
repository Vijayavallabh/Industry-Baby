import numpy as np
import pandas as pd
import joblib
import shap
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.le = LabelEncoder()
        
    def process_features(self, df):
        num_cols = ['Annual_Income', 'Num_Bank_Accounts',
                   'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
                   'Delay_from_due_date', 'Num_of_Delayed_Payment',
                   'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                   'Total_EMI_per_month', 'Amount_invested_monthly',
                   'Monthly_Balance']
        
        cat_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']
        
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        
        for col in cat_cols:
            df[col] = self.le.fit_transform(df[col])
            
        return df

class EnsembleModel:
    def __init__(self, models, n_splits=5):
        self.models = models
        self.n_splits = n_splits
        self.trained_models = {name: [] for name in models.keys()}
        self.weights = None
        
    def train(self, X, y):
        X_resampled, y_resampled = X, y
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        val_predictions = np.zeros((len(X_resampled), len(self.models), 3))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled)):
            X_train, X_val = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
            y_train, y_val = y_resampled.iloc[train_idx], y_resampled.iloc[val_idx]
            
            for i, (name, model) in enumerate(self.models.items()):
                model.fit(X_train, y_train)
                self.trained_models[name].append(model)
                val_pred = model.predict_proba(X_val)
                val_predictions[val_idx, i] = val_pred
                
        self.optimize_weights(val_predictions, y_resampled)
        
        for name, model in self.models.items():
            model.fit(X_resampled, y_resampled)
            
    def optimize_weights(self, predictions, y_true):
        from scipy.optimize import minimize
        
        def loss_function(weights):
            weighted_preds = np.sum(predictions * weights.reshape(1, -1, 1), axis=1)
            return -accuracy_score(y_true, weighted_preds.argmax(axis=1))
        
        initial_weights = np.ones(len(self.models)) / len(self.models)
        bounds = [(0, 1)] * len(self.models)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        result = minimize(loss_function, initial_weights, bounds=bounds, constraints=constraints)
        self.weights = result.x
        
    def predict_proba(self, X):
        predictions = np.zeros((len(X), len(self.models), 3))
        
        for i, (name, model) in enumerate(self.models.items()):
            pred = model.predict_proba(X)
            predictions[:, i] = pred
            
        return np.sum(predictions * self.weights.reshape(1, -1, 1), axis=1)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def predict_credit_score(data):
    artifacts = joblib.load('credit_score_ensemble.joblib')
    processor = artifacts['processor']
    
    data = processor.process_features(data.copy())
    
    ensemble = EnsembleModel(artifacts['models'])
    ensemble.weights = artifacts['weights']
    
    predictions = ensemble.predict(data)
    probabilities = ensemble.predict_proba(data)
    
    return predictions, probabilities,data

def generate_shap_values(test_data):
    artifacts = joblib.load('credit_score_ensemble.joblib')
    
    explainer_lgb = shap.TreeExplainer(artifacts['models']['lgb'])
    shap_values_lgb = explainer_lgb(test_data)

    #explainer_xgb = shap.TreeExplainer(artifacts['models']['xgb'][0])
    #shap_values_xgb = explainer_xgb(test_data)

    shap_values_combined = shap_values_lgb#(ensemble.weights[0] * shap_values_lgb + ensemble.weights[1] * shap_values_xgb)
    return shap_values_combined