import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.le = LabelEncoder()
        
    def process_features(self, df):
        # Numerical features
        num_cols = ['Annual_Income', 'Num_Bank_Accounts',
                   'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
                   'Delay_from_due_date', 'Num_of_Delayed_Payment',
                   'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                   'Total_EMI_per_month', 'Amount_invested_monthly',
                   'Monthly_Balance']
                   
        # Categorical features
        cat_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']
        
        # Scale numerical features
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        
        # Encode categorical features
        for col in cat_cols:
            df[col] = self.le.fit_transform(df[col])
            
        return df
    
    

df = pd.read_csv(r"C:\Users\HP\Downloads\Industry-Baby\datasets\Creditscore_train_cleaned.csv")
processor = DataProcessor()
df = processor.process_features(df)
df = df.select_dtypes(exclude='object')
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

joblib.dump(processor, 'credit_score_processor.joblib')

def get_base_models():
    models = {}
    
    # LightGBM
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'max_depth': 7,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # XGBoost
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    models['lgb'] = lgb.LGBMClassifier(**lgb_params)
    models['xgb'] = xgb.XGBClassifier(**xgb_params)
    
    return models



class EnsembleModel:
    def __init__(self, models, n_splits=5):
        self.models = models
        self.n_splits = n_splits
        self.trained_models = {name: [] for name in models.keys()}
        self.weights = None
        
    def train(self, X, y):
        X_resampled, y_resampled = X, y
        
        # Cross validation predictions
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        val_predictions = np.zeros((len(X_resampled), len(self.models), 3))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled)):
            X_train, X_val = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
            y_train, y_val = y_resampled.iloc[train_idx], y_resampled.iloc[val_idx]
            
            for i, (name, model) in enumerate(self.models.items()):
                # Train model
                model.fit(X_train, y_train)
                self.trained_models[name].append(model)
                
                # Get validation predictions
                val_pred = model.predict_proba(X_val)
                val_predictions[val_idx, i] = val_pred
                
        # Optimize weights using validation predictions
        self.optimize_weights(val_predictions, y_resampled)
        
        # Final training on full dataset
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
    
    
base_models = get_base_models()
ensemble = EnsembleModel(base_models)

# Train
print("Training ensemble model...")
ensemble.train(X, y)

# Evaluate
y_pred = ensemble.predict(X)
print("\nTraining Results:")
print(classification_report(y, y_pred))

# Save models and weights
model_artifacts = {
    'processor': processor,
    'models': ensemble.models,
    'weights': ensemble.weights
}
joblib.dump(model_artifacts, 'credit_score_ensemble.joblib')
print("\nModel artifacts saved successfully!")


def predict_credit_score(data):
    # Load artifacts
    artifacts = joblib.load('credit_score_ensemble.joblib')
    processor = artifacts['processor']
    
    # Process features
    X = processor.process_features(data.copy())
    
    # Initialize ensemble
    ensemble = EnsembleModel(artifacts['models'])
    ensemble.weights = artifacts['weights']
    
    # Generate predictions
    predictions = ensemble.predict(X)
    probabilities = ensemble.predict_proba(X)
    
    return predictions, probabilities

# Test prediction
test_data = pd.read_csv(r"C:\Users\HP\Downloads\Industry-Baby\Creditscore_test_cleaned.csv")
test_data = processor.process_features(test_data)
test_data = test_data.select_dtypes(exclude='object')
preds, probs = predict_credit_score(test_data)

results_df = pd.DataFrame({
    'Predicted_Score': preds,
    'Good_Prob': probs[:,0],
    'Standard_Prob': probs[:,1],
    'Poor_Prob': probs[:,2]
})



import shap
explainer_lgb = shap.TreeExplainer(ensemble.trained_models['lgb'][0])
shap_values_lgb = explainer_lgb(test_data)

explainer_xgb = shap.TreeExplainer(ensemble.trained_models['xgb'][0])
shap_values_xgb = explainer_xgb(test_data)


shap_values_combined = (ensemble.weights[0] * shap_values_lgb + ensemble.weights[1] * shap_values_xgb)
shap.summary_plot([shap_values_combined.values[:, :, i] for i in range(shap_values_combined.shape[2])], test_data)


