import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize

def get_base_models():
    models = {}
    
    # LightGBM params
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
    
    # XGBoost params
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
                model.fit(X_train, y_train)
                self.trained_models[name].append(model)
                val_pred = model.predict_proba(X_val)
                val_predictions[val_idx, i] = val_pred
                
        self.optimize_weights(val_predictions, y_resampled)
        
        # Final training on full dataset
        for name, model in self.models.items():
            model.fit(X_resampled, y_resampled)
            
    def optimize_weights(self, predictions, y_true):
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