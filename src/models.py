# src/models.py
"""
Model Training and Evaluation Module
Trains Random Forest, XGBoost, and LightGBM models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Train and compare multiple ML models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def prepare_data(self, X, y, test_size=0.2):
        """
        Split and scale data
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest with hyperparameter tuning
        """
        print("Training Random Forest...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'class_weight': ['balanced']
        }
        
        rf = RandomForestClassifier(random_state=self.random_state)
        
        # Use GridSearchCV for tuning
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, n_jobs=-1, scoring='roc_auc'
        )
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_rf.predict(X_test)
        y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'best_params': grid_search.best_params_
        }
        
        self.models['RandomForest'] = best_rf
        self.results['RandomForest'] = metrics
        
        print(f"RF ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return best_rf, metrics
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost
        """
        print("Training XGBoost...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = metrics
        
        print(f"XGBoost ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return xgb_model, metrics
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """
        Train LightGBM
        """
        print("Training LightGBM...")
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=31,
            random_state=self.random_state,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = lgb_model.predict(X_test)
        y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        self.models['LightGBM'] = lgb_model
        self.results['LightGBM'] = metrics
        
        print(f"LightGBM ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return lgb_model, metrics
    
    def get_best_model(self):
        """
        Return model with highest ROC-AUC
        """
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        self.best_model = self.models[best_name]
        
        print(f"\nBest Model: {best_name}")
        print(f"ROC-AUC: {self.results[best_name]['roc_auc']:.4f}")
        
        return self.best_model, best_name
    
    def save_model(self, model, path):
        """
        Save model to disk
        """
        joblib.dump(model, path)
        print(f"Model saved to {path}")
    
    def save_scaler(self, path):
        """
        Save scaler to disk
        """
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")
    
    def get_results_dataframe(self):
        """
        Return comparison table
        """
        return pd.DataFrame(self.results).T


# Example usage function
def train_all_models(X, y):
    """
    Train all models and return best one
    """
    trainer = ModelTrainer()
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    trainer.train_xgboost(X_train, y_train, X_test, y_test)
    trainer.train_lightgbm(X_train, y_train, X_test, y_test)
    
    best_model, best_name = trainer.get_best_model()
    results_df = trainer.get_results_dataframe()
    
    return trainer, best_model, best_name, results_df
