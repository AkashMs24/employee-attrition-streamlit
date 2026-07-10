# src/explainability.py
"""
Explainability Module
SHAP values, LIME, feature importance, and prediction explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    Use SHAP (SHapley Additive exPlanations) for individual prediction explanations
    
    SHAP values show:
    - How much each feature contributed to the prediction
    - Direction (positive/negative impact)
    - Individual level explanations
    """
    
    def __init__(self, model, background_data=None):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """
        Initialize SHAP explainer
        """
        try:
            # Use TreeExplainer for tree-based models (Random Forest, XGBoost, LightGBM)
            self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fallback to KernelExplainer for other models
            if self.background_data is not None:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    self.background_data
                )
            else:
                self.explainer = None
    
    def explain_prediction(self, features, feature_names=None):
        """
        Explain individual prediction using SHAP values
        
        Args:
            features: Array of shape (1, n_features)
            feature_names: List of feature names
        
        Returns:
            Dictionary with SHAP explanation
        """
        if self.explainer is None:
            return None
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Attrition class
        
        # Base value (average model prediction)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(features[0]))]
        
        # Create explanation dataframe
        explanation = pd.DataFrame({
            'feature': feature_names,
            'value': features[0],
            'shap_value': shap_values[0]
        })
        
        # Sort by absolute SHAP value
        explanation['abs_shap'] = np.abs(explanation['shap_value'])
        explanation = explanation.sort_values('abs_shap', ascending=False)
        
        return {
            'base_prediction': base_value,
            'feature_contributions': explanation,
            'positive_factors': explanation[explanation['shap_value'] > 0],
            'negative_factors': explanation[explanation['shap_value'] < 0],
            'top_positive_contributor': (
                explanation[explanation['shap_value'] > 0].iloc[0] if len(explanation[explanation['shap_value'] > 0]) > 0 else None
            ),
            'top_negative_contributor': (
                explanation[explanation['shap_value'] < 0].iloc[0] if len(explanation[explanation['shap_value'] < 0]) > 0 else None
            )
        }
    
    def create_explanation_text(self, explanation, prediction_probability):
        """
        Create human-readable explanation of prediction
        """
        if explanation is None:
            return "Explanation not available"
        
        text = f"Model predicts {prediction_probability:.1%} attrition risk\n\n"
        
        text += "KEY FACTORS INCREASING RISK:\n"
        for _, row in explanation['positive_factors'].head(3).iterrows():
            text += f"  • {row['feature']}: {row['value']:.0f} (impact: +{row['shap_value']:.2%})\n"
        
        text += "\nKEY FACTORS DECREASING RISK:\n"
        for _, row in explanation['negative_factors'].head(3).iterrows():
            text += f"  • {row['feature']}: {row['value']:.0f} (impact: {row['shap_value']:.2%})\n"
        
        return text
    
    def plot_force_plot(self, features, feature_names=None):
        """
        Create SHAP force plot for visualization
        """
        if self.explainer is None:
            return None
        
        shap_values = self.explainer.shap_values(features)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        try:
            fig = shap.force_plot(
                base_value, shap_values[0], features[0],
                feature_names=feature_names,
                matplotlib=True
            )
            return fig
        except:
            return None


class FeatureImportanceExplainer:
    """
    Explain using traditional feature importance methods
    """
    
    def __init__(self, model):
        self.model = model
    
    def get_model_feature_importance(self, feature_names=None):
        """
        Get built-in feature importance from model
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_permutation_importance(self, X, y, feature_names=None):
        """
        Calculate permutation importance
        """
        perm_importance = permutation_importance(
            self.model, X, y, n_repeats=10, random_state=42
        )
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(perm_importance.importances_mean))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_importance(self, importance_df, top_n=10):
        """
        Plot feature importance
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_features = importance_df.head(top_n)
        
        ax.barh(
            top_features['feature'],
            top_features['importance'],
            color='#4CAF50'
        )
        
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Feature Importance')
        
        return fig


class PDPExplainer:
    """
    Partial Dependence Plots
    Show marginal effect of each feature on predictions
    """
    
    def __init__(self, model):
        self.model = model
    
    def create_pdp(self, X, feature_idx, feature_name=None, num_points=20):
        """
        Create partial dependence plot for one feature
        """
        if feature_name is None:
            feature_name = f'Feature {feature_idx}'
        
        # Get range of feature values
        feature_values = np.linspace(
            X.iloc[:, feature_idx].min(),
            X.iloc[:, feature_idx].max(),
            num_points
        )
        
        predictions = []
        
        for value in feature_values:
            X_modified = X.copy()
            X_modified.iloc[:, feature_idx] = value
            
            pred = self.model.predict_proba(X_modified)[:, 1].mean()
            predictions.append(pred)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(feature_values, predictions, marker='o', linewidth=2)
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Average Predicted Probability')
        ax.set_title(f'Partial Dependence: {feature_name}')
        ax.grid(True, alpha=0.3)
        
        return fig


class CounterfactualExplainer:
    """
    Generate counterfactual explanations
    "What changes would flip the prediction?"
    """
    
    def __init__(self, model):
        self.model = model
    
    def find_counterfactual(self, features, feature_names, target_probability=0.3):
        """
        Find minimum changes to flip prediction from high to low risk
        
        Args:
            features: Current feature values
            feature_names: Feature names
            target_probability: Desired risk level
        
        Returns:
            Counterfactual changes
        """
        current_prob = self.model.predict_proba([features])[0][1]
        
        if current_prob < target_probability:
            return {
                'message': 'Already at target risk level',
                'current_risk': current_prob,
                'target_risk': target_probability
            }
        
        counterfactuals = []
        
        # Try modifying each feature
        for i, (feature, value) in enumerate(zip(feature_names, features)):
            # Income - try increasing
            if 'Income' in feature or 'income' in feature.lower():
                for pct in [5, 10, 15, 20]:
                    modified = features.copy()
                    modified[i] = value * (1 + pct / 100)
                    
                    new_prob = self.model.predict_proba([modified])[0][1]
                    
                    if new_prob < target_probability:
                        counterfactuals.append({
                            'feature': feature,
                            'current_value': value,
                            'suggested_value': modified[i],
                            'change': f'+{pct}%',
                            'new_risk': new_prob,
                            'explanation': f'Increase {feature} by {pct}%'
                        })
                        break
            
            # Overtime - try removing
            if 'Over' in feature or 'ot' in feature.lower():
                modified = features.copy()
                modified[i] = 0
                
                new_prob = self.model.predict_proba([modified])[0][1]
                
                if new_prob < target_probability:
                    counterfactuals.append({
                        'feature': feature,
                        'current_value': value,
                        'suggested_value': 0,
                        'change': 'Remove',
                        'new_risk': new_prob,
                        'explanation': f'Eliminate {feature}'
                    })
        
        return {
            'current_risk': current_prob,
            'target_risk': target_probability,
            'counterfactuals': sorted(counterfactuals, key=lambda x: x['new_risk']),
            'best_action': counterfactuals[0] if counterfactuals else None
        }


class ModelExplanationReport:
    """
    Generate comprehensive explanation report
    """
    
    @staticmethod
    def create_report(model, X_test, y_test, features_instance, feature_names):
        """
        Create full explanation report
        """
        report = {
            'timestamp': pd.Timestamp.now(),
            'sections': {}
        }
        
        # 1. Global explanation (feature importance)
        fi_explainer = FeatureImportanceExplainer(model)
        fi_df = fi_explainer.get_model_feature_importance(feature_names)
        report['sections']['feature_importance'] = fi_df
        
        # 2. Individual explanation (SHAP)
        shap_explainer = SHAPExplainer(model, X_test.values)
        shap_explanation = shap_explainer.explain_prediction(
            features_instance.reshape(1, -1),
            feature_names
        )
        report['sections']['individual_explanation'] = shap_explanation
        
        # 3. Counterfactual
        cf_explainer = CounterfactualExplainer(model)
        counterfactual = cf_explainer.find_counterfactual(
            features_instance, feature_names
        )
        report['sections']['counterfactual'] = counterfactual
        
        return report


# Example usage
def explain_single_prediction(model, features, feature_names, prediction_prob):
    """
    Helper to explain single prediction
    """
    explainer = SHAPExplainer(model)
    explanation = explainer.explain_prediction(
        features.reshape(1, -1),
        feature_names
    )
    
    text_explanation = explainer.create_explanation_text(explanation, prediction_prob)
    
    return {
        'explanation': explanation,
        'text': text_explanation
    }
