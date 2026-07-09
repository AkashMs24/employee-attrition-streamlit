# src/fairness_audit.py
"""
Fairness & Bias Audit Module
Ensure model predictions are fair and unbiased across demographics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


class FairnessAudit:
    """
    Audit model for fairness and bias across demographics
    
    Metrics:
    - False Positive Rate (FPR)
    - False Negative Rate (FNR)
    - Precision parity
    - Equal opportunity
    - Demographic parity
    """
    
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.audit_results = {}
    
    def audit_by_demographic(self, X, y, demographic_df, demographic_column):
        """
        Check fairness across demographic groups
        
        Args:
            X: Feature matrix
            y: True labels
            demographic_df: DataFrame with demographic info
            demographic_column: Column name to audit (e.g., 'Age', 'Gender')
        
        Returns:
            Dictionary with fairness metrics by group
        """
        
        results = {}
        unique_values = demographic_df[demographic_column].unique()
        
        for value in unique_values:
            mask = demographic_df[demographic_column] == value
            X_subset = X[mask]
            y_subset = y[mask]
            
            if len(X_subset) == 0:
                continue
            
            # Get predictions
            predictions_proba = self.model.predict_proba(X_subset)[:, 1]
            predictions = (predictions_proba > self.threshold).astype(int)
            
            # Calculate metrics
            if len(np.unique(y_subset)) == 1:
                # Only one class in subset
                accuracy = 1.0 if np.mean(predictions == y_subset) == 1.0 else 0.0
                precision = 0.0
                recall = 0.0
                fpr = 0.0
                fnr = 0.0
            else:
                tn, fp, fn, tp = confusion_matrix(y_subset, predictions).ravel()
                
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            results[str(value)] = {
                'sample_size': len(X_subset),
                'positive_rate': y_subset.mean(),
                'prediction_rate': predictions.mean(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'predicted_positive': int(np.sum(predictions)),
                'actual_positive': int(np.sum(y_subset))
            }
        
        self.audit_results[demographic_column] = results
        return results
    
    def detect_bias(self, results_dict):
        """
        Detect statistically significant bias
        Uses 80% rule: max/min ratio should not exceed 1.25
        """
        bias_report = []
        
        for demographic_column, results in results_dict.items():
            # Extract metrics
            fpr_values = [v['false_positive_rate'] for v in results.values()]
            fnr_values = [v['false_negative_rate'] for v in results.values()]
            precision_values = [v['precision'] for v in results.values() if v['precision'] > 0]
            recall_values = [v['recall'] for v in results.values() if v['recall'] > 0]
            
            # Check FPR disparity
            if fpr_values and max(fpr_values) > 0:
                fpr_ratio = max(fpr_values) / (min(fpr_values) + 0.001)
                if fpr_ratio > 1.25:
                    bias_report.append({
                        'demographic': demographic_column,
                        'metric': 'False Positive Rate',
                        'issue': 'FPR disparity',
                        'severity': 'HIGH' if fpr_ratio > 1.5 else 'MEDIUM',
                        'ratio': fpr_ratio,
                        'description': f'False positive rate varies {fpr_ratio:.2f}x across {demographic_column}',
                        'impact': 'Some groups unfairly flagged as high-risk'
                    })
            
            # Check FNR disparity
            if fnr_values and max(fnr_values) > 0:
                fnr_ratio = max(fnr_values) / (min(fnr_values) + 0.001)
                if fnr_ratio > 1.25:
                    bias_report.append({
                        'demographic': demographic_column,
                        'metric': 'False Negative Rate',
                        'issue': 'FNR disparity',
                        'severity': 'HIGH' if fnr_ratio > 1.5 else 'MEDIUM',
                        'ratio': fnr_ratio,
                        'description': f'False negative rate varies {fnr_ratio:.2f}x across {demographic_column}',
                        'impact': 'Some groups\' actual attrition missed'
                    })
            
            # Check precision disparity
            if precision_values and min(precision_values) > 0:
                precision_ratio = max(precision_values) / min(precision_values)
                if precision_ratio > 1.25:
                    bias_report.append({
                        'demographic': demographic_column,
                        'metric': 'Precision',
                        'issue': 'Precision disparity',
                        'severity': 'MEDIUM',
                        'ratio': precision_ratio,
                        'description': f'Precision varies {precision_ratio:.2f}x across {demographic_column}',
                        'impact': 'Different reliability across groups'
                    })
        
        return bias_report
    
    def statistical_parity(self, predictions, demographic_df, demographic_column):
        """
        Check if selection rate is similar across groups
        (Demographic Parity: P(Ŷ=1) should be similar for all groups)
        """
        results = {}
        
        for value in demographic_df[demographic_column].unique():
            mask = demographic_df[demographic_column] == value
            selection_rate = predictions[mask].mean()
            results[str(value)] = selection_rate
        
        # Check if rates differ significantly
        max_rate = max(results.values())
        min_rate = min(results.values())
        
        if min_rate > 0:
            disparity_ratio = max_rate / min_rate
            return {
                'metric': 'Demographic Parity',
                'by_group': results,
                'disparity_ratio': disparity_ratio,
                'fair': disparity_ratio < 1.25,
                'interpretation': 'Similar selection rates across groups' if disparity_ratio < 1.25 
                                 else 'Significantly different selection rates'
            }
        
        return None
    
    def equal_opportunity(self, y_true, predictions, demographic_df, demographic_column):
        """
        Check Equal Opportunity: TPR should be similar for all groups
        (True Positive Rate = Sensitivity)
        """
        results = {}
        
        for value in demographic_df[demographic_column].unique():
            mask = demographic_df[demographic_column] == value
            
            y_subset = y_true[mask]
            pred_subset = predictions[mask]
            
            if len(y_subset[y_subset == 1]) > 0:
                tpr = np.sum((pred_subset == 1) & (y_subset == 1)) / np.sum(y_subset == 1)
                results[str(value)] = tpr
        
        if not results:
            return None
        
        max_tpr = max(results.values())
        min_tpr = min(results.values())
        
        if min_tpr > 0:
            disparity_ratio = max_tpr / min_tpr
            return {
                'metric': 'Equal Opportunity (TPR Parity)',
                'by_group': results,
                'disparity_ratio': disparity_ratio,
                'fair': disparity_ratio < 1.25,
                'interpretation': 'Similar true positive rates across groups' if disparity_ratio < 1.25
                                 else 'Different detection rates across groups'
            }
        
        return None


class BiasRemediationRecommender:
    """
    Recommend ways to address detected bias
    """
    
    @staticmethod
    def recommend_remediation(bias_findings):
        """
        Recommend remediation steps for detected bias
        """
        recommendations = []
        
        for finding in bias_findings:
            if finding['metric'] == 'False Positive Rate':
                recommendations.append({
                    'issue': finding['description'],
                    'impact': f"Group {finding['demographic']} unfairly flagged",
                    'remediation_steps': [
                        'Adjust prediction threshold for affected group',
                        'Re-weight features that disproportionately affect group',
                        'Collect more training data for underrepresented group',
                        'Use fairness-aware learning techniques (e.g., Fairlearn)',
                        'Manual review process for affected predictions'
                    ]
                })
            
            elif finding['metric'] == 'False Negative Rate':
                recommendations.append({
                    'issue': finding['description'],
                    'impact': f"Actual attrition in {finding['demographic']} missed",
                    'remediation_steps': [
                        'Lower threshold for affected group to improve recall',
                        'Review feature importance for bias',
                        'Add group-specific features',
                        'Implement group-specific models',
                        'Increase monitoring for this group'
                    ]
                })
        
        return recommendations


class FairMLMonitor:
    """
    Monitor fairness over time as model makes predictions
    """
    
    def __init__(self, protected_attributes):
        self.protected_attributes = protected_attributes
        self.prediction_log = []
    
    def log_prediction(self, features, prediction, actual_outcome=None, demographics=None):
        """
        Log prediction for fairness monitoring
        """
        self.prediction_log.append({
            'timestamp': pd.Timestamp.now(),
            'prediction': prediction,
            'actual': actual_outcome,
            'demographics': demographics
        })
    
    def compute_fairness_metrics(self):
        """
        Compute fairness metrics from logged predictions
        """
        if not self.prediction_log:
            return None
        
        df = pd.DataFrame(self.prediction_log)
        metrics = {}
        
        for attr in self.protected_attributes:
            if attr in df.columns:
                metrics[attr] = df.groupby(attr)['prediction'].mean()
        
        return metrics
    
    def detect_fairness_drift(self):
        """
        Detect if fairness is degrading over time
        """
        metrics = self.compute_fairness_metrics()
        
        if not metrics:
            return None
        
        drift_report = []
        
        for attr, values in metrics.items():
            max_val = max(values)
            min_val = min(values)
            
            if min_val > 0:
                ratio = max_val / min_val
                if ratio > 1.25:
                    drift_report.append({
                        'attribute': attr,
                        'disparity_ratio': ratio,
                        'status': 'DRIFT DETECTED',
                        'recommendation': 'Review model retraining'
                    })
        
        return drift_report


def generate_fairness_report(model, X, y, demographic_df, demographic_columns):
    """
    Generate comprehensive fairness report
    """
    auditor = FairnessAudit(model)
    
    report = {
        'timestamp': pd.Timestamp.now(),
        'audits': {},
        'biases_detected': [],
        'recommendations': []
    }
    
    for column in demographic_columns:
        audit_results = auditor.audit_by_demographic(X, y, demographic_df, column)
        report['audits'][column] = audit_results
    
    bias_findings = auditor.detect_bias(report['audits'])
    report['biases_detected'] = bias_findings
    
    if bias_findings:
        recommendations = BiasRemediationRecommender.recommend_remediation(bias_findings)
        report['recommendations'] = recommendations
    
    return report
