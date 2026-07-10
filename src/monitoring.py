# src/monitoring.py
"""
Production Monitoring Module
Monitor model performance, drift, and data quality in production
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ModelPerformanceMonitor:
    """
    Monitor model prediction accuracy over time
    Detect model drift (performance degradation)
    """
    
    def __init__(self, baseline_roc_auc=0.85):
        self.baseline_roc_auc = baseline_roc_auc
        self.predictions_log = []
        self.performance_history = []
    
    def log_prediction(self, prediction_proba, actual_outcome=None, timestamp=None):
        """
        Log a prediction for monitoring
        
        Args:
            prediction_proba: Model probability (0-1)
            actual_outcome: True label (0 or 1) if available
            timestamp: When prediction was made
        """
        self.predictions_log.append({
            'timestamp': timestamp or datetime.now(),
            'prediction': prediction_proba,
            'actual': actual_outcome,
            'prediction_binary': 1 if prediction_proba > 0.5 else 0
        })
    
    def batch_log_predictions(self, predictions_proba, actual_outcomes=None):
        """
        Log multiple predictions at once
        """
        for i, pred in enumerate(predictions_proba):
            actual = actual_outcomes[i] if actual_outcomes is not None else None
            self.log_prediction(pred, actual)
    
    def calculate_calibration(self, window_size=100):
        """
        Check if predicted probabilities match actual outcomes
        (Calibration: when we say 70% risk, do ~70% actually leave?)
        """
        if len(self.predictions_log) < window_size:
            return None
        
        recent = self.predictions_log[-window_size:]
        
        # Only use records with actual outcomes
        complete_records = [r for r in recent if r['actual'] is not None]
        
        if len(complete_records) < window_size // 2:
            return None
        
        predictions = np.array([r['prediction'] for r in complete_records])
        actuals = np.array([r['actual'] for r in complete_records])
        
        # Expected vs observed attrition rate
        expected_rate = predictions.mean()
        observed_rate = actuals.mean()
        
        calibration_error = abs(expected_rate - observed_rate)
        
        return {
            'window_size': len(complete_records),
            'expected_attrition_rate': expected_rate,
            'observed_attrition_rate': observed_rate,
            'calibration_error': calibration_error,
            'is_calibrated': calibration_error < 0.10,
            'status': 'GOOD' if calibration_error < 0.10 else 'DRIFT DETECTED'
        }
    
    def detect_prediction_distribution_shift(self, window_size=100):
        """
        Detect if prediction distribution is changing (data distribution shift)
        """
        if len(self.predictions_log) < window_size * 2:
            return None
        
        # Split into two windows
        early_window = [r['prediction'] for r in self.predictions_log[-window_size*2:-window_size]]
        recent_window = [r['prediction'] for r in self.predictions_log[-window_size:]]
        
        early_mean = np.mean(early_window)
        recent_mean = np.mean(recent_window)
        
        early_std = np.std(early_window)
        recent_std = np.std(recent_window)
        
        shift_magnitude = abs(recent_mean - early_mean)
        
        return {
            'early_window_mean': early_mean,
            'recent_window_mean': recent_mean,
            'mean_shift': shift_magnitude,
            'early_std': early_std,
            'recent_std': recent_std,
            'shift_detected': shift_magnitude > 0.10,
            'recommendation': 'Consider retraining' if shift_magnitude > 0.10 else 'No action needed'
        }
    
    def detect_feature_drift(self, new_features_df, baseline_features_df):
        """
        Detect if input features are changing significantly
        """
        drift_report = {}
        
        for column in baseline_features_df.columns:
            if column not in new_features_df.columns:
                continue
            
            baseline_mean = baseline_features_df[column].mean()
            baseline_std = baseline_features_df[column].std()
            
            new_mean = new_features_df[column].mean()
            
            # Z-score: how many std deviations is the shift?
            z_score = abs(new_mean - baseline_mean) / (baseline_std + 0.001)
            
            drift_report[column] = {
                'baseline_mean': baseline_mean,
                'current_mean': new_mean,
                'z_score': z_score,
                'drift_detected': z_score > 2,  # >2 std dev = drift
                'severity': 'HIGH' if z_score > 3 else 'MEDIUM' if z_score > 2 else 'LOW'
            }
        
        return drift_report
    
    def get_performance_summary(self):
        """
        Get summary of model performance monitoring
        """
        calibration = self.calculate_calibration()
        distribution_shift = self.detect_prediction_distribution_shift()
        
        return {
            'total_predictions_logged': len(self.predictions_log),
            'calibration': calibration,
            'distribution_shift': distribution_shift,
            'overall_status': self._determine_overall_status(calibration, distribution_shift)
        }
    
    def _determine_overall_status(self, calibration, distribution_shift):
        """
        Determine if model needs retraining
        """
        if calibration is None or distribution_shift is None:
            return "INSUFFICIENT_DATA"
        
        if not calibration['is_calibrated'] or distribution_shift['shift_detected']:
            return "⚠️ DRIFT DETECTED - RETRAIN RECOMMENDED"
        else:
            return "✓ HEALTHY - NO ACTION NEEDED"


class DataQualityMonitor:
    """
    Monitor data quality of incoming predictions
    Detect missing values, outliers, anomalies
    """
    
    def __init__(self, expected_ranges=None):
        self.expected_ranges = expected_ranges or {
            'Age': (18, 70),
            'MonthlyIncome': (1000, 500000),
            'YearsAtCompany': (0, 50),
            'OverTime': (0, 1)
        }
        self.data_quality_log = []
    
    def check_data_quality(self, features_df):
        """
        Check data quality of features
        """
        issues = []
        
        # Check for missing values
        missing = features_df.isnull().sum()
        if missing.sum() > 0:
            issues.append({
                'type': 'Missing Values',
                'severity': 'HIGH',
                'details': missing[missing > 0].to_dict()
            })
        
        # Check for out-of-range values
        for col, (min_val, max_val) in self.expected_ranges.items():
            if col in features_df.columns:
                out_of_range = (
                    (features_df[col] < min_val) | (features_df[col] > max_val)
                ).sum()
                
                if out_of_range > 0:
                    issues.append({
                        'type': f'{col} out of range',
                        'severity': 'MEDIUM',
                        'count': out_of_range,
                        'valid_range': (min_val, max_val)
                    })
        
        # Check for duplicates
        duplicates = features_df.duplicated().sum()
        if duplicates > 0:
            issues.append({
                'type': 'Duplicate Records',
                'severity': 'LOW',
                'count': duplicates
            })
        
        quality_score = 1.0 - (len(issues) * 0.1)
        quality_score = max(0, min(1, quality_score))
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'status': 'PASS' if quality_score > 0.9 else 'WARNING' if quality_score > 0.7 else 'FAIL'
        }
    
    def detect_anomalies(self, features_df, z_score_threshold=3):
        """
        Detect anomalous records using z-score
        """
        anomalies = []
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = features_df[col].mean()
            std = features_df[col].std()
            
            z_scores = np.abs((features_df[col] - mean) / (std + 0.001))
            anomalous = z_scores > z_score_threshold
            
            if anomalous.sum() > 0:
                anomalies.append({
                    'column': col,
                    'anomaly_count': int(anomalous.sum()),
                    'percentage': f"{anomalous.sum() / len(features_df) * 100:.2f}%"
                })
        
        return anomalies


class AlertingSystem:
    """
    Alert when models needs attention
    """
    
    def __init__(self, alert_thresholds=None):
        self.alert_thresholds = alert_thresholds or {
            'calibration_error': 0.15,
            'prediction_shift': 0.15,
            'data_quality': 0.80,
            'feature_drift_z_score': 2.5
        }
        self.alerts = []
    
    def generate_alerts(self, monitoring_results):
        """
        Generate alerts based on monitoring results
        """
        alerts = []
        
        # Check calibration
        calibration = monitoring_results.get('calibration')
        if calibration and calibration['calibration_error'] > self.alert_thresholds['calibration_error']:
            alerts.append({
                'severity': 'HIGH',
                'type': 'Calibration Drift',
                'message': f"Calibration error: {calibration['calibration_error']:.2%}",
                'action': 'Review model predictions and consider retraining',
                'timestamp': datetime.now()
            })
        
        # Check prediction distribution shift
        shift = monitoring_results.get('distribution_shift')
        if shift and shift['shift_detected']:
            alerts.append({
                'severity': 'HIGH',
                'type': 'Prediction Distribution Shift',
                'message': f"Mean shift: {shift['mean_shift']:.2%}",
                'action': 'Investigate data changes and retrain model',
                'timestamp': datetime.now()
            })
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_active_alerts(self, hours=24):
        """
        Get alerts from last N hours
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a['timestamp'] > cutoff_time]


class RetrainingRecommender:
    """
    Recommend when to retrain the model
    """
    
    @staticmethod
    def should_retrain(monitoring_results, min_predictions=1000):
        """
        Determine if model should be retrained
        """
        reasons = []
        
        if monitoring_results.get('total_predictions_logged', 0) < min_predictions:
            return False, "Not enough predictions logged yet"
        
        # Check calibration
        calibration = monitoring_results.get('calibration')
        if calibration and calibration['calibration_error'] > 0.15:
            reasons.append("Calibration drift detected")
        
        # Check distribution shift
        shift = monitoring_results.get('distribution_shift')
        if shift and shift['shift_detected']:
            reasons.append("Prediction distribution shift detected")
        
        should_retrain = len(reasons) > 0
        
        return should_retrain, reasons
    
    @staticmethod
    def get_retraining_plan():
        """
        Get plan for model retraining
        """
        return {
            'frequency': 'Monthly or on-demand when drift detected',
            'steps': [
                '1. Collect recent data with actual outcomes',
                '2. Run fairness audit on new data',
                '3. Compare new model vs current model',
                '4. A/B test new model on sample users',
                '5. Deploy new model if performance improves',
                '6. Monitor new model performance'
            ],
            'timeline': '1-2 weeks from drift detection',
            'responsible_team': 'ML Engineering + Data Science'
        }


# Example usage functions
def create_monitoring_dashboard(monitoring_results):
    """
    Create data for monitoring dashboard
    """
    return {
        'status': monitoring_results.get('overall_status'),
        'calibration': monitoring_results.get('calibration'),
        'distribution_shift': monitoring_results.get('distribution_shift'),
        'timestamp': datetime.now()
    }


def export_monitoring_report(monitor, file_path='monitoring_report.csv'):
    """
    Export monitoring log to CSV
    """
    df = pd.DataFrame(monitor.predictions_log)
    df.to_csv(file_path, index=False)
    return f"Report exported to {file_path}"
