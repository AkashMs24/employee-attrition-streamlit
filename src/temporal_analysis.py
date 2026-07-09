# src/temporal_analysis.py
"""
Temporal Risk Analysis Module
Analyzes how attrition risk changes over time
Predicts risk trajectory and early warnings
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TemporalAttritionAnalysis:
    """
    Analyze employee risk over time, predict trajectory
    
    Features:
    - Risk velocity (rate of change)
    - Risk trajectory classification
    - 3-month and 6-month risk projections
    - Early warning system
    """
    
    def __init__(self, model):
        self.model = model
    
    def calculate_risk_trajectory(self, employee_history):
        """
        Calculate risk for each time period, fit trend line
        
        Args:
            employee_history: List of dicts with quarterly/monthly snapshots
            [
                {
                    'period': 'Q1',
                    'age': 30,
                    'income': 50000,
                    'years': 5,
                    'overtime': 1,
                    'satisfaction': 7,
                    'date': '2023-01-01'
                },
                ...
            ]
        
        Returns:
            Dictionary with risk metrics and trajectory
        """
        
        if len(employee_history) < 2:
            return {'error': 'Need at least 2 historical records'}
        
        quarterly_risks = []
        periods = []
        dates = []
        
        # Calculate risk for each snapshot
        for i, snapshot in enumerate(employee_history):
            features = [
                snapshot['age'],
                snapshot['income'],
                snapshot['years'],
                snapshot['overtime']
            ]
            
            risk = self.model.predict_proba([features])[0][1]
            quarterly_risks.append(risk)
            periods.append(i)
            dates.append(snapshot.get('date', f'Period {i}'))
        
        quarterly_risks = np.array(quarterly_risks)
        periods = np.array(periods)
        
        # Fit linear regression to calculate trend
        X = periods.reshape(-1, 1)
        y = quarterly_risks
        
        trend_model = LinearRegression()
        trend_model.fit(X, y)
        
        # Risk velocity (change per period)
        risk_velocity = trend_model.coef_[0]
        
        # Project future risk
        current_period = len(periods)
        risk_3m = min(1.0, max(0.0, quarterly_risks[-1] + (risk_velocity * 3)))
        risk_6m = min(1.0, max(0.0, quarterly_risks[-1] + (risk_velocity * 6)))
        
        # Classify trajectory
        trajectory_classification = self._classify_trajectory(
            quarterly_risks, risk_velocity
        )
        
        # Alert level
        alert_level = self._get_alert_level(
            quarterly_risks[-1], risk_velocity, trajectory_classification
        )
        
        return {
            'current_risk': float(quarterly_risks[-1]),
            'risk_history': [float(r) for r in quarterly_risks],
            'dates': dates,
            'risk_velocity': float(risk_velocity),  # Change per period
            'risk_trajectory': trajectory_classification,
            'projected_3m_risk': float(risk_3m),
            'projected_6m_risk': float(risk_6m),
            'alert_level': alert_level,
            'trend_analysis': self._trend_analysis(quarterly_risks, risk_velocity),
            'recommendation': self._get_recommendation(
                quarterly_risks[-1], risk_velocity, alert_level
            )
        }
    
    def _classify_trajectory(self, risks, velocity):
        """
        Classify if risk is rising, stable, or improving
        """
        if velocity > 0.08:
            return "🔴 RAPIDLY RISING"
        elif velocity > 0.03:
            return "🟠 GRADUALLY RISING"
        elif velocity > -0.03:
            return "🟡 STABLE"
        elif velocity > -0.08:
            return "🟢 GRADUALLY IMPROVING"
        else:
            return "🟢 RAPIDLY IMPROVING"
    
    def _get_alert_level(self, current_risk, velocity, trajectory):
        """
        Determine urgency level
        """
        if current_risk > 0.75 and "RISING" in trajectory:
            return "🔴 CRITICAL - Immediate action required"
        elif current_risk > 0.75:
            return "🔴 HIGH - High risk (stable)"
        elif current_risk > 0.60 and velocity > 0.05:
            return "🟠 URGENT - Risk increasing rapidly"
        elif current_risk > 0.60:
            return "🟠 HIGH - Monitor closely"
        elif current_risk > 0.35 and velocity > 0.05:
            return "🟡 MEDIUM - Risk increasing"
        elif current_risk > 0.35:
            return "🟡 MEDIUM - Watch for changes"
        else:
            return "🟢 LOW - Maintain engagement"
    
    def _trend_analysis(self, risks, velocity):
        """
        Detailed trend analysis
        """
        recent_trend = risks[-1] - risks[-2] if len(risks) > 1 else 0
        overall_trend = risks[-1] - risks[0]
        
        return {
            'recent_change': float(recent_trend),
            'overall_change': float(overall_trend),
            'volatility': float(np.std(np.diff(risks))),
            'direction': 'UP' if velocity > 0 else 'DOWN'
        }
    
    def _get_recommendation(self, current_risk, velocity, alert_level):
        """
        Get actionable recommendations based on trajectory
        """
        recommendations = []
        
        if current_risk > 0.75:
            recommendations.append("URGENT: Schedule immediate 1-on-1 meeting")
            recommendations.append("Consider retention package review")
        
        if velocity > 0.08:
            recommendations.append("Risk increasing rapidly - investigate recent changes")
            recommendations.append("Check for dissatisfaction signals (missed meetings, etc)")
        
        if current_risk > 0.50:
            recommendations.append("Explore career development opportunities")
            recommendations.append("Review compensation and benefits")
        
        if "STABLE" in alert_level and current_risk > 0.35:
            recommendations.append("Schedule quarterly check-ins")
            recommendations.append("Monitor for stress signals")
        
        if velocity < -0.05:
            recommendations.append("Continue current engagement strategies")
            recommendations.append("Risk is improving - maintain momentum")
        
        return recommendations
    
    def batch_trajectory_analysis(self, employees_df):
        """
        Analyze trajectories for multiple employees
        
        Args:
            employees_df: DataFrame with columns:
            employee_id, age, income, years, overtime, period, date
        
        Returns:
            DataFrame with trajectory metrics for each employee
        """
        
        results = []
        
        for emp_id in employees_df['employee_id'].unique():
            emp_data = employees_df[employees_df['employee_id'] == emp_id]
            
            history = []
            for _, row in emp_data.iterrows():
                history.append({
                    'age': row['age'],
                    'income': row['income'],
                    'years': row['years'],
                    'overtime': row['overtime'],
                    'date': row.get('date', row.get('period', ''))
                })
            
            trajectory = self.calculate_risk_trajectory(history)
            
            if 'error' not in trajectory:
                results.append({
                    'employee_id': emp_id,
                    'current_risk': trajectory['current_risk'],
                    'risk_velocity': trajectory['risk_velocity'],
                    'trajectory': trajectory['risk_trajectory'],
                    'alert_level': trajectory['alert_level'],
                    'projected_3m': trajectory['projected_3m_risk']
                })
        
        return pd.DataFrame(results)


class EarlyWarningSystem:
    """
    Detect employees at early risk stage before high-risk reaches crisis
    """
    
    def __init__(self, model):
        self.model = model
        self.temporal = TemporalAttritionAnalysis(model)
    
    def identify_at_risk_early(self, employees_df):
        """
        Find employees with:
        1. Medium current risk + rising trend
        2. Low current risk + rapidly rising
        3. Sudden changes
        """
        
        trajectory_df = self.temporal.batch_trajectory_analysis(employees_df)
        
        early_warnings = []
        
        for _, row in trajectory_df.iterrows():
            # Rule 1: Medium risk + rising
            if (0.35 < row['current_risk'] < 0.60 and row['risk_velocity'] > 0.05):
                early_warnings.append({
                    'employee_id': row['employee_id'],
                    'reason': 'Rising medium risk',
                    'severity': 'HIGH',
                    'current_risk': row['current_risk'],
                    'velocity': row['risk_velocity']
                })
            
            # Rule 2: Low risk but rapidly rising
            if (row['current_risk'] < 0.35 and row['risk_velocity'] > 0.10):
                early_warnings.append({
                    'employee_id': row['employee_id'],
                    'reason': 'Rapidly increasing (early stage)',
                    'severity': 'MEDIUM',
                    'current_risk': row['current_risk'],
                    'velocity': row['risk_velocity']
                })
            
            # Rule 3: Sudden jump in risk
            if len(trajectory_df) > 1 and row['risk_velocity'] > 0.15:
                early_warnings.append({
                    'employee_id': row['employee_id'],
                    'reason': 'Sudden risk spike',
                    'severity': 'CRITICAL',
                    'current_risk': row['current_risk'],
                    'velocity': row['risk_velocity']
                })
        
        return pd.DataFrame(early_warnings)


# Example usage
def analyze_employee_trajectory(model, employee_data_list):
    """
    Helper function to analyze single employee trajectory
    """
    temporal = TemporalAttritionAnalysis(model)
    return temporal.calculate_risk_trajectory(employee_data_list)
