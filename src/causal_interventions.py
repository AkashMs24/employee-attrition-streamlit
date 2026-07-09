# src/causal_interventions.py
"""
Causal Intervention Recommendation Engine
Recommends specific actions with predicted impact on attrition risk
"""

import numpy as np
import pandas as pd


class InterventionRecommender:
    """
    Test different HR interventions and predict their impact
    
    Interventions:
    - Salary increase
    - Overtime reduction
    - Career development (promotion)
    - Flexible work arrangements
    """
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or ['Age', 'Income', 'Tenure', 'OverTime']
    
    def recommend_interventions(self, employee_data, current_risk):
        """
        Recommend actionable interventions with ROI
        
        Args:
            employee_data: dict with keys ['age', 'income', 'years', 'ot_encoded']
            current_risk: current attrition probability (0-1)
        
        Returns:
            List of recommendations sorted by impact
        """
        
        recommendations = []
        
        # If risk is already low, maintain status
        if current_risk < 0.25:
            return [{
                'category': 'Maintenance',
                'action': 'Continue current engagement',
                'current_risk': current_risk,
                'new_risk': current_risk,
                'improvement': 0,
                'effort': 'Low',
                'cost': '₹0',
                'timeline': 'Ongoing',
                'roi_score': 0
            }]
        
        # Test salary interventions
        salary_recs = self._test_salary_increase(employee_data, current_risk)
        recommendations.extend(salary_recs)
        
        # Test overtime reduction
        ot_recs = self._test_overtime_reduction(employee_data, current_risk)
        recommendations.extend(ot_recs)
        
        # Test career development
        career_recs = self._test_career_development(employee_data, current_risk)
        recommendations.extend(career_recs)
        
        # Test work flexibility
        flex_recs = self._test_work_flexibility(employee_data, current_risk)
        recommendations.extend(flex_recs)
        
        # Calculate ROI score and sort
        for rec in recommendations:
            rec['roi_score'] = self._calculate_roi(rec)
        
        # Filter to show only impactful interventions
        recommendations = [r for r in recommendations if r['improvement'] > 0.05]
        
        return sorted(recommendations, key=lambda x: x['improvement'], reverse=True)
    
    def _test_salary_increase(self, employee_data, current_risk):
        """
        Test salary increase impact
        """
        recommendations = []
        original_income = employee_data['income']
        
        for increase_pct in [5, 10, 15, 20, 25]:
            new_income = original_income * (1 + increase_pct / 100)
            new_risk = self._predict_with_modification(
                employee_data,
                {'income': new_income}
            )
            
            improvement = (current_risk - new_risk) * 100
            
            if improvement > 3:  # Only recommend if meaningful improvement
                annual_cost = original_income * increase_pct / 100 * 12
                
                recommendations.append({
                    'category': 'Compensation',
                    'action': f'Salary Increase',
                    'details': f'+{increase_pct}% increase (₹{new_income:,.0f}/month)',
                    'current_risk': current_risk,
                    'new_risk': new_risk,
                    'improvement': improvement,
                    'improvement_pct': improvement,
                    'effort': 'Medium',
                    'cost': f'₹{annual_cost:,.0f}/year',
                    'cost_value': annual_cost,
                    'timeline': '1-2 weeks',
                    'probability': 'High',
                    'notes': 'Direct impact on compensation satisfaction'
                })
        
        return recommendations
    
    def _test_overtime_reduction(self, employee_data, current_risk):
        """
        Test reducing overtime/workload
        """
        recommendations = []
        
        if employee_data['ot_encoded'] == 0:
            return []  # Not applicable
        
        # Overtime reduction (convert 1 to 0)
        new_data = employee_data.copy()
        new_data['ot_encoded'] = 0
        
        new_risk = self._predict_with_modification(employee_data, new_data)
        improvement = (current_risk - new_risk) * 100
        
        if improvement > 3:
            recommendations.append({
                'category': 'Workload',
                'action': 'Overtime Reduction',
                'details': 'Reduce/eliminate mandatory overtime',
                'current_risk': current_risk,
                'new_risk': new_risk,
                'improvement': improvement,
                'improvement_pct': improvement,
                'effort': 'High',
                'cost': '₹0 (process change)',
                'cost_value': 0,
                'timeline': '2-4 weeks',
                'probability': 'High',
                'notes': 'Reduces burnout, improves work-life balance'
            })
        
        return recommendations
    
    def _test_career_development(self, employee_data, current_risk):
        """
        Test career advancement/promotion
        """
        recommendations = []
        
        # Promotion proxy: salary increase + tenure benefit
        new_data = employee_data.copy()
        new_data['income'] = employee_data['income'] * 1.20  # 20% raise with promotion
        new_data['years'] = employee_data['years'] + 1  # Count as 1 more year stable
        
        new_risk = self._predict_with_modification(employee_data, new_data)
        improvement = (current_risk - new_risk) * 100
        
        if improvement > 5:
            annual_cost = employee_data['income'] * 0.20 * 12
            
            recommendations.append({
                'category': 'Career Development',
                'action': 'Promotion to Senior Role',
                'details': 'Promote with salary increase + responsibility',
                'current_risk': current_risk,
                'new_risk': new_risk,
                'improvement': improvement,
                'improvement_pct': improvement,
                'effort': 'Very High',
                'cost': f'₹{annual_cost:,.0f}/year + training',
                'cost_value': annual_cost,
                'timeline': '1-3 months',
                'probability': 'Very High',
                'notes': 'Addresses career growth and compensation'
            })
        
        return recommendations
    
    def _test_work_flexibility(self, employee_data, current_risk):
        """
        Test flexible work arrangements (proxy: small income benefit + OT reduction)
        """
        recommendations = []
        
        new_data = employee_data.copy()
        new_data['income'] = employee_data['income'] * 1.05  # 5% benefit
        new_data['ot_encoded'] = 0 if employee_data['ot_encoded'] == 1 else 0  # Flexibility helps with OT
        
        new_risk = self._predict_with_modification(employee_data, new_data)
        improvement = (current_risk - new_risk) * 100
        
        if improvement > 2:
            recommendations.append({
                'category': 'Work Flexibility',
                'action': 'Flexible Work Arrangement',
                'details': 'Remote option / flexible hours',
                'current_risk': current_risk,
                'new_risk': new_risk,
                'improvement': improvement,
                'improvement_pct': improvement,
                'effort': 'Low',
                'cost': '₹0-50K/year (tools)',
                'cost_value': 25000,
                'timeline': '1-2 weeks',
                'probability': 'High',
                'notes': 'Low cost, high satisfaction impact'
            })
        
        return recommendations
    
    def _predict_with_modification(self, original_data, modifications):
        """
        Predict with modified features
        """
        modified_data = original_data.copy()
        modified_data.update(modifications)
        
        features = [
            modified_data['age'],
            modified_data['income'],
            modified_data['years'],
            modified_data['ot_encoded']
        ]
        
        return self.model.predict_proba([features])[0][1]
    
    def _calculate_roi(self, recommendation):
        """
        Calculate ROI score (improvement per cost)
        """
        if recommendation['cost_value'] == 0:
            return recommendation['improvement'] * 100
        else:
            return (recommendation['improvement'] / recommendation['cost_value']) * 10000


class InterventionImpactAnalyzer:
    """
    Analyze combined intervention strategies
    """
    
    def __init__(self, model):
        self.model = model
        self.recommender = InterventionRecommender(model)
    
    def combined_intervention_strategy(self, employee_data, current_risk, budget_limit=None):
        """
        Find best combination of interventions within budget
        """
        individual_recs = self.recommender.recommend_interventions(
            employee_data, current_risk
        )
        
        if not individual_recs:
            return {'status': 'Low risk - no interventions needed'}
        
        # Find best combination
        best_strategy = self._find_optimal_combination(
            individual_recs, current_risk, budget_limit
        )
        
        return best_strategy
    
    def _find_optimal_combination(self, recommendations, current_risk, budget_limit=None):
        """
        Find best combination respecting budget
        """
        # Start with highest impact
        top_rec = recommendations[0]
        
        return {
            'primary_intervention': top_rec['action'],
            'details': top_rec['details'],
            'expected_new_risk': top_rec['new_risk'],
            'risk_reduction': f"{top_rec['improvement']:.1f}%",
            'cost': top_rec['cost'],
            'timeline': top_rec['timeline'],
            'secondary_recommendations': [
                r['action'] for r in recommendations[1:3]
            ],
            'success_probability': top_rec['probability']
        }


def create_intervention_report(model, employees_df):
    """
    Create batch intervention recommendations for all at-risk employees
    """
    recommender = InterventionRecommender(model)
    report = []
    
    for _, emp in employees_df.iterrows():
        employee_data = {
            'age': emp['Age'],
            'income': emp['MonthlyIncome'],
            'years': emp['YearsAtCompany'],
            'ot_encoded': 1 if emp['OverTime'] == 'Yes' else 0
        }
        
        current_risk = model.predict_proba([
            [emp['Age'], emp['MonthlyIncome'], emp['YearsAtCompany'], 
             1 if emp['OverTime'] == 'Yes' else 0]
        ])[0][1]
        
        if current_risk > 0.35:  # Only for at-risk employees
            recs = recommender.recommend_interventions(employee_data, current_risk)
            
            if recs:
                report.append({
                    'employee_id': emp.get('EmployeeID', 'N/A'),
                    'current_risk': current_risk,
                    'recommended_intervention': recs[0]['action'],
                    'expected_impact': recs[0]['improvement'],
                    'cost': recs[0]['cost']
                })
    
    return pd.DataFrame(report)
