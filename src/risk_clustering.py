# src/risk_clustering.py
"""
Risk Clustering Module
Segment employees by risk profile for targeted HR strategies
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class RiskSegmentation:
    """
    Segment employees into clusters based on attrition risk profiles
    
    Different clusters have different root causes:
    - Cluster A: High income but burning out → Workload issue
    - Cluster B: Low income → Compensation issue
    - Cluster C: Stagnant tenure → Career growth issue
    - Cluster D: Low risk → Maintain
    """
    
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.random_state = random_state
        self.pca = PCA(n_components=2)
    
    def segment_employees(self, employees_df, model):
        """
        Segment employees by risk profile
        
        Args:
            employees_df: DataFrame with employee data
            model: Trained attrition model
        
        Returns:
            DataFrame with cluster assignments and interpretation
        """
        
        # Calculate risk for each employee
        X_features = employees_df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'OverTime']]
        
        # Convert OverTime to numeric if needed
        if X_features['OverTime'].dtype == 'object':
            X_features = X_features.copy()
            X_features['OverTime'] = (X_features['OverTime'] == 'Yes').astype(int)
        
        employees_df = employees_df.copy()
        employees_df['RiskScore'] = model.predict_proba(X_features)[:, 1]
        
        # Create clustering features
        employees_df['Income_per_tenure'] = (
            employees_df['MonthlyIncome'] / (employees_df['YearsAtCompany'] + 1)
        )
        employees_df['Overtime_intensity'] = employees_df['OverTime'].astype(int)
        
        clustering_features = employees_df[[
            'RiskScore',
            'Income_per_tenure',
            'Overtime_intensity',
            'YearsAtCompany'
        ]]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(clustering_features)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        employees_df['Cluster'] = self.kmeans.fit_predict(scaled_features)
        
        # Fit PCA for visualization
        self.pca.fit(scaled_features)
        
        # Interpret clusters
        cluster_info = self._interpret_clusters(employees_df)
        
        return employees_df, cluster_info
    
    def _interpret_clusters(self, df):
        """
        Interpret each cluster and provide recommendations
        """
        clusters = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            avg_risk = cluster_data['RiskScore'].mean()
            avg_income = cluster_data['MonthlyIncome'].mean()
            avg_tenure = cluster_data['YearsAtCompany'].mean()
            avg_age = cluster_data['Age'].mean()
            avg_ot = cluster_data['Overtime_intensity'].mean()
            
            # Characterize cluster
            cluster_profile = self._profile_cluster(
                avg_risk, avg_income, avg_tenure, avg_age, avg_ot
            )
            
            clusters[cluster_id] = {
                'name': cluster_profile['name'],
                'color': cluster_profile['color'],
                'description': cluster_profile['description'],
                'size': len(cluster_data),
                'percentage': f"{len(cluster_data) / len(df) * 100:.1f}%",
                'avg_risk': avg_risk,
                'avg_income': avg_income,
                'avg_tenure': avg_tenure,
                'avg_age': avg_age,
                'avg_overtime': avg_ot,
                'primary_issue': cluster_profile['primary_issue'],
                'recommendation': cluster_profile['recommendation'],
                'urgent_actions': cluster_profile['urgent_actions'],
                'members': cluster_data,
                'member_count': len(cluster_data)
            }
        
        return clusters
    
    def _profile_cluster(self, risk, income, tenure, age, overtime):
        """
        Profile cluster based on characteristics
        """
        
        # Cluster 1: HIGH RISK + OVERTIME → Burnout
        if risk > 0.65 and overtime > 0.5:
            return {
                'name': '🔴 BURNOUT RISK',
                'color': '#ff4444',
                'description': 'High attrition risk due to work overload and burnout',
                'primary_issue': 'Excessive overtime and workload pressure',
                'recommendation': 'Immediate intervention: reduce overtime, redistribute workload',
                'urgent_actions': [
                    'Schedule 1-on-1 meetings with managers',
                    'Analyze workload distribution',
                    'Offer flexible work arrangements',
                    'Provide mental health support',
                    'Review project deadlines'
                ]
            }
        
        # Cluster 2: HIGH RISK + LOW INCOME → Compensation
        elif risk > 0.65 and income < 40000:
            return {
                'name': '🔴 COMPENSATION RISK',
                'color': '#ff6666',
                'description': 'High risk driven by low compensation',
                'primary_issue': 'Insufficient salary relative to market',
                'recommendation': 'Conduct market salary review and adjustment',
                'urgent_actions': [
                    'Market rate analysis for roles',
                    'Compensation adjustment plan',
                    'Career progression pathways',
                    'Performance bonus opportunities',
                    'Skill development for advancement'
                ]
            }
        
        # Cluster 3: HIGH RISK + HIGH TENURE → Stagnation
        elif risk > 0.65 and tenure > 8:
            return {
                'name': '🔴 STAGNATION RISK',
                'color': '#ff8844',
                'description': 'Long-term employees at risk due to career stagnation',
                'primary_issue': 'Limited advancement and growth opportunities',
                'recommendation': 'Create clear career progression and development plans',
                'urgent_actions': [
                    'Career development conversations',
                    'Internal job opportunities',
                    'Leadership training programs',
                    'Mentorship assignments',
                    'Project leadership opportunities'
                ]
            }
        
        # Cluster 4: MEDIUM RISK → Monitor
        elif risk > 0.35:
            return {
                'name': '🟡 MEDIUM RISK',
                'color': '#ffcc00',
                'description': 'Moderate attrition risk, requires monitoring',
                'primary_issue': 'Mixed factors contributing to risk',
                'recommendation': 'Regular check-ins and engagement activities',
                'urgent_actions': [
                    'Quarterly career development reviews',
                    'Engagement survey feedback',
                    'Manager 1-on-1s',
                    'Recognition programs',
                    'Learning opportunities'
                ]
            }
        
        # Cluster 5: EARLY CAREER + GROWING → Develop
        elif age < 35 and tenure < 3:
            return {
                'name': '🔵 JUNIOR TALENT',
                'color': '#4488ff',
                'description': 'Young professionals with growth potential',
                'primary_issue': 'Need for clear career paths and mentorship',
                'recommendation': 'Invest in development and career pathing',
                'urgent_actions': [
                    'Mentorship matching',
                    'Technical skill development',
                    'Career pathing workshops',
                    'Cross-functional projects',
                    'Performance feedback'
                ]
            }
        
        # Cluster 6: LOW RISK + STABLE → Retain
        else:
            return {
                'name': '🟢 STABLE & ENGAGED',
                'color': '#44ff44',
                'description': 'Low risk, stable, engaged employees',
                'primary_issue': 'None - maintain satisfaction',
                'recommendation': 'Continue current engagement practices',
                'urgent_actions': [
                    'Maintain good management',
                    'Career growth discussions',
                    'Recognition programs',
                    'Retention bonus consideration',
                    'Leadership opportunities'
                ]
            }
    
    def get_cluster_summary(self, clusters):
        """
        Create summary table of clusters
        """
        summary = []
        
        for cluster_id, info in clusters.items():
            summary.append({
                'Cluster': info['name'],
                'Size': info['size'],
                'Percentage': info['percentage'],
                'Avg Risk': f"{info['avg_risk']:.1%}",
                'Avg Income': f"₹{info['avg_income']:,.0f}",
                'Avg Tenure': f"{info['avg_tenure']:.1f} yrs",
                'Primary Issue': info['primary_issue']
            })
        
        return pd.DataFrame(summary)
    
    def get_cluster_comparison(self, clusters):
        """
        Detailed comparison table
        """
        comparison = []
        
        for cluster_id, info in clusters.items():
            comparison.append({
                'Cluster': info['name'],
                'Recommendation': info['recommendation'],
                'Urgent Actions': ' | '.join(info['urgent_actions'][:2])
            })
        
        return pd.DataFrame(comparison)
    
    def visualize_clusters(self, employees_df, clusters):
        """
        Create visualization of clusters
        """
        scaling_features = employees_df[[
            'RiskScore',
            'Income_per_tenure',
            'Overtime_intensity',
            'YearsAtCompany'
        ]]
        
        scaled_features = self.scaler.transform(scaling_features)
        pca_features = self.pca.transform(scaled_features)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cluster visualization
        colors = [clusters[i]['color'] for i in employees_df['Cluster']]
        ax1.scatter(pca_features[:, 0], pca_features[:, 1], c=colors, alpha=0.6, s=100)
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        ax1.set_title('Employee Risk Clusters')
        
        # Risk distribution by cluster
        cluster_risks = [clusters[i]['avg_risk'] for i in range(self.n_clusters)]
        cluster_names = [clusters[i]['name'] for i in range(self.n_clusters)]
        cluster_colors = [clusters[i]['color'] for i in range(self.n_clusters)]
        
        ax2.barh(cluster_names, cluster_risks, color=cluster_colors, alpha=0.7)
        ax2.set_xlabel('Average Risk Score')
        ax2.set_title('Risk by Cluster')
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        return fig


class TargetedRetentionStrategy:
    """
    Generate targeted HR strategies for each cluster
    """
    
    @staticmethod
    def get_strategy_for_cluster(cluster_info):
        """
        Get detailed strategy for specific cluster
        """
        return {
            'cluster': cluster_info['name'],
            'size': cluster_info['size'],
            'strategy': {
                'immediate_actions': cluster_info['urgent_actions'],
                'timeline': 'This month',
                'responsible_team': 'HR + Direct Managers',
                'success_metrics': [
                    'Risk score reduction by 15%',
                    'Employee satisfaction survey',
                    'Retention rate improvement'
                ]
            }
        }
