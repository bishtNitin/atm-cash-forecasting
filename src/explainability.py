"""
Explainability Module for ATM Cash Forecasting
Implements SHAP analysis and natural language generation
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ATMExplainer:
    """Explainability engine for ATM forecasting models"""
    
    def __init__(self, model, feature_names: List[str], config: Dict = None):
        """
        Initialize explainer
        
        Args:
            model: Trained model (LightGBM or sklearn-compatible)
            feature_names: List of feature names
            config: Configuration dictionary
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or {}
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        
    def calculate_shap_values(self, X: pd.DataFrame, 
                             sample_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate SHAP values for the dataset
        
        Args:
            X: Feature DataFrame
            sample_size: Number of samples to use for SHAP calculation
            
        Returns:
            SHAP values array
        """
        # Sample if dataset is large
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create SHAP explainer
        try:
            # For tree-based models (LightGBM)
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)
            self.base_value = self.explainer.expected_value
        except:
            # Fallback to KernelExplainer for other models
            background = shap.sample(X, min(100, len(X)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            self.shap_values = self.explainer.shap_values(X_sample)
            self.base_value = self.explainer.expected_value
        
        return self.shap_values
    
    def explain_prediction(self, atm_id: str, date: str, 
                          shap_values: np.ndarray, 
                          features: pd.Series,
                          prediction: float) -> str:
        """
        Generate natural language explanation for a prediction
        
        Args:
            atm_id: ATM identifier
            date: Date of prediction
            shap_values: SHAP values for this prediction
            features: Feature values for this prediction
            prediction: Model prediction value
            
        Returns:
            Natural language explanation string
        """
        # Get top contributing features
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values,
            'feature_value': features.values
        })
        feature_contributions['abs_shap'] = np.abs(feature_contributions['shap_value'])
        feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)
        
        top_features = feature_contributions.head(5)
        
        # Convert prediction to Lakhs (Indian currency format)
        prediction_lakhs = prediction / 100000
        
        # Determine if prediction is high/medium/low
        if prediction_lakhs > 10:
            level = "High"
        elif prediction_lakhs > 5:
            level = "Medium"
        else:
            level = "Low"
        
        # Start explanation
        explanation = f"Forecast for ATM {atm_id} on {date} is {level} (₹{prediction_lakhs:.2f} Lakhs)"
        
        # Add primary drivers
        drivers = []
        for idx, row in top_features.iterrows():
            feature_name = row['feature']
            shap_val = row['shap_value']
            feature_val = row['feature_value']
            
            # Create human-readable feature descriptions
            feature_desc = self._humanize_feature(feature_name, feature_val)
            impact = abs(shap_val)
            direction = "increasing" if shap_val > 0 else "decreasing"
            
            if impact > 10000:  # Significant impact
                drivers.append(f"{feature_desc} is {direction} demand by ₹{impact/100000:.2f} Lakhs")
        
        if drivers:
            explanation += " primarily because:\n- " + "\n- ".join(drivers[:3])
        
        return explanation
    
    def _humanize_feature(self, feature_name: str, feature_value: float) -> str:
        """
        Convert feature name and value to human-readable description
        
        Args:
            feature_name: Technical feature name
            feature_value: Feature value
            
        Returns:
            Human-readable description
        """
        descriptions = {
            'is_payday': "it is a Payday (Start/Mid/End of month)",
            'is_holiday': "it is a Public Holiday",
            'is_weekend': "it is a Weekend",
            'is_month_start': "it is the Start of Month",
            'is_month_end': "it is the End of Month",
            'is_diwali_season': "it is the Diwali Festival Season",
            'is_holi_season': "it is the Holi Festival Season",
            'day_of_week': f"it is {self._get_day_name(feature_value)}",
            'lag_1': "yesterday's high withdrawal",
            'lag_7': "last week's pattern",
            'lag_30': "last month's trend",
            'rolling_mean_7': "the 7-day average trend",
            'rolling_mean_30': "the 30-day average trend",
            'ema_7': "recent exponential trend",
            'day_of_month': f"it is Day {int(feature_value)} of the month",
            'month': f"it is {self._get_month_name(feature_value)}"
        }
        
        if feature_name in descriptions:
            if feature_name in ['is_payday', 'is_holiday', 'is_weekend', 
                               'is_month_start', 'is_month_end', 
                               'is_diwali_season', 'is_holi_season']:
                if feature_value > 0.5:
                    return descriptions[feature_name]
                else:
                    return f"it is NOT {descriptions[feature_name].replace('it is ', '')}"
            return descriptions[feature_name]
        
        return feature_name
    
    def _get_day_name(self, day_num: float) -> str:
        """Convert day number to name"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[int(day_num)] if 0 <= day_num < 7 else 'Unknown'
    
    def _get_month_name(self, month_num: float) -> str:
        """Convert month number to name"""
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        return months[int(month_num) - 1] if 1 <= month_num <= 12 else 'Unknown'
    
    def plot_waterfall(self, shap_values: np.ndarray, 
                      features: pd.Series,
                      prediction: float,
                      max_display: int = 10,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create waterfall chart showing feature contributions
        
        Args:
            shap_values: SHAP values for prediction
            features: Feature values
            prediction: Model prediction
            max_display: Maximum features to display
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Sort features by absolute SHAP value
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values,
            'feature_value': features.values
        })
        feature_contributions = feature_contributions.sort_values(
            'shap_value', key=abs, ascending=False
        )
        
        # Take top features
        top_features = feature_contributions.head(max_display)
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate cumulative values
        cumulative = self.base_value
        positions = []
        values = []
        colors = []
        labels = []
        
        for idx, row in top_features.iterrows():
            positions.append(len(positions))
            values.append(row['shap_value'])
            colors.append('green' if row['shap_value'] > 0 else 'red')
            
            # Create label with feature value
            feature_desc = self._humanize_feature(row['feature'], row['feature_value'])
            labels.append(f"{feature_desc}\n({row['shap_value']/100000:+.2f}L)")
        
        # Plot bars
        bars = ax.bar(positions, values, color=colors, alpha=0.7)
        
        # Add baseline and final prediction lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Customize plot
        ax.set_ylabel('Impact on Prediction (₹)', fontsize=12)
        ax.set_title('Drivers of Demand - SHAP Waterfall Chart', fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'₹{val/100000:.1f}L',
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add prediction info
        ax.text(0.02, 0.98, f'Baseline: ₹{self.base_value/100000:.2f} Lakhs\nPrediction: ₹{prediction/100000:.2f} Lakhs',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create SHAP summary plot
        
        Args:
            X: Feature DataFrame
            save_path: Optional path to save figure
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, 
                         feature_names=self.feature_names,
                         show=False)
        plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features globally
        
        Args:
            n: Number of top features
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("Calculate SHAP values first using calculate_shap_values()")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(n)


def create_business_impact_message(prediction: float, 
                                   stockout_prob: float,
                                   threshold: float = 1000000) -> str:
    """
    Create business impact message for operations team
    
    Args:
        prediction: Predicted cash demand
        stockout_prob: Probability of stockout
        threshold: Cash threshold for stockout
        
    Returns:
        Business impact message
    """
    prediction_lakhs = prediction / 100000
    threshold_lakhs = threshold / 100000
    
    message = f"**Recommended Load: ₹{prediction_lakhs:.2f} Lakhs**\n\n"
    
    if stockout_prob > 0.4:
        message += f"⚠️ **HIGH RISK**: Loading less than ₹{threshold_lakhs:.1f} Lakhs "
        message += f"carries a {stockout_prob*100:.0f}% risk of stockout by 4 PM.\n\n"
        message += "**Recommendation**: Load the full recommended amount or higher."
    elif stockout_prob > 0.2:
        message += f"⚡ **MEDIUM RISK**: Loading less than ₹{threshold_lakhs:.1f} Lakhs "
        message += f"carries a {stockout_prob*100:.0f}% risk of stockout.\n\n"
        message += "**Recommendation**: Monitor closely and consider loading full amount."
    else:
        message += f"✅ **LOW RISK**: Stockout probability is low ({stockout_prob*100:.0f}%).\n\n"
        message += "**Recommendation**: Proceed with standard loading procedures."
    
    return message


if __name__ == "__main__":
    # Example usage
    print("Explainability module loaded successfully")
    print("Use ATMExplainer class to generate explanations for predictions")
