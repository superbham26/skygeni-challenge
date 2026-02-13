"""
Part 3: Revenue Forecasting Engine
===================================

This module builds a probabilistic revenue forecast that:
1. Predicts expected revenue for next periods
2. Provides confidence intervals (not just point estimates)
3. Identifies which deals are at risk
4. Recommends actions to hit targets

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from datetime import datetime, timedelta
import os

class RevenueForecaster:
    """
    Probabilistic Revenue Forecasting Engine
    
    Unlike traditional forecasting that gives a single number, this provides:
    - P10, P50, P90 revenue scenarios (conservative, expected, optimistic)
    - Deal-level win probabilities
    - Action recommendations to close gaps
    """
    
    def __init__(self):
        self.win_model = None
        self.amount_model = None
        self.label_encoders = {}
        self.feature_cols = []
        
    def prepare_features(self, df, fit=True):
        """Engineer features for prediction"""
        
        # Create copy
        X = df.copy()
        
        # Categorical encoding
        cat_cols = ['sales_rep_id', 'industry', 'region', 'product_type', 'lead_source', 'deal_stage']
        
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                X[f'{col}_encoded'] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                X[f'{col}_encoded'] = X[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Temporal features
        X['created_month'] = X['created_date'].dt.month
        X['created_quarter'] = X['created_date'].dt.quarter
        X['created_day_of_week'] = X['created_date'].dt.dayofweek
        
        # Deal characteristics
        X['deal_amount_log'] = np.log1p(X['deal_amount'])
        X['sales_cycle_days_log'] = np.log1p(X['sales_cycle_days'])
        X['velocity_score'] = X['deal_amount'] / (X['sales_cycle_days'] + 1)
        
        # Historical performance features (this is the magic)
        if fit:
            # Rep historical win rate
            rep_stats = df.groupby('sales_rep_id')['won'].agg(['mean', 'count']).to_dict()
            X['rep_historical_win_rate'] = X['sales_rep_id'].map(rep_stats['mean'])
            X['rep_deal_count'] = X['sales_rep_id'].map(rep_stats['count'])
            
            # Industry win rate
            industry_stats = df.groupby('industry')['won'].mean().to_dict()
            X['industry_win_rate'] = X['industry'].map(industry_stats)
            
            # Store for prediction
            self.rep_stats = rep_stats
            self.industry_stats = industry_stats
        else:
            X['rep_historical_win_rate'] = X['sales_rep_id'].map(self.rep_stats['mean']).fillna(0.45)
            X['rep_deal_count'] = X['sales_rep_id'].map(self.rep_stats['count']).fillna(10)
            X['industry_win_rate'] = X['industry'].map(self.industry_stats).fillna(0.45)
        
        # Fill NAs
        X = X.fillna(0)
        
        # Select feature columns
        feature_cols = [
            'sales_rep_id_encoded', 'industry_encoded', 'region_encoded',
            'product_type_encoded', 'lead_source_encoded', 'deal_stage_encoded',
            'created_month', 'created_quarter', 'created_day_of_week',
            'deal_amount_log', 'sales_cycle_days_log', 'velocity_score',
            'rep_historical_win_rate', 'rep_deal_count', 'industry_win_rate'
        ]
        
        if fit:
            self.feature_cols = feature_cols
        
        return X[feature_cols]
    
    def train(self, df_train):
        """Train win probability and deal value models"""
        
        print("Training revenue forecasting models...")
        
        # Prepare features
        X = self.prepare_features(df_train, fit=True)
        y_win = df_train['won']
        
        # Train win probability model
        print("  → Training win probability model...")
        self.win_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            random_state=42,
            n_jobs=-1
        )
        self.win_model.fit(X, y_win)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.win_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 5 Features Driving Win Probability:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
        
        # Train accuracy
        train_pred = self.win_model.predict(X)
        train_acc = (train_pred == y_win).mean()
        print(f"\n  Training Accuracy: {train_acc*100:.1f}%")
        
        return self
    
    def forecast_pipeline(self, df_pipeline, target_month=None):
        """
        Forecast revenue for open pipeline deals
        
        Returns:
            DataFrame with deal-level predictions and recommendations
        """
        
        print("\n" + "="*80)
        print("GENERATING REVENUE FORECAST")
        print("="*80)
        
        # Prepare features
        X = self.prepare_features(df_pipeline, fit=False)
        
        # Predict win probabilities
        win_proba = self.win_model.predict_proba(X)[:, 1]
        
        # Create results DataFrame
        results = df_pipeline.copy()
        results['win_probability'] = win_proba
        results['expected_revenue'] = results['deal_amount'] * results['win_probability']
        
        # Risk classification
        results['risk_level'] = pd.cut(
            results['win_probability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['High Risk', 'Medium Risk', 'Low Risk']
        )
        
        # Generate recommendations
        results['recommendation'] = results.apply(self._generate_recommendation, axis=1)
        
        return results
    
    def _generate_recommendation(self, row):
        """Generate action recommendation for a deal"""
        
        if row['win_probability'] < 0.3:
            return " HIGH RISK: Schedule executive sponsor call, offer pilot/POC"
        elif row['win_probability'] < 0.6:
            if row['sales_cycle_days'] > 60:
                return "  MEDIUM RISK: Push for decision, address objections, create urgency"
            else:
                return "  MEDIUM RISK: Continue nurturing, share customer success stories"
        else:
            return " ON TRACK: Move to close, send contract, schedule final review"
    
    def scenario_analysis(self, forecast_df):
        """
        Calculate P10, P50, P90 revenue scenarios
        
        P10 = Conservative (only high-confidence deals close)
        P50 = Expected (probabilistic)
        P90 = Optimistic (most deals close)
        """
        
        print("\n" + "="*80)
        print("SCENARIO ANALYSIS")
        print("="*80)
        
        # Sort by win probability
        sorted_deals = forecast_df.sort_values('win_probability', ascending=False)
        
        # P90 - Optimistic: Top 90% of deals by probability
        p90_revenue = sorted_deals.head(int(len(sorted_deals) * 0.9))['deal_amount'].sum()
        
        # P50 - Expected: Probability-weighted
        p50_revenue = sorted_deals['expected_revenue'].sum()
        
        # P10 - Conservative: Only deals with >70% win probability
        p10_revenue = sorted_deals[sorted_deals['win_probability'] > 0.7]['deal_amount'].sum()
        
        scenarios = {
            'P10_Conservative': p10_revenue,
            'P50_Expected': p50_revenue,
            'P90_Optimistic': p90_revenue
        }
        
        print(f"\nRevenue Scenarios for {len(forecast_df)} open deals:")
        print(f"  P90 (Optimistic):  ${p90_revenue:,.0f}")
        print(f"  P50 (Expected):    ${p50_revenue:,.0f}")
        print(f"  P10 (Conservative): ${p10_revenue:,.0f}")
        print(f"\nRange: ${p90_revenue - p10_revenue:,.0f}")
        
        return scenarios
    
    def gap_analysis(self, forecast_df, target_revenue):
        """
        Identify which deals need intervention to hit target
        """
        
        print("\n" + "="*80)
        print("GAP-TO-TARGET ANALYSIS")
        print("="*80)
        
        expected_revenue = forecast_df['expected_revenue'].sum()
        gap = target_revenue - expected_revenue
        
        print(f"\nTarget Revenue: ${target_revenue:,.0f}")
        print(f"Expected Revenue: ${expected_revenue:,.0f}")
        print(f"Gap: ${gap:,.0f} ({gap/target_revenue*100:.1f}%)")
        
        if gap > 0:
            print("\n DEALS TO PRIORITIZE (to close gap):")
            
            # Find deals where small probability increase = big revenue impact
            forecast_df['leverage_score'] = (
                forecast_df['deal_amount'] * (1 - forecast_df['win_probability'])
            )
            
            priority_deals = forecast_df.nlargest(10, 'leverage_score')
            
            for idx, deal in priority_deals.iterrows():
                potential_gain = deal['deal_amount'] * (0.8 - deal['win_probability'])
                if potential_gain > 0:
                    print(f"\n  Deal {deal['deal_id']}: ${deal['deal_amount']:,.0f}")
                    print(f"    Current win probability: {deal['win_probability']*100:.1f}%")
                    print(f"    If improved to 80%: +${potential_gain:,.0f}")
                    print(f"    {deal['recommendation']}")
        else:
            print("\n Forecast exceeds target! Focus on deal velocity.")
        
        return gap

def main():
    """Main execution for revenue forecasting"""
    
    # Load processed data
    print("Loading data...")
    df = pd.read_csv('processed_sales_data.csv', parse_dates=['created_date', 'closed_date'])
    
    # Define train/test split
    # Use deals closed through May 2024 for training
    # Forecast June-July 2024 (simulating pipeline forecast)
    train_mask = df['closed_year_month'] < '2024-06'
    test_mask = df['closed_year_month'].isin(['2024-06', '2024-07'])
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"Training on {len(df_train):,} closed deals (through May 2024)")
    print(f"Forecasting {len(df_test):,} deals (June-July 2024 actuals for validation)")
    
    # Simulate "open pipeline" - remove outcome for testing
    df_pipeline = df_test.copy()
    actual_outcomes = df_pipeline['won'].copy()
    actual_revenue = df_pipeline[df_pipeline['won'] == 1]['deal_amount'].sum()
    
    # Initialize and train forecaster
    forecaster = RevenueForecaster()
    forecaster.train(df_train)
    
    # Generate forecast
    forecast = forecaster.forecast_pipeline(df_pipeline)
    
    # Scenario analysis
    scenarios = forecaster.scenario_analysis(forecast)
    
    # Gap analysis (assume target = 10M for March)
    target = 10_000_000
    gap = forecaster.gap_analysis(forecast, target)
    
    # Validation against actuals
    print("\n" + "="*80)
    print("FORECAST VALIDATION (June-July 2024 Actuals)")
    print("="*80)
    
    print(f"\nActual Revenue: ${actual_revenue:,.0f}")
    print(f"Forecasted P50: ${scenarios['P50_Expected']:,.0f}")
    print(f"Forecast Error: ${scenarios['P50_Expected'] - actual_revenue:,.0f} ({(scenarios['P50_Expected']/actual_revenue - 1)*100:.1f}%)")
    
    # Check if actual fell within scenarios
    if scenarios['P10_Conservative'] <= actual_revenue <= scenarios['P90_Optimistic']:
        print(" Actual revenue fell within predicted range")
    else:
        print(" Actual revenue outside predicted range")
    
    # Save outputs
    directory_path = "outputs"
    os.makedirs(directory_path, exist_ok=True)
    forecast.to_csv('./outputs/revenue_forecast_june_july.csv', index=False)
    
    # Convert numpy types to Python types for JSON
    scenarios_json = {k: int(v) for k, v in scenarios.items()}
    with open('./outputs/forecast_scenarios.json', 'w') as f:
        json.dump(scenarios_json, f, indent=2)
    
    # Save model
    with open('./outputs/forecaster_model.pkl', 'wb') as f:
        pickle.dump(forecaster, f)

    
    print("\n" + "="*80)
    print(" Forecast Complete! Files saved:")
    print("   - revenue_forecast_june_july.csv (deal-level predictions)")
    print("   - forecast_scenarios.json (P10/P50/P90 scenarios)")
    print("   - forecaster_model.pkl (trained model)")
    print("="*80)
    
    # Return high-value deals for CRO attention
    print("\n" + "="*80)
    print(" TOP 10 DEALS FOR CRO ATTENTION")
    print("="*80)
    
    top_deals = forecast.nlargest(10, 'leverage_score')[
        ['deal_id', 'deal_amount', 'win_probability', 'risk_level', 'recommendation']
    ]
    
    print(top_deals.to_string(index=False))

if __name__ == "__main__":
    main()
