"""
SkyGeni Sales Intelligence Analysis
====================================
Author: Shubham
Date: February 2025

This analysis investigates the declining win rate problem and builds
a revenue forecasting engine with actionable insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PART 1: DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_prepare_data(filepath):
    """Load sales data and perform initial preprocessing"""
    df = pd.read_csv(filepath)
    
    # Convert dates
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['closed_date'] = pd.to_datetime(df['closed_date'])
    
    # Extract time features
    df['created_year'] = df['created_date'].dt.year
    df['created_month'] = df['created_date'].dt.month
    df['created_quarter'] = df['created_date'].dt.quarter
    df['created_year_month'] = df['created_date'].dt.to_period('M')
    
    df['closed_year'] = df['closed_date'].dt.year
    df['closed_month'] = df['closed_date'].dt.month
    df['closed_quarter'] = df['closed_date'].dt.quarter
    df['closed_year_month'] = df['closed_date'].dt.to_period('M')
    
    # Create won flag
    df['won'] = (df['outcome'] == 'Won').astype(int)
    
    return df

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

def analyze_win_rate_trends(df):
    """Analyze win rate trends over time"""
    print("=" * 80)
    print("WIN RATE TREND ANALYSIS")
    print("=" * 80)
    
    # Monthly win rate by closed date
    monthly_stats = df.groupby('closed_year_month').agg({
        'deal_id': 'count',
        'won': ['sum', 'mean'],
        'deal_amount': ['sum', 'mean']
    }).round(4)
    
    monthly_stats.columns = ['total_deals', 'deals_won', 'win_rate', 'total_revenue', 'avg_deal_size']
    monthly_stats = monthly_stats.reset_index()
    monthly_stats['closed_year_month'] = monthly_stats['closed_year_month'].astype(str)
    
    print("\nMonthly Performance Summary:")
    print(monthly_stats.tail(10).to_string(index=False))
    
    return monthly_stats

def calculate_custom_metrics(df):
    """Calculate custom business metrics"""
    print("\n" + "=" * 80)
    print("CUSTOM METRICS DISCOVERY")
    print("=" * 80)
    
    # CUSTOM METRIC 1: Deal Velocity Score
    # Measures how quickly deals move through pipeline relative to their size
    df['velocity_score'] = df['deal_amount'] / (df['sales_cycle_days'] + 1)
    
    print("\n METRIC 1: Deal Velocity Score")
    print("Definition: Revenue generated per day in sales cycle (Deal Amount / Cycle Days)")
    print("Why it matters: Fast-moving deals indicate product-market fit and rep effectiveness")
    
    velocity_by_outcome = df.groupby('outcome')['velocity_score'].agg(['mean', 'median'])
    print(f"\nWon deals velocity: ${velocity_by_outcome.loc['Won', 'median']:.0f}/day")
    print(f"Lost deals velocity: ${velocity_by_outcome.loc['Lost', 'median']:.0f}/day")
    
    # CUSTOM METRIC 2: Pipeline Concentration Risk
    # Measures revenue concentration in specific segments
    print("\n METRIC 2: Revenue Concentration Index")
    print("Definition: Measures how dependent revenue is on top performers/segments")
    
    # Top 20% of reps contribution
    rep_revenue = df[df['won'] == 1].groupby('sales_rep_id')['deal_amount'].sum().sort_values(ascending=False)
    top_20_pct = int(len(rep_revenue) * 0.2)
    concentration = rep_revenue.head(top_20_pct).sum() / rep_revenue.sum()
    
    print(f"Top 20% of reps generate {concentration*100:.1f}% of revenue")
    print(f"Risk Level: {'HIGH' if concentration > 0.6 else 'MODERATE' if concentration > 0.5 else 'LOW'}")
    
    return df

def deep_dive_analysis(df):
    """Perform comprehensive segmentation analysis"""
    print("\n" + "=" * 80)
    print("DEEP DIVE: WIN RATE DRIVERS")
    print("=" * 80)
    
    # Analyze by multiple dimensions
    dimensions = ['product_type', 'lead_source', 'industry', 'region']
    
    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"Analysis by {dim.upper()}")
        print(f"{'='*60}")
        
        segment_perf = df.groupby(dim).agg({
            'deal_id': 'count',
            'won': 'mean',
            'deal_amount': lambda x: x[df.loc[x.index, 'won'] == 1].sum()
        }).round(3)
        
        segment_perf.columns = ['deal_count', 'win_rate', 'total_won_revenue']
        segment_perf = segment_perf.sort_values('win_rate', ascending=False)
        
        print(segment_perf.to_string())
        
        # Identify insights
        best = segment_perf.index[0]
        worst = segment_perf.index[-1]
        print(f"\n Insight: {best} has {segment_perf.loc[best, 'win_rate']*100:.1f}% win rate")
        print(f"  Warning: {worst} has only {segment_perf.loc[worst, 'win_rate']*100:.1f}% win rate")

def time_series_diagnostics(df):
    """Diagnose what changed in recent months"""
    print("\n" + "=" * 80)
    print("TIME SERIES DIAGNOSTICS: WHAT CHANGED?")
    print("=" * 80)
    
    # Compare Q1 2024 vs Q4 2023
    q4_2023 = df[(df['closed_year'] == 2023) & (df['closed_quarter'] == 4)]
    q1_2024 = df[(df['closed_year'] == 2024) & (df['closed_quarter'] == 1)]
    
    print("\nQ4 2023 vs Q1 2024 Comparison:")
    print(f"Q4 2023 Win Rate: {q4_2023['won'].mean()*100:.1f}%")
    print(f"Q1 2024 Win Rate: {q1_2024['won'].mean()*100:.1f}%")
    print(f"Change: {(q1_2024['won'].mean() - q4_2023['won'].mean())*100:.1f} percentage points")
    
    # What shifted in the mix?
    print("\n Mix Shift Analysis:")
    
    for col in ['product_type', 'lead_source', 'industry']:
        q4_mix = q4_2023[col].value_counts(normalize=True)
        q1_mix = q1_2024[col].value_counts(normalize=True)
        
        # Find biggest shifts
        mix_change = (q1_mix - q4_mix).abs().sort_values(ascending=False)
        if len(mix_change) > 0:
            biggest_shift = mix_change.index[0]
            print(f"\n{col}: Biggest shift in '{biggest_shift}'")
            print(f"  Q4 2023: {q4_mix.get(biggest_shift, 0)*100:.1f}%")
            print(f"  Q1 2024: {q1_mix.get(biggest_shift, 0)*100:.1f}%")

# ============================================================================
# PART 3: KEY BUSINESS INSIGHTS
# ============================================================================

def generate_key_insights(df):
    """Generate 3 meaningful business insights"""
    print("\n" + "=" * 80)
    print(" KEY BUSINESS INSIGHTS")
    print("=" * 80)
    
    # INSIGHT 1: Sales Cycle Length Impact
    print("\n" + "="*60)
    print("INSIGHT #1: Deal Complexity is Killing Win Rates")
    print("="*60)
    
    # Recent months
    recent = df[df['closed_year_month'] >= '2024-01']
    older = df[df['closed_year_month'] < '2024-01']
    
    recent_avg_cycle = recent['sales_cycle_days'].mean()
    older_avg_cycle = older['sales_cycle_days'].mean()
    
    print(f"\nAverage Sales Cycle:")
    print(f"  Before 2024: {older_avg_cycle:.1f} days")
    print(f"  2024 onwards: {recent_avg_cycle:.1f} days")
    print(f"  Change: +{recent_avg_cycle - older_avg_cycle:.1f} days ({((recent_avg_cycle/older_avg_cycle - 1)*100):.1f}%)")
    
    # Correlation with win rate
    cycle_buckets = pd.cut(df['sales_cycle_days'], bins=[0, 30, 60, 90, 1000], 
                           labels=['<30 days', '30-60 days', '60-90 days', '>90 days'])
    cycle_win_rate = df.groupby(cycle_buckets)['won'].mean()
    
    print("\nWin Rate by Sales Cycle Length:")
    for bucket, wr in cycle_win_rate.items():
        print(f"  {bucket}: {wr*100:.1f}%")
    
    print("\n Business Action:")
    print("   → Implement deal velocity checkpoints at 30 and 60 days")
    print("   → Train reps on objection handling to shorten cycles")
    print("   → Consider qualifying out low-intent leads earlier")
    
    # INSIGHT 2: Lead Source Performance Divergence
    print("\n" + "="*60)
    print("INSIGHT #2: Outbound Efficiency Has Collapsed")
    print("="*60)
    
    source_trend = df.groupby(['closed_year_month', 'lead_source'])['won'].mean().unstack()
    
    print("\nWin Rate by Lead Source (Recent 6 Months):")
    print(source_trend.tail(6).round(3).to_string())
    
    # Calculate trend
    if 'Outbound' in source_trend.columns:
        outbound_q4 = source_trend['Outbound']['2023-10':'2023-12'].mean()
        outbound_q1 = source_trend['Outbound']['2024-01':'2024-03'].mean()
        print(f"\nOutbound Performance Change:")
        print(f"  Q4 2023: {outbound_q4*100:.1f}%")
        print(f"  Q1 2024: {outbound_q1*100:.1f}%")
        print(f"  Decline: {(outbound_q1 - outbound_q4)*100:.1f} percentage points")
    
    print("\n Business Action:")
    print("   → Audit outbound targeting and messaging")
    print("   → Shift budget toward higher-performing channels (Referral/Partner)")
    print("   → Implement lead scoring to filter outbound prospects")
    
    # INSIGHT 3: Product-Market Fit by Segment
    print("\n" + "="*60)
    print("INSIGHT #3: Enterprise Product Struggling in Recent Months")
    print("="*60)
    
    product_trend = df.groupby(['closed_year_month', 'product_type']).agg({
        'won': 'mean',
        'deal_id': 'count'
    }).round(3)
    
    print("\nRecent Product Performance:")
    recent_product = product_trend.loc['2024-01':'2024-03']
    print(recent_product.to_string())
    
    print("\n Business Action:")
    print("   → Review Enterprise product positioning and pricing")
    print("   → Conduct win/loss interviews for Enterprise deals")
    print("   → Consider product-market fit testing in each segment")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SKYGENI SALES INTELLIGENCE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data('skygeni_sales_data.csv')
    print(f"Loaded {len(df):,} deals from {df['created_date'].min().date()} to {df['created_date'].max().date()}")
    print(f"Closed dates range: {df['closed_date'].min().date()} to {df['closed_date'].max().date()}")
    
    # Analysis
    monthly_stats = analyze_win_rate_trends(df)
    df = calculate_custom_metrics(df)
    deep_dive_analysis(df)
    time_series_diagnostics(df)
    generate_key_insights(df)
    
    # Save processed data for modeling
    directory_path = "outputs"
    os.makedirs(directory_path, exist_ok=True)
    df.to_csv('./outputs/processed_sales_data.csv', index=False)
    monthly_stats.to_csv('./outputs/monthly_performance.csv', index=False)
    
    print("\n" + "="*80)
    print(" Analysis Complete! Files saved:")
    print("   - processed_sales_data.csv")
    print("   - monthly_performance.csv")
    print("="*80)
