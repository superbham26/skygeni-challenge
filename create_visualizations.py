"""
Visualization Script - Create Executive Dashboard
==================================================

This creates professional visualizations for the CRO to understand:
1. Win rate trends over time
2. Deal velocity analysis
3. Forecast scenarios with confidence intervals
4. Risk segmentation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def create_win_rate_trend_chart(monthly_stats_path):
    """Create win rate trend visualization"""
    
    df = pd.read_csv(monthly_stats_path)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Convert to datetime for plotting
    df['month'] = pd.to_datetime(df['closed_year_month'])
    
    # Chart 1: Win Rate Trend
    ax1.plot(df['month'], df['win_rate'] * 100, marker='o', linewidth=2.5, 
             markersize=8, color='#2E86AB', label='Win Rate')
    
    # Add Q4 2023 average line
    q4_2023 = df[(df['month'] >= '2023-10') & (df['month'] <= '2023-12')]['win_rate'].mean()
    ax1.axhline(y=q4_2023 * 100, color='green', linestyle='--', linewidth=2, 
                label=f'Q4 2023 Avg ({q4_2023*100:.1f}%)', alpha=0.7)
    
    # Add Q1 2024 average line
    q1_2024 = df[(df['month'] >= '2024-01') & (df['month'] <= '2024-03')]['win_rate'].mean()
    ax1.axhline(y=q1_2024 * 100, color='red', linestyle='--', linewidth=2,
                label=f'Q1 2024 Avg ({q1_2024*100:.1f}%)', alpha=0.7)
    
    ax1.set_title('Win Rate Trend - The Problem is Clear', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Annotate the drop
    march_2024_idx = df[df['month'] == '2024-07'].index[0]
    march_wr = df.loc[march_2024_idx, 'win_rate'] * 100
    ax1.annotate(' July 2024\n Win Rate',
                xy=(df.loc[march_2024_idx, 'month'], march_wr),
                xytext=(df.loc[march_2024_idx, 'month'], march_wr - 5),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2))
    
    # Chart 2: Deal Volume vs Revenue
    ax2_twin = ax2.twinx()
    
    ax2.bar(df['month'], df['total_deals'], alpha=0.6, color='#A23B72', label='Deal Volume')
    ax2_twin.plot(df['month'], df['total_revenue'] / 1e6, marker='s', linewidth=2.5,
                  markersize=8, color='#F18F01', label='Revenue ($M)')
    
    ax2.set_title('Deal Volume vs Revenue - Volume is Healthy, Conversion is Not', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Number of Deals', fontsize=12, color='#A23B72')
    ax2_twin.set_ylabel('Revenue ($M)', fontsize=12, color='#F18F01')
    
    ax2.legend(loc='upper left', fontsize=11)
    ax2_twin.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./outputs/viz_1_win_rate_trends.png', dpi=300, bbox_inches='tight')
    print(" Saved: viz_1_win_rate_trends.png")
    plt.close()

def create_forecast_scenario_chart(forecast_path, scenarios_path):
    """Create forecast scenario visualization"""
    
    forecast = pd.read_csv(forecast_path)
    
    with open(scenarios_path, 'r') as f:
        import json
        scenarios = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Scenario Waterfall
    scenario_names = ['P10\n(Conservative)', 'P50\n(Expected)', 'P90\n(Optimistic)']
    scenario_values = [
        scenarios['P10_Conservative'] / 1e6,
        scenarios['P50_Expected'] / 1e6,
        scenarios['P90_Optimistic'] / 1e6
    ]
    colors = ['#D62828', '#F77F00', '#06A77D']
    
    bars = ax1.bar(scenario_names, scenario_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add target line (adjusted for smaller forecast period)
    target = 5  # $5M for June-July combined
    ax1.axhline(y=target, color='black', linestyle='--', linewidth=2, label='Target ($5M)')
    
    # Add value labels
    for bar, value in zip(bars, scenario_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${value:.1f}M',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax1.set_title('Revenue Forecast Scenarios - June-July 2024', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Revenue ($M)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add gap annotation
    gap = target - scenario_values[1]
    if gap > 0:
        ax1.annotate(f'Gap to Target:\n${gap:.1f}M',
                    xy=(1, scenario_values[1]), xytext=(1.5, (target + scenario_values[1])/2),
                    fontsize=12, ha='center',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='red', lw=2))
    
    # Chart 2: Win Probability Distribution
    risk_counts = forecast['risk_level'].value_counts()
    risk_revenue = forecast.groupby('risk_level')['deal_amount'].sum() / 1e6
    
    x = np.arange(len(risk_counts))
    width = 0.35
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, risk_counts.values, width, label='Deal Count',
                    color='#264653', alpha=0.8)
    bars2 = ax2_twin.bar(x + width/2, risk_revenue.values, width, label='Revenue ($M)',
                        color='#E76F51', alpha=0.8)
    
    ax2.set_title('Deal Risk Segmentation', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Risk Level', fontsize=12)
    ax2.set_ylabel('Number of Deals', fontsize=12, color='#264653')
    ax2_twin.set_ylabel('Total Revenue ($M)', fontsize=12, color='#E76F51')
    ax2.set_xticks(x)
    ax2.set_xticklabels(risk_counts.index)
    
    ax2.legend(loc='upper left', fontsize=11)
    ax2_twin.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('./outputs/viz_2_forecast_scenarios.png', dpi=300, bbox_inches='tight')
    print(" Saved: viz_2_forecast_scenarios.png")
    plt.close()

def create_channel_performance_chart(processed_data_path):
    """Create channel performance analysis"""
    
    df = pd.read_csv(processed_data_path, parse_dates=['created_date', 'closed_date'])
    
    # Filter to recent data
    recent = df[df['closed_year_month'] >= '2023-10']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Win Rate by Lead Source Over Time
    pivot = recent.pivot_table(
        values='won',
        index='closed_year_month',
        columns='lead_source',
        aggfunc='mean'
    ) * 100
    
    for column in pivot.columns:
        ax1.plot(pivot.index, pivot[column], marker='o', linewidth=2.5, 
                markersize=7, label=column)
    
    ax1.set_title('Win Rate by Lead Source - Outbound Declining', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.legend(fontsize=11, title='Lead Source')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Chart 2: Product Performance
    product_perf = recent.groupby(['product_type', 'closed_year_month'])['won'].mean().unstack()
    
    x = np.arange(len(product_perf.columns))
    width = 0.25
    
    for i, product in enumerate(product_perf.index):
        ax2.bar(x + i*width, product_perf.loc[product] * 100, width, 
               label=product, alpha=0.8)
    
    ax2.set_title('Win Rate by Product Type - Enterprise Struggling', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(product_perf.columns, rotation=45)
    ax2.legend(fontsize=11, title='Product Type')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight July Enterprise
    march_idx = list(product_perf.columns).index('2024-07')
    enterprise_idx = list(product_perf.index).index('Enterprise')
    march_enterprise_wr = product_perf.loc['Enterprise', '2024-07'] * 100
    
    ax2.annotate(f' Only {march_enterprise_wr:.1f}%',
                xy=(march_idx + enterprise_idx*width, march_enterprise_wr),
                xytext=(march_idx + enterprise_idx*width, march_enterprise_wr + 5),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    plt.savefig('./outputs/viz_3_channel_product_performance.png', dpi=300, bbox_inches='tight')
    print(" Saved: viz_3_channel_product_performance.png")
    plt.close()

def main():
    """Generate all visualizations"""
    
    print("\n" + "="*80)
    print("GENERATING EXECUTIVE DASHBOARD VISUALIZATIONS")
    print("="*80 + "\n")
    
    create_win_rate_trend_chart('outputs/monthly_performance.csv')
    create_forecast_scenario_chart('outputs/revenue_forecast_june_july.csv',
                                   'outputs/forecast_scenarios.json')
    create_channel_performance_chart('outputs/processed_sales_data.csv')
    
    print("\n" + "="*80)
    print(" All visualizations created!")
    print("="*80)
    print("\nFiles created:")
    print("  1. viz_1_win_rate_trends.png")
    print("  2. viz_2_forecast_scenarios.png")
    print("  3. viz_3_channel_product_performance.png")

if __name__ == "__main__":
    main()
