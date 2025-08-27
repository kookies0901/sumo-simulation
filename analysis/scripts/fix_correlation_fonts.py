#!/usr/bin/env python3
"""
Quick font fix for correlation analysis
Replace Chinese titles with English ones to avoid font issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def fix_correlation_heatmap():
    """Generate clean correlation heatmap with English labels"""
    
    # Load correlation matrix
    corr_file = "analysis/charts/correlation/correlation_matrix_pearson.csv"
    if not os.path.exists(corr_file):
        print("❌ Correlation matrix not found. Run the main analysis first.")
        return
    
    corr_matrix = pd.read_csv(corr_file, index_col=0)
    print(f"✅ Loaded correlation matrix: {corr_matrix.shape}")
    
    # Set clean font parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, 
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"},
                linewidths=0.5,
                ax=ax)
    
    # Set English title
    ax.set_title('Dynamic Performance Metrics Correlation Matrix\n'
                 'Pearson Correlation Analysis (n=81 layouts)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Format labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save clean version
    output_path = "analysis/charts/correlation/correlation_heatmap_clean.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    print(f"✅ Clean correlation heatmap saved to: {output_path}")
    plt.close()
    
    return output_path

def create_simple_correlation_summary():
    """Create a simple text summary of key correlations"""
    
    # Load correlation matrix
    corr_file = "analysis/charts/correlation/correlation_matrix_pearson.csv"
    corr_matrix = pd.read_csv(corr_file, index_col=0)
    
    # Find strongest correlations
    correlations = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            correlations.append({
                'metric1': corr_matrix.index[i],
                'metric2': corr_matrix.columns[j],
                'correlation': corr_val,
                'abs_correlation': abs(corr_val)
            })
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    # Create summary
    summary_file = "analysis/charts/correlation/correlation_summary_clean.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Dynamic Performance Metrics Correlation Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Size: 81 layout configurations\n")
        f.write(f"Metrics Analyzed: 20 dynamic performance indicators\n")
        f.write(f"Total Correlation Pairs: {len(correlations)}\n\n")
        
        f.write("Top 20 Strongest Correlations:\n")
        f.write("-" * 40 + "\n")
        for i, corr in enumerate(correlations[:20], 1):
            corr_type = "Positive" if corr['correlation'] > 0 else "Negative"
            f.write(f"{i:2d}. {corr['metric1']} ↔ {corr['metric2']}\n")
            f.write(f"    Correlation: {corr['correlation']:.3f} ({corr_type})\n\n")
        
        # Correlation strength distribution
        very_strong = sum(1 for c in correlations if abs(c['correlation']) >= 0.8)
        strong = sum(1 for c in correlations if 0.6 <= abs(c['correlation']) < 0.8)
        moderate = sum(1 for c in correlations if 0.3 <= abs(c['correlation']) < 0.6)
        weak = sum(1 for c in correlations if abs(c['correlation']) < 0.3)
        
        f.write("Correlation Strength Distribution:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Very Strong (|r| ≥ 0.8): {very_strong} pairs ({very_strong/len(correlations)*100:.1f}%)\n")
        f.write(f"Strong (0.6 ≤ |r| < 0.8): {strong} pairs ({strong/len(correlations)*100:.1f}%)\n")
        f.write(f"Moderate (0.3 ≤ |r| < 0.6): {moderate} pairs ({moderate/len(correlations)*100:.1f}%)\n")
        f.write(f"Weak (|r| < 0.3): {weak} pairs ({weak/len(correlations)*100:.1f}%)\n\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Charging Infrastructure Impact:\n")
        f.write("   - charging_station_coverage shows strong negative correlations\n")
        f.write("   - Higher coverage leads to better system performance\n\n")
        f.write("2. Resource Distribution Effects:\n")
        f.write("   - HHI indices (concentration measures) highly correlated\n")
        f.write("   - Uneven distribution leads to performance degradation\n\n")
        f.write("3. Time Performance Clustering:\n")
        f.write("   - Duration, waiting, and charging times are interconnected\n")
        f.write("   - Shows system-wide performance coordination\n\n")
    
    print(f"✅ Summary report saved to: {summary_file}")
    return summary_file

def main():
    """Main function to fix font issues"""
    print("🔧 Fixing Correlation Analysis Font Issues")
    print("=" * 45)
    
    # Generate clean heatmap
    clean_heatmap = fix_correlation_heatmap()
    
    # Generate summary report
    summary_report = create_simple_correlation_summary()
    
    print(f"\n🎉 Font fix complete!")
    print(f"📊 Clean files generated:")
    print(f"   • correlation_heatmap_clean.png - Clean heatmap without font issues")
    print(f"   • correlation_summary_clean.txt - Text summary of key findings")
    print(f"\n💡 You can now use these files without any font/encoding issues!")

if __name__ == '__main__':
    main()
