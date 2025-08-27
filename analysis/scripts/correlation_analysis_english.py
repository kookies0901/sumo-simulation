#!/usr/bin/env python3
"""
Performance Correlation Analysis - English Version
Fixed font issues with English-only labels and titles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_correlation_results():
    """Load correlation analysis results"""
    pearson_file = "analysis/charts/correlation/correlation_matrix_pearson.csv"
    spearman_file = "analysis/charts/correlation/correlation_matrix_spearman.csv"
    
    results = {}
    
    if os.path.exists(pearson_file):
        results['pearson'] = pd.read_csv(pearson_file, index_col=0)
        print(f"✅ Loaded Pearson correlation matrix: {results['pearson'].shape}")
    else:
        print("❌ Pearson correlation matrix not found")
    
    if os.path.exists(spearman_file):
        results['spearman'] = pd.read_csv(spearman_file, index_col=0)
        print(f"✅ Loaded Spearman correlation matrix: {results['spearman'].shape}")
    else:
        print("❌ Spearman correlation matrix not found")
    
    if not results:
        print("❌ Please run performance_correlation_analysis.py first")
        return None
    
    return results

def create_english_heatmap(corr_matrix, output_path, method_name, figsize=(16, 14)):
    """Create correlation heatmap with English labels only"""
    # Set matplotlib parameters for better font handling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    sns.heatmap(corr_matrix, 
                annot=True,                    # Show values
                fmt='.3f',                     # Value format
                cmap='RdBu_r',                # Color scheme: Red-White-Blue
                center=0,                      # Center value at 0
                square=True,                   # Square cells
                cbar_kws={"shrink": .8, "label": f"{method_name} Correlation Coefficient"},
                linewidths=0.5,               # Grid line width
                ax=ax)
    
    # Set title and labels in English only
    ax.set_title(f'Dynamic Performance Metrics Correlation Heatmap\n'
                 f'({method_name} Correlation, n=81 layouts)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✅ {method_name} heatmap saved to: {output_path}")
    
    return fig, ax

def create_clustered_english_heatmap(corr_matrix, output_path, method_name, figsize=(18, 16)):
    """Create clustered correlation heatmap with English labels"""
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create clustermap with English labels
    g = sns.clustermap(corr_matrix,
                       annot=True,
                       fmt='.3f',
                       cmap='RdBu_r',
                       center=0,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={"shrink": .8, "label": f"{method_name} Correlation Coefficient"},
                       figsize=figsize,
                       dendrogram_ratio=0.15,
                       cbar_pos=(0.02, 0.83, 0.03, 0.15))
    
    # Set title in English
    g.fig.suptitle(f'Dynamic Performance Metrics Correlation Heatmap with Hierarchical Clustering\n'
                   f'({method_name} Correlation, n=81 layouts)', 
                   fontsize=16, fontweight='bold', y=0.98)
    
    # Rotate labels
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
    
    # Save figure
    g.savefig(output_path, dpi=300, bbox_inches='tight',
              facecolor='white', edgecolor='none')
    print(f"✅ {method_name} clustered heatmap saved to: {output_path}")
    
    return g

def find_top_correlations(corr_matrix, top_n=20):
    """Find strongest correlations"""
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
    
    return correlations[:top_n]

def create_top_correlations_english_plot(top_correlations, output_path):
    """Create top correlations plot with English labels"""
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Prepare data
    correlations = [x['correlation'] for x in top_correlations]
    labels = [f"{x['metric1'][:20]}...\nvs\n{x['metric2'][:20]}..." 
              if len(x['metric1']) > 20 or len(x['metric2']) > 20
              else f"{x['metric1']}\nvs\n{x['metric2']}" 
              for x in top_correlations]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create color mapping
    colors = ['red' if x < 0 else 'blue' for x in correlations]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(correlations)), correlations, color=colors, alpha=0.7)
    
    # Set labels
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    ax.set_title('Strongest Correlations Among Dynamic Performance Metrics (Top 20)\n'
                 'n=81 layouts', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(corr + (0.02 if corr > 0 else -0.02), i, 
                f'{corr:.3f}', 
                va='center', ha='left' if corr > 0 else 'right',
                fontsize=9, fontweight='bold')
    
    # Add legend
    import matplotlib.patches as patches
    legend_elements = [
        patches.Patch(color='blue', alpha=0.7, label='Positive Correlation'),
        patches.Patch(color='red', alpha=0.7, label='Negative Correlation')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✅ English top correlations plot saved to: {output_path}")
    
    plt.close()

def create_categorized_english_heatmap(corr_matrix, output_path):
    """Create categorized correlation heatmap with English labels"""
    # Set matplotlib parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Define metric categories in English
    categories = {
        'Duration Metrics': ['duration_mean', 'duration_median', 'duration_p90'],
        'Charging Time': ['charging_time_mean', 'charging_time_median', 'charging_time_p90'],
        'Waiting Time': ['waiting_time_mean', 'waiting_time_median', 'waiting_time_p90'],
        'Energy Distribution': ['energy_gini', 'energy_cv', 'energy_hhi', 'energy_p90_p50_ratio'],
        'Vehicle Distribution': ['vehicle_gini', 'vehicle_cv', 'vehicle_hhi'],
        'System Indicators': ['charging_station_coverage', 'reroute_count', 
                             'ev_charging_participation_rate', 'ev_charging_failures']
    }
    
    # Reorder correlation matrix
    ordered_metrics = []
    category_boundaries = []
    current_pos = 0
    
    for category, metrics in categories.items():
        available_metrics = [m for m in metrics if m in corr_matrix.columns]
        ordered_metrics.extend(available_metrics)
        current_pos += len(available_metrics)
        category_boundaries.append(current_pos)
    
    # Reorder matrix
    ordered_corr = corr_matrix.loc[ordered_metrics, ordered_metrics]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Generate heatmap
    sns.heatmap(ordered_corr, 
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"},
                linewidths=0.5,
                ax=ax)
    
    # Add category boundary lines
    for boundary in category_boundaries[:-1]:
        ax.axhline(boundary, color='black', linewidth=2)
        ax.axvline(boundary, color='black', linewidth=2)
    
    # Add category labels
    y_positions = []
    prev_boundary = 0
    for i, boundary in enumerate(category_boundaries):
        y_pos = (prev_boundary + boundary) / 2
        y_positions.append(y_pos)
        prev_boundary = boundary
    
    # Add category labels on the right side
    for i, (category, y_pos) in enumerate(zip(categories.keys(), y_positions)):
        ax.text(len(ordered_metrics) + 0.5, y_pos, category, 
               rotation=0, ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Set title
    ax.set_title('Categorized Correlation Heatmap of Dynamic Performance Metrics\n'
                 '(Pearson Correlation, n=81 layouts)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ English categorized heatmap saved to: {output_path}")
    
    plt.close()

def main():
    """Main function"""
    print("📊 Creating English Version of Correlation Analysis")
    print("=" * 50)
    
    # Load correlation matrices
    corr_results = load_correlation_results()
    if corr_results is None:
        return
    
    # Create output directory
    output_dir = "analysis/charts/correlation/english"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate English visualizations
    print("\n🎨 Generating English visualizations...")
    
    generated_files = []
    
    # Generate for each available correlation method
    for method, corr_matrix in corr_results.items():
        method_name = method.capitalize()
        
        # 1. Standard heatmap
        heatmap_path = os.path.join(output_dir, f"correlation_heatmap_{method}.png")
        fig1, ax1 = create_english_heatmap(corr_matrix, heatmap_path, method_name)
        plt.close(fig1)
        generated_files.append(f"correlation_heatmap_{method}.png")
        
        # 2. Clustered heatmap  
        clustered_path = os.path.join(output_dir, f"correlation_heatmap_clustered_{method}.png")
        g = create_clustered_english_heatmap(corr_matrix, clustered_path, method_name)
        plt.close(g.fig)
        generated_files.append(f"correlation_heatmap_clustered_{method}.png")
    
    # Use Pearson for additional visualizations if available
    if 'pearson' in corr_results:
        corr_matrix = corr_results['pearson']
        
        # 3. Top correlations plot
        top_correlations = find_top_correlations(corr_matrix, top_n=20)
        top_corr_path = os.path.join(output_dir, "top_correlations_english.png")
        create_top_correlations_english_plot(top_correlations, top_corr_path)
        generated_files.append("top_correlations_english.png")
        
        # 4. Categorized heatmap
        category_path = os.path.join(output_dir, "categorized_correlation_english.png")
        create_categorized_english_heatmap(corr_matrix, category_path)
        generated_files.append("categorized_correlation_english.png")
        
        print(f"\n🎉 English correlation analysis complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Generated files:")
        for file in generated_files:
            print(f"   • {file}")
        
        # Show top 10 correlations
        print(f"\n🔥 Top 10 Strongest Correlations (Pearson):")
        for i, corr in enumerate(top_correlations[:10], 1):
            corr_type = "Positive" if corr['correlation'] > 0 else "Negative"
            print(f"{i:2d}. {corr['metric1']} ↔ {corr['metric2']}")
            print(f"    Correlation: {corr['correlation']:.3f} ({corr_type})")
    else:
        print(f"\n🎉 English correlation analysis complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Generated files:")
        for file in generated_files:
            print(f"   • {file}")

if __name__ == '__main__':
    main()
