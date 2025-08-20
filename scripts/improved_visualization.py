#!/usr/bin/env python3
"""
改进的可视化脚本 - 解决数据分布不均匀的展示问题
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_improved_scatter_plot(df, x_col, y_col, output_dir):
    """创建改进的散点图，解决分布不均匀问题"""
    
    # 创建2x2的子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x = df[x_col].values
    y = df[y_col].values
    
    # 移除NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # 1. 全数据散点图（左上）
    ax1.scatter(x_clean, y_clean, alpha=0.7, s=80, color='steelblue', 
               edgecolors='black', linewidth=0.5)
    
    # 拟合全数据
    X = x_clean.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y_clean)
    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    r2_full = r2_score(y_clean, model.predict(X))
    
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'R² = {r2_full:.3f}')
    ax1.set_title(f'Full Dataset (N={len(x_clean)})', fontweight='bold')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 中等密度区间（右上）
    # 定义中等密度区间
    x_min_central = np.percentile(x_clean, 25)
    x_max_central = np.percentile(x_clean, 75)
    
    mask_central = (x_clean >= x_min_central) & (x_clean <= x_max_central)
    x_central = x_clean[mask_central]
    y_central = y_clean[mask_central]
    
    ax2.scatter(x_central, y_central, alpha=0.7, s=80, color='green', 
               edgecolors='black', linewidth=0.5)
    
    # 拟合中等密度数据
    if len(x_central) > 3:
        X_central = x_central.reshape(-1, 1)
        model_central = LinearRegression()
        model_central.fit(X_central, y_central)
        x_fit_central = np.linspace(x_central.min(), x_central.max(), 100)
        y_fit_central = model_central.predict(x_fit_central.reshape(-1, 1))
        r2_central = r2_score(y_central, model_central.predict(X_central))
        
        ax2.plot(x_fit_central, y_fit_central, 'r-', linewidth=2, 
                label=f'R² = {r2_central:.3f}')
    
    ax2.set_title(f'Central Density Range (N={len(x_central)})', fontweight='bold')
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 数据分布直方图（左下）
    ax3.hist(x_clean, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x_min_central, color='red', linestyle='--', alpha=0.7, 
               label='Central Range')
    ax3.axvline(x_max_central, color='red', linestyle='--', alpha=0.7)
    ax3.set_title('X-axis Data Distribution', fontweight='bold')
    ax3.set_xlabel(x_col)
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 分区间分析（右下）
    # 分成3个区间
    q33 = np.percentile(x_clean, 33)
    q67 = np.percentile(x_clean, 67)
    
    regions = [
        (x_clean < q33, 'Low', 'blue'),
        ((x_clean >= q33) & (x_clean < q67), 'Medium', 'green'),
        (x_clean >= q67, 'High', 'red')
    ]
    
    r2_regions = []
    for mask_region, name, color in regions:
        x_region = x_clean[mask_region]
        y_region = y_clean[mask_region]
        
        if len(x_region) > 2:
            ax4.scatter(x_region, y_region, alpha=0.7, s=80, color=color, 
                       label=f'{name} (N={len(x_region)})', edgecolors='black', linewidth=0.5)
            
            # 局部拟合
            X_region = x_region.reshape(-1, 1)
            model_region = LinearRegression()
            model_region.fit(X_region, y_region)
            x_fit_region = np.linspace(x_region.min(), x_region.max(), 50)
            y_fit_region = model_region.predict(x_fit_region.reshape(-1, 1))
            r2_region = r2_score(y_region, model_region.predict(X_region))
            r2_regions.append((name, r2_region))
            
            ax4.plot(x_fit_region, y_fit_region, color=color, linewidth=2, alpha=0.7)
    
    ax4.set_title('Regional Analysis', fontweight='bold')
    ax4.set_xlabel(x_col)
    ax4.set_ylabel(y_col)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加总体信息
    fig.suptitle(f'{x_col} vs {y_col} - Multi-perspective Analysis', fontsize=16, fontweight='bold')
    
    # 添加文本说明
    info_text = f"Full Data R²: {r2_full:.3f}\\n"
    if len(x_central) > 3:
        info_text += f"Central Range R²: {r2_central:.3f}\\n"
    for name, r2_val in r2_regions:
        info_text += f"{name} Region R²: {r2_val:.3f}\\n"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存
    filename = f"{x_col}_{y_col}_improved.png"
    filepath = f"{output_dir}/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def main():
    # 加载数据
    df = pd.read_csv("/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv")
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/models/plots_improved"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择几个重要的关系进行改进展示
    important_pairs = [
        ('avg_dist_to_center', 'duration_mean'),
        ('avg_dist_to_center', 'waiting_time_mean'),
        ('gini_coefficient', 'energy_gini'),
        ('cluster_count', 'charging_station_coverage')
    ]
    
    print("🎨 Generating improved visualization charts...")
    
    for x_col, y_col in important_pairs:
        if x_col in df.columns and y_col in df.columns:
            print(f"   📊 {x_col} vs {y_col}")
            create_improved_scatter_plot(df, x_col, y_col, output_dir)
    
    print(f"✅ Improved charts saved to: {output_dir}")
    print("💡 These charts show multi-perspective analysis, solving data distribution issues")

if __name__ == '__main__':
    main()
