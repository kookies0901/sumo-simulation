#!/usr/bin/env python3
"""
展示几个典型的3D多目标优化图表
用于演示可视化效果和发现关键洞察
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

def load_and_analyze_data():
    """加载数据并进行快速分析"""
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv"
    df = pd.read_csv(data_file)
    
    print("🔍 数据快速分析:")
    print(f"   样本数量: {len(df)}")
    
    # 分析关键变量的分布
    key_vars = ['coverage_ratio', 'gini_coefficient', 'cluster_count', 
                'avg_pairwise_distance', 'max_gap_distance']
    
    for var in key_vars:
        if var in df.columns:
            print(f"   {var}: [{df[var].min():.3f}, {df[var].max():.3f}], 均值={df[var].mean():.3f}")
    
    return df

def create_showcase_plot(df, x_col, y_col, z_col, title, save_path, colormap='viridis'):
    """创建展示用的高质量3D图表"""
    
    # 数据预处理
    mask = ~(np.isnan(df[x_col]) | np.isnan(df[y_col]) | np.isnan(df[z_col]))
    x_data = df[x_col][mask].values
    y_data = df[y_col][mask].values
    z_data = df[z_col][mask].values
    
    # 创建高密度网格
    grid_size = 60
    xi = np.linspace(x_data.min(), x_data.max(), grid_size)
    yi = np.linspace(y_data.min(), y_data.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 插值生成平滑曲面
    Zi = griddata((x_data, y_data), z_data, (Xi, Yi), method='linear')
    mask_nan = np.isnan(Zi)
    if np.any(mask_nan):
        Zi_nearest = griddata((x_data, y_data), z_data, (Xi, Yi), method='nearest')
        Zi[mask_nan] = Zi_nearest[mask_nan]
    
    # 创建超高质量图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 主曲面 - 使用更精细的渲染
    surf = ax.plot_surface(Xi, Yi, Zi, 
                          cmap=colormap, 
                          alpha=0.85,
                          linewidth=0,
                          antialiased=True,
                          shade=True,
                          rcount=50, ccount=50)
    
    # 添加精细网格线
    wireframe = ax.plot_wireframe(Xi, Yi, Zi, 
                                 linewidth=0.3, 
                                 alpha=0.4, 
                                 color='white')
    
    # 数据点 - 更精美的样式
    scatter = ax.scatter(x_data, y_data, z_data, 
                        c=z_data, cmap=colormap, 
                        s=80, alpha=0.95, 
                        edgecolors='black', linewidth=0.8,
                        depthshade=True)
    
    # 美化坐标轴标签
    ax.set_xlabel(f'{x_col.replace("_", " ").title()}', 
                  fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel(f'{y_col.replace("_", " ").title()}', 
                  fontsize=13, fontweight='bold', labelpad=12)
    ax.set_zlabel(f'{z_col.replace("_", " ").title()}', 
                  fontsize=13, fontweight='bold', labelpad=12)
    
    # 标题设置
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
    
    # 设置最佳视角
    ax.view_init(elev=25, azim=50)
    
    # 精美的颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=25, pad=0.15)
    cbar.set_label(f'{z_col.replace("_", " ").title()}', 
                   fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # 网格和背景美化
    ax.grid(True, alpha=0.2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 设置坐标轴平面颜色
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    
    # 设置刻度样式
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    
    plt.tight_layout()
    
    # 保存超高清图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none',
               transparent=False)
    
    print(f"✨ 生成展示图表: {os.path.basename(save_path)}")
    plt.close()

def create_analysis_summary(df, output_dir):
    """创建分析洞察总结"""
    
    print("\n📊 生成关键洞察分析...")
    
    # 分析各组合的关键发现
    insights = {
        "combo1_insights": {
            "title": "效率-公平权衡洞察",
            "findings": [
                f"覆盖率范围: {df['coverage_ratio'].min():.3f} - {df['coverage_ratio'].max():.3f}",
                f"基尼系数范围: {df['gini_coefficient'].min():.3f} - {df['gini_coefficient'].max():.3f}",
                f"覆盖率与基尼系数相关性: {df['coverage_ratio'].corr(df['gini_coefficient']):.3f}",
                "高覆盖率是否总是伴随低公平性？",
                "存在帕累托最优解吗？"
            ]
        },
        "combo2_insights": {
            "title": "布局模式洞察", 
            "findings": [
                f"聚类数范围: {df['cluster_count'].min():.0f} - {df['cluster_count'].max():.0f}",
                f"平均距离范围: {df['avg_pairwise_distance'].min():.0f} - {df['avg_pairwise_distance'].max():.0f}米",
                f"聚类数与距离相关性: {df['cluster_count'].corr(df['avg_pairwise_distance']):.3f}",
                "集中布局 vs 分散布局的性能差异？",
                "最优聚类数是多少？"
            ]
        },
        "combo3_insights": {
            "title": "极值优化洞察",
            "findings": [
                f"最大间隙范围: {df['max_gap_distance'].min():.0f} - {df['max_gap_distance'].max():.0f}米",
                f"覆盖率范围: {df['coverage_ratio'].min():.3f} - {df['coverage_ratio'].max():.3f}",
                f"间隙与覆盖率相关性: {df['max_gap_distance'].corr(df['coverage_ratio']):.3f}",
                "高覆盖率能否保证低最大间隙？",
                "如何平衡整体性能与极端情况？"
            ]
        }
    }
    
    # 保存洞察分析
    insights_file = os.path.join(output_dir, "key_insights.md")
    with open(insights_file, 'w', encoding='utf-8') as f:
        f.write("# 多目标优化3D分析关键洞察\n\n")
        
        for combo, data in insights.items():
            f.write(f"## {data['title']}\n\n")
            for finding in data['findings']:
                f.write(f"- {finding}\n")
            f.write("\n")
        
        f.write("## 可视化特征说明\n\n")
        f.write("### 视觉设计原则\n")
        f.write("- **3D透视角度**: 25度仰角, 50度方位角，最佳展示效果\n")
        f.write("- **平滑曲面渲染**: 高密度网格插值(60×60)，连续渐变效果\n")
        f.write("- **透明网格线**: 白色半透明网格，增强空间层次感\n")
        f.write("- **高对比色图**: viridis/plasma/coolwarm等科学色彩映射\n")
        f.write("- **数据点标注**: 黑边散点，深度阴影，突出原始数据\n\n")
        
        f.write("### 分析价值\n")
        f.write("1. **约束关系可视化**: 直观展示变量间的制约和冲突\n")
        f.write("2. **最优解识别**: 发现帕累托前沿和最优操作区域\n")
        f.write("3. **权衡量化**: 精确测量不同目标间的trade-off比例\n")
        f.write("4. **设计指导**: 为实际充电桩布局提供科学依据\n")

def main():
    print("🎨 创建多目标优化3D可视化展示")
    
    # 加载数据
    df = load_and_analyze_data()
    
    # 创建展示目录
    showcase_dir = "/home/ubuntu/project/MSC/Msc_Project/models/showcase_3d_examples"
    os.makedirs(showcase_dir, exist_ok=True)
    
    print(f"\n✨ 生成精选展示图表...")
    
    # 展示图表1: 效率-公平权衡 (最经典)
    create_showcase_plot(
        df, 'coverage_ratio', 'gini_coefficient', 'waiting_time_mean',
        'Multi-objective Trade-off: Coverage vs Fairness vs Waiting Time',
        os.path.join(showcase_dir, 'showcase_efficiency_fairness.png'),
        'viridis'
    )
    
    # 展示图表2: 布局模式分析 (最直观)
    create_showcase_plot(
        df, 'cluster_count', 'avg_pairwise_distance', 'duration_mean',
        'Layout Pattern Analysis: Clusters vs Distance vs Duration',
        os.path.join(showcase_dir, 'showcase_layout_patterns.png'),
        'plasma'
    )
    
    # 展示图表3: 极值优化 (最实用)
    create_showcase_plot(
        df, 'max_gap_distance', 'coverage_ratio', 'duration_p90',
        'Extremes Optimization: Max Gap vs Coverage vs 90th Percentile',
        os.path.join(showcase_dir, 'showcase_extremes_optimization.png'),
        'coolwarm'
    )
    
    # 展示图表4: 能源分布公平性
    create_showcase_plot(
        df, 'coverage_ratio', 'gini_coefficient', 'energy_gini',
        'Energy Distribution Fairness: Coverage vs Service Equity vs Energy Equity',
        os.path.join(showcase_dir, 'showcase_energy_fairness.png'),
        'RdYlBu_r'
    )
    
    # 创建分析总结
    create_analysis_summary(df, showcase_dir)
    
    print(f"\n🎉 展示图表创建完成！")
    print(f"📁 保存位置: {showcase_dir}")
    print(f"✨ 特色: 超高清渲染 + 精美配色 + 科学可视化")
    print(f"🎯 适用: 论文展示 + 学术汇报 + 政策建议")

if __name__ == "__main__":
    main()
