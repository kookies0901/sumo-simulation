#!/usr/bin/env python3
"""
多目标优化的3D可视化分析
生成三个推荐组合的多维曲面图，展示变量间的约束关系和权衡
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置全局样式
plt.style.use('default')
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

def load_data(data_file):
    """加载数据集"""
    try:
        df = pd.read_csv(data_file)
        print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
        return df
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def create_3d_surface_plot(df, x_col, y_col, z_col, title="3D Surface Plot", 
                          colormap='viridis', save_path=None, figsize=(12, 9)):
    """
    创建美观的3D曲面图
    """
    # 提取数据并移除NaN值
    mask = ~(np.isnan(df[x_col]) | np.isnan(df[y_col]) | np.isnan(df[z_col]))
    x_data = df[x_col][mask].values
    y_data = df[y_col][mask].values
    z_data = df[z_col][mask].values
    
    if len(x_data) < 10:
        print(f"⚠️ 数据点太少 ({len(x_data)})，跳过 {title}")
        return None
    
    # 创建网格插值
    grid_size = 50
    xi = np.linspace(x_data.min(), x_data.max(), grid_size)
    yi = np.linspace(y_data.min(), y_data.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 使用线性插值生成平滑曲面
    try:
        Zi = griddata((x_data, y_data), z_data, (Xi, Yi), method='linear')
        # 填充NaN值
        mask_nan = np.isnan(Zi)
        if np.any(mask_nan):
            Zi_cubic = griddata((x_data, y_data), z_data, (Xi, Yi), method='nearest')
            Zi[mask_nan] = Zi_cubic[mask_nan]
    except Exception as e:
        print(f"⚠️ 插值失败: {e}")
        return None
    
    # 创建3D图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D曲面
    surf = ax.plot_surface(Xi, Yi, Zi, 
                          cmap=colormap, 
                          alpha=0.8,
                          linewidth=0,
                          antialiased=True,
                          shade=True)
    
    # 添加网格线效果
    ax.plot_wireframe(Xi, Yi, Zi, 
                     linewidth=0.5, 
                     alpha=0.3, 
                     color='white')
    
    # 绘制原始数据点
    scatter = ax.scatter(x_data, y_data, z_data, 
                        c=z_data, cmap=colormap, 
                        s=60, alpha=0.9, 
                        edgecolors='black', linewidth=0.5)
    
    # 设置标签和标题
    ax.set_xlabel(f'{x_col}', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'{y_col}', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel(f'{z_col}', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label(f'{z_col}', fontsize=11, fontweight='bold')
    
    # 美化网格
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 设置网格线颜色
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📊 保存图片: {os.path.basename(save_path)}")
    
    plt.close()
    return fig

def generate_combination_1_plots(df, output_dir):
    """
    组合1: 效率-公平权衡
    coverage_ratio × gini_coefficient × 性能指标
    """
    print("\n🎨 生成组合1: 效率-公平权衡 (coverage_ratio × gini_coefficient)")
    
    combo1_dir = os.path.join(output_dir, "combo1_efficiency_fairness")
    os.makedirs(combo1_dir, exist_ok=True)
    
    # 所有性能指标
    performance_metrics = [
        'duration_mean', 'duration_median', 'duration_p90',
        'charging_time_mean', 'charging_time_median', 'charging_time_p90',
        'waiting_time_mean', 'waiting_time_median', 'waiting_time_p90',
        'energy_gini', 'energy_cv', 'energy_hhi', 'energy_p90_p50_ratio',
        'vehicle_gini', 'vehicle_cv', 'vehicle_hhi',
        'charging_station_coverage', 'reroute_count', 
        'ev_charging_participation_rate', 'ev_charging_failures'
    ]
    
    x_col = 'coverage_ratio'
    y_col = 'gini_coefficient'
    
    # 检查列是否存在
    if x_col not in df.columns or y_col not in df.columns:
        print(f"❌ 缺少必要列: {x_col}, {y_col}")
        return
    
    colormaps = ['viridis', 'plasma', 'coolwarm', 'RdYlBu_r', 'Spectral']
    
    for i, z_col in enumerate(performance_metrics):
        if z_col not in df.columns:
            print(f"⚠️ 跳过缺失列: {z_col}")
            continue
        
        colormap = colormaps[i % len(colormaps)]
        
        title = f'Efficiency-Fairness Trade-off\nCoverage vs Gini vs {z_col.replace("_", " ").title()}'
        save_path = os.path.join(combo1_dir, f'combo1_{z_col}_3d.png')
        
        create_3d_surface_plot(df, x_col, y_col, z_col, 
                             title=title, colormap=colormap, 
                             save_path=save_path)

def generate_combination_2_plots(df, output_dir):
    """
    组合2: 布局模式分析
    cluster_count × avg_pairwise_distance × 性能指标
    """
    print("\n🎨 生成组合2: 布局模式分析 (cluster_count × avg_pairwise_distance)")
    
    combo2_dir = os.path.join(output_dir, "combo2_layout_patterns")
    os.makedirs(combo2_dir, exist_ok=True)
    
    performance_metrics = [
        'duration_mean', 'duration_median', 'duration_p90',
        'charging_time_mean', 'charging_time_median', 'charging_time_p90',
        'waiting_time_mean', 'waiting_time_median', 'waiting_time_p90',
        'energy_gini', 'energy_cv', 'energy_hhi', 'energy_p90_p50_ratio',
        'vehicle_gini', 'vehicle_cv', 'vehicle_hhi',
        'charging_station_coverage', 'reroute_count', 
        'ev_charging_participation_rate', 'ev_charging_failures'
    ]
    
    x_col = 'cluster_count'
    y_col = 'avg_pairwise_distance'
    
    # 检查列是否存在
    if x_col not in df.columns or y_col not in df.columns:
        print(f"❌ 缺少必要列: {x_col}, {y_col}")
        return
    
    colormaps = ['plasma', 'viridis', 'inferno', 'magma', 'cividis']
    
    for i, z_col in enumerate(performance_metrics):
        if z_col not in df.columns:
            print(f"⚠️ 跳过缺失列: {z_col}")
            continue
        
        colormap = colormaps[i % len(colormaps)]
        
        title = f'Layout Pattern Analysis\nCluster Count vs Avg Distance vs {z_col.replace("_", " ").title()}'
        save_path = os.path.join(combo2_dir, f'combo2_{z_col}_3d.png')
        
        create_3d_surface_plot(df, x_col, y_col, z_col, 
                             title=title, colormap=colormap, 
                             save_path=save_path)

def generate_combination_3_plots(df, output_dir):
    """
    组合3: 极值优化
    max_gap_distance × coverage_ratio × 性能指标
    """
    print("\n🎨 生成组合3: 极值优化 (max_gap_distance × coverage_ratio)")
    
    combo3_dir = os.path.join(output_dir, "combo3_extremes_optimization")
    os.makedirs(combo3_dir, exist_ok=True)
    
    performance_metrics = [
        'duration_mean', 'duration_median', 'duration_p90',
        'charging_time_mean', 'charging_time_median', 'charging_time_p90',
        'waiting_time_mean', 'waiting_time_median', 'waiting_time_p90',
        'energy_gini', 'energy_cv', 'energy_hhi', 'energy_p90_p50_ratio',
        'vehicle_gini', 'vehicle_cv', 'vehicle_hhi',
        'charging_station_coverage', 'reroute_count', 
        'ev_charging_participation_rate', 'ev_charging_failures'
    ]
    
    x_col = 'max_gap_distance'
    y_col = 'coverage_ratio'
    
    # 检查列是否存在
    if x_col not in df.columns or y_col not in df.columns:
        print(f"❌ 缺少必要列: {x_col}, {y_col}")
        return
    
    colormaps = ['coolwarm', 'RdYlBu_r', 'seismic', 'bwr', 'PiYG']
    
    for i, z_col in enumerate(performance_metrics):
        if z_col not in df.columns:
            print(f"⚠️ 跳过缺失列: {z_col}")
            continue
        
        colormap = colormaps[i % len(colormaps)]
        
        title = f'Extremes Optimization\nMax Gap vs Coverage vs {z_col.replace("_", " ").title()}'
        save_path = os.path.join(combo3_dir, f'combo3_{z_col}_3d.png')
        
        create_3d_surface_plot(df, x_col, y_col, z_col, 
                             title=title, colormap=colormap, 
                             save_path=save_path)

def create_summary_analysis(df, output_dir):
    """创建组合分析总结"""
    print("\n📊 创建组合分析总结...")
    
    summary_dir = os.path.join(output_dir, "summary_analysis")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 相关性分析
    key_variables = [
        'coverage_ratio', 'gini_coefficient', 'cluster_count', 
        'avg_pairwise_distance', 'max_gap_distance',
        'duration_mean', 'waiting_time_mean', 'charging_time_mean'
    ]
    
    # 过滤存在的列
    available_vars = [col for col in key_variables if col in df.columns]
    
    if len(available_vars) >= 4:
        # 相关性矩阵热图
        plt.figure(figsize=(12, 10))
        corr_matrix = df[available_vars].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Multi-objective Variables Correlation Matrix', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'correlation_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 创建组合说明文档
    readme_content = """# 多目标优化3D可视化分析结果

## 三个核心组合

### 组合1: 效率-公平权衡 (combo1_efficiency_fairness)
- **X轴**: coverage_ratio (覆盖率)
- **Y轴**: gini_coefficient (基尼系数)
- **Z轴**: 各种性能指标
- **核心问题**: 如何在服务效率和社会公平间找到平衡？
- **预期发现**: 帕累托前沿，最优权衡点

### 组合2: 布局模式分析 (combo2_layout_patterns)
- **X轴**: cluster_count (聚类数量)
- **Y轴**: avg_pairwise_distance (平均两两距离)
- **Z轴**: 各种性能指标
- **核心问题**: 集中 vs 分散布局的优劣比较？
- **预期发现**: 不同空间布局模式的性能差异

### 组合3: 极值优化 (combo3_extremes_optimization)
- **X轴**: max_gap_distance (最大间隙距离)
- **Y轴**: coverage_ratio (覆盖率)
- **Z轴**: 各种性能指标
- **核心问题**: 如何平衡整体性能与最坏情况？
- **预期发现**: 风险管理角度的布局优化策略

## 可视化特征
- **3D透视**: 倾斜视角展示三变量关系
- **平滑曲面**: 渐变色填充，连续平滑效果
- **网格线**: 透明网格增加空间感
- **高对比色图**: viridis/plasma/coolwarm等鲜艳色彩

## 分析价值
1. **约束关系识别**: 发现变量间的制约和冲突
2. **优化方向指导**: 确定多目标优化的方向
3. **权衡分析**: 量化不同目标间的trade-off
4. **设计原则**: 提取充电桩布局的一般规律
"""
    
    with open(os.path.join(summary_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)

def main():
    print("🚀 开始生成多目标优化3D可视化分析")
    
    # 设置路径
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv"
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/models/multiobjective_3d_analysis"
    
    print(f"📊 数据文件: {data_file}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    df = load_data(data_file)
    if df is None:
        return 1
    
    print(f"📋 数据概况:")
    print(f"   总样本数: {len(df)}")
    print(f"   总特征数: {len(df.columns)}")
    
    # 检查关键列
    key_columns = [
        'coverage_ratio', 'gini_coefficient', 'cluster_count', 
        'avg_pairwise_distance', 'max_gap_distance'
    ]
    
    missing_cols = [col for col in key_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 缺少关键列: {missing_cols}")
        print("可用列:", list(df.columns))
    
    # 生成三个组合的图表
    try:
        # 组合1: 效率-公平权衡 (20张图)
        generate_combination_1_plots(df, output_dir)
        
        # 组合2: 布局模式分析 (20张图)
        generate_combination_2_plots(df, output_dir)
        
        # 组合3: 极值优化 (20张图)
        generate_combination_3_plots(df, output_dir)
        
        # 创建总结分析
        create_summary_analysis(df, output_dir)
        
        print(f"\n🎉 多目标优化3D可视化生成完成！")
        print(f"📁 所有图表保存在: {output_dir}")
        print(f"📊 预期生成约60张3D图表")
        print(f"🎨 视觉特征: 3D透视 + 平滑曲面 + 网格线 + 高对比色图")
        
    except Exception as e:
        print(f"❌ 生成过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
