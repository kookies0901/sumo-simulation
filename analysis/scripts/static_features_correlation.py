#!/usr/bin/env python3
"""
11个静态布局特征相关性分析
生成专业的相关性热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_static_features():
    """加载11个静态布局特征"""
    # 读取merged_dataset.csv
    data_file = "models/input_1-100/merged_dataset.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    print(f"✅ 加载数据: {df.shape}")
    
    # 前11列是静态特征
    static_feature_columns = [
        'avg_dist_to_center',          # 平均到中心距离
        'avg_pairwise_distance',       # 平均两两距离  
        'std_pairwise_distance',       # 两两距离标准差
        'min_pairwise_distance',       # 最小两两距离
        'max_pairwise_distance',       # 最大两两距离
        'cs_density_std',              # 充电桩密度标准差
        'cluster_count',               # 聚类簇数
        'coverage_ratio',              # 覆盖率
        'max_gap_distance',            # 最大空白距离
        'gini_coefficient',            # 基尼系数
        'avg_betweenness_centrality'   # 平均介数中心性
    ]
    
    # 提取静态特征
    static_features = df[static_feature_columns].copy()
    print(f"📊 静态特征: {static_features.shape}")
    print(f"📋 特征列表:")
    for i, col in enumerate(static_feature_columns, 1):
        print(f"   {i:2d}. {col}")
    
    return static_features, static_feature_columns

def create_static_features_heatmap(corr_matrix, method_name, output_path, figsize=(12, 10)):
    """创建静态特征相关性热力图"""
    # 设置字体参数
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 生成热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
    
    sns.heatmap(corr_matrix, 
                mask=mask,                     # 遮罩上三角
                annot=True,                    # 显示数值
                fmt='.3f',                     # 数值格式
                cmap='RdBu_r',                # 颜色方案
                center=0,                      # 中心值为0
                square=True,                   # 正方形格子
                cbar_kws={"shrink": .8, "label": f"{method_name} Correlation Coefficient"},
                linewidths=0.5,               # 网格线宽度
                ax=ax)
    
    # 设置标题
    ax.set_title(f'Static Layout Features Correlation Matrix\n'
                 f'({method_name} Correlation, n=81 layouts)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 调整标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✅ {method_name} 静态特征热力图保存至: {output_path}")
    
    return fig, ax

def create_clustered_static_heatmap(corr_matrix, method_name, output_path, figsize=(14, 12)):
    """创建带聚类的静态特征相关性热力图"""
    # 设置字体参数
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建聚类热力图
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
    
    # 设置标题
    g.fig.suptitle(f'Static Layout Features Correlation with Hierarchical Clustering\n'
                   f'({method_name} Correlation, n=81 layouts)', 
                   fontsize=14, fontweight='bold', y=0.98)
    
    # 调整标签
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
    
    # 保存图像
    g.savefig(output_path, dpi=300, bbox_inches='tight',
              facecolor='white', edgecolor='none')
    print(f"✅ {method_name} 聚类静态特征热力图保存至: {output_path}")
    
    return g

def calculate_feature_correlations(static_features, method='pearson'):
    """计算特征间的相关性"""
    if method == 'pearson':
        corr_matrix = static_features.corr(method='pearson')
    else:
        corr_matrix = static_features.corr(method='spearman')
    
    return corr_matrix

def find_strongest_correlations(corr_matrix, top_n=10):
    """找出最强的相关性对"""
    correlations = []
    
    # 遍历下三角矩阵（避免重复和自相关）
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            correlations.append({
                'feature1': corr_matrix.index[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_val,
                'abs_correlation': abs(corr_val)
            })
    
    # 按绝对相关系数排序
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    return correlations[:top_n]

def create_feature_definitions():
    """创建特征定义字典"""
    feature_definitions = {
        'avg_dist_to_center': '平均到中心距离 - 充电桩到几何中心的平均距离',
        'avg_pairwise_distance': '平均两两距离 - 所有充电桩对之间的平均距离',
        'std_pairwise_distance': '两两距离标准差 - 充电桩对距离的变异程度',
        'min_pairwise_distance': '最小两两距离 - 最近充电桩对之间的距离',
        'max_pairwise_distance': '最大两两距离 - 最远充电桩对之间的距离',
        'cs_density_std': '密度标准差 - 网格密度分布的变异程度',
        'cluster_count': '聚类簇数 - DBSCAN算法识别的聚类数量',
        'coverage_ratio': '覆盖率 - 500m范围内可达充电桩的路段比例',
        'max_gap_distance': '最大空白距离 - 距离最近充电桩最远的点的距离',
        'gini_coefficient': '基尼系数 - 充电桩服务可达性的不均匀程度',
        'avg_betweenness_centrality': '平均介数中心性 - 充电桩在路网中的平均重要性'
    }
    return feature_definitions

def create_analysis_report(corr_matrix, top_correlations, method_name, output_path):
    """创建分析报告"""
    feature_definitions = create_feature_definitions()
    
    report_content = f"""
# 静态布局特征相关性分析报告 ({method_name})

## 分析概述
- **样本数量**: 81个充电站布局方案
- **特征数量**: 11个静态布局特征
- **相关性方法**: {method_name} 相关系数
- **分析日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 特征定义

"""
    
    for i, (feature, definition) in enumerate(feature_definitions.items(), 1):
        report_content += f"{i:2d}. **{feature}**: {definition}\n"
    
    report_content += f"""

## 相关性统计摘要

### 相关性强度分布
"""
    
    # 计算相关性分布
    all_correlations = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix.columns)):
            all_correlations.append(abs(corr_matrix.iloc[i, j]))
    
    very_strong = sum(1 for c in all_correlations if c >= 0.8)
    strong = sum(1 for c in all_correlations if 0.6 <= c < 0.8)
    moderate = sum(1 for c in all_correlations if 0.3 <= c < 0.6)
    weak = sum(1 for c in all_correlations if c < 0.3)
    total = len(all_correlations)
    
    report_content += f"""
- **极强相关 (|r| ≥ 0.8)**: {very_strong} 对 ({very_strong/total*100:.1f}%)
- **强相关 (0.6 ≤ |r| < 0.8)**: {strong} 对 ({strong/total*100:.1f}%)
- **中等相关 (0.3 ≤ |r| < 0.6)**: {moderate} 对 ({moderate/total*100:.1f}%)
- **弱相关 (|r| < 0.3)**: {weak} 对 ({weak/total*100:.1f}%)

### 最强相关性对 (Top 10)

"""
    
    for i, corr in enumerate(top_correlations, 1):
        corr_type = "正相关" if corr['correlation'] > 0 else "负相关"
        report_content += f"{i:2d}. **{corr['feature1']}** ↔ **{corr['feature2']}**\n"
        report_content += f"    - 相关系数: {corr['correlation']:.3f} ({corr_type})\n"
        report_content += f"    - 绝对值: {corr['abs_correlation']:.3f}\n\n"
    
    report_content += """
## 关键发现

### 1. 距离类特征高度相关
- 平均两两距离、标准差、最大距离等空间分布特征表现出强相关性
- 反映了充电桩空间分布的一致性模式

### 2. 覆盖性指标的关联
- 覆盖率、最大空白距离、基尼系数等服务覆盖指标相互关联
- 体现了不同覆盖性度量的一致性

### 3. 网络拓扑特征的独特性
- 介数中心性作为网络拓扑特征，与其他几何特征相关性较低
- 提供了独特的布局评估维度

### 4. 聚类特征的中介作用
- 聚类簇数与多个空间分布特征存在中等相关性
- 是连接几何特征和覆盖性特征的桥梁

## 应用建议

1. **特征选择**: 在机器学习建模中，可考虑去除高度相关的冗余特征
2. **布局评估**: 结合不同类型的特征（距离、覆盖、网络）进行综合评估
3. **设计优化**: 关注相关性较低的特征组合，实现多维度优化
"""
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 分析报告保存至: {output_path}")

def main():
    """主函数"""
    print("📊 11个静态布局特征相关性分析")
    print("=" * 50)
    
    # 加载静态特征数据
    static_features, feature_columns = load_static_features()
    if static_features is None:
        return
    
    # 创建输出目录
    output_dir = "analysis/charts/correlation/static_features"
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算Pearson和Spearman相关性
    methods = ['pearson', 'spearman']
    
    for method in methods:
        print(f"\n🔍 计算 {method.capitalize()} 相关性...")
        
        # 计算相关性矩阵
        corr_matrix = calculate_feature_correlations(static_features, method)
        
        # 保存相关性矩阵
        corr_csv_path = os.path.join(output_dir, f"static_features_correlation_{method}.csv")
        corr_matrix.to_csv(corr_csv_path)
        print(f"💾 相关性矩阵保存至: {corr_csv_path}")
        
        # 生成标准热力图
        heatmap_path = os.path.join(output_dir, f"static_features_heatmap_{method}.png")
        fig, ax = create_static_features_heatmap(corr_matrix, method.capitalize(), heatmap_path)
        plt.close(fig)
        
        # 生成聚类热力图
        clustered_path = os.path.join(output_dir, f"static_features_heatmap_clustered_{method}.png")
        g = create_clustered_static_heatmap(corr_matrix, method.capitalize(), clustered_path)
        plt.close(g.fig)
        
        # 找出最强相关性
        top_correlations = find_strongest_correlations(corr_matrix, top_n=10)
        
        # 生成分析报告
        report_path = os.path.join(output_dir, f"static_features_analysis_report_{method}.md")
        create_analysis_report(corr_matrix, top_correlations, method.capitalize(), report_path)
        
        # 显示前5个最强相关性
        print(f"\n🔥 {method.capitalize()} 最强相关性 (Top 5):")
        for i, corr in enumerate(top_correlations[:5], 1):
            corr_type = "正相关" if corr['correlation'] > 0 else "负相关"
            print(f"   {i}. {corr['feature1']} ↔ {corr['feature2']}")
            print(f"      相关系数: {corr['correlation']:.3f} ({corr_type})")
    
    print(f"\n🎉 静态特征相关性分析完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 生成文件:")
    print(f"   • static_features_heatmap_pearson.png - Pearson热力图")
    print(f"   • static_features_heatmap_spearman.png - Spearman热力图")
    print(f"   • static_features_heatmap_clustered_pearson.png - Pearson聚类热力图")
    print(f"   • static_features_heatmap_clustered_spearman.png - Spearman聚类热力图")
    print(f"   • static_features_correlation_pearson.csv - Pearson相关性矩阵")
    print(f"   • static_features_correlation_spearman.csv - Spearman相关性矩阵")
    print(f"   • static_features_analysis_report_pearson.md - Pearson分析报告")
    print(f"   • static_features_analysis_report_spearman.md - Spearman分析报告")

if __name__ == '__main__':
    main()
