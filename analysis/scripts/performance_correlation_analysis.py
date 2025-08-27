#!/usr/bin/env python3
"""
动态性能指标相关性分析
计算81个布局方案的20个动态性能指标之间的相关性，并生成专业热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_data(csv_file_path):
    """加载并验证数据"""
    print("📊 加载数据...")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    print(f"✅ 数据加载成功: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    return df

def identify_performance_metrics(df):
    """识别20个动态性能指标列"""
    # 静态布局特征列（前11列）
    static_features = [
        'avg_dist_to_center', 'avg_pairwise_distance', 'std_pairwise_distance',
        'min_pairwise_distance', 'max_pairwise_distance', 'cs_density_std',
        'cluster_count', 'coverage_ratio', 'max_gap_distance',
        'gini_coefficient', 'avg_betweenness_centrality'
    ]
    
    # layout_id 列
    id_column = 'layout_id'
    
    # 动态性能指标列（除了静态特征和layout_id之外的所有列）
    all_columns = df.columns.tolist()
    performance_metrics = [col for col in all_columns 
                          if col not in static_features and col != id_column]
    
    print(f"🔍 识别到 {len(performance_metrics)} 个动态性能指标:")
    for i, metric in enumerate(performance_metrics, 1):
        print(f"   {i:2d}. {metric}")
    
    return performance_metrics

def calculate_correlation_matrix(df, metrics, method='pearson'):
    """计算相关性矩阵"""
    print(f"\n🧮 计算{method}相关系数矩阵...")
    
    # 提取性能指标数据
    metrics_data = df[metrics].copy()
    
    # 检查数据质量
    print(f"📋 数据质量检查:")
    print(f"   - 样本数量: {len(metrics_data)}")
    print(f"   - 指标数量: {len(metrics)}")
    
    # 检查缺失值
    missing_counts = metrics_data.isnull().sum()
    if missing_counts.any():
        print(f"   ⚠️ 发现缺失值:")
        for metric, count in missing_counts[missing_counts > 0].items():
            print(f"     - {metric}: {count} 个缺失值")
        
        # 填充缺失值（使用中位数）
        metrics_data = metrics_data.fillna(metrics_data.median())
        print(f"   ✅ 已使用中位数填充缺失值")
    else:
        print(f"   ✅ 无缺失值")
    
    # 计算相关性矩阵
    if method == 'pearson':
        corr_matrix = metrics_data.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = metrics_data.corr(method='spearman')
    else:
        raise ValueError("method 必须是 'pearson' 或 'spearman'")
    
    print(f"✅ {method}相关性矩阵计算完成")
    
    return corr_matrix, metrics_data

def calculate_significance_matrix(metrics_data, method='pearson'):
    """计算相关性显著性矩阵（p值）"""
    print(f"🔬 计算相关性显著性（p值）...")
    
    n_metrics = len(metrics_data.columns)
    p_matrix = np.ones((n_metrics, n_metrics))
    
    for i in range(n_metrics):
        for j in range(i+1, n_metrics):
            if method == 'pearson':
                _, p_value = pearsonr(metrics_data.iloc[:, i], metrics_data.iloc[:, j])
            else:  # spearman
                _, p_value = spearmanr(metrics_data.iloc[:, i], metrics_data.iloc[:, j])
            
            p_matrix[i, j] = p_value
            p_matrix[j, i] = p_value
    
    p_df = pd.DataFrame(p_matrix, 
                       index=metrics_data.columns, 
                       columns=metrics_data.columns)
    
    print(f"✅ 显著性矩阵计算完成")
    return p_df

def create_correlation_heatmap(corr_matrix, p_matrix=None, method='pearson', 
                              output_path=None, figsize=(16, 14)):
    """创建专业的相关性热力图"""
    print(f"🎨 生成{method}相关性热力图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建mask for上三角（可选，显示完整矩阵）
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # 生成热力图
    sns.heatmap(corr_matrix, 
                # mask=mask,
                annot=True,                    # 显示数值
                fmt='.3f',                     # 数值格式
                cmap='RdBu_r',                # 配色方案：红-白-蓝
                center=0,                      # 中心值为0
                square=True,                   # 正方形格子
                cbar_kws={"shrink": .8, "label": f"{method.capitalize()} Correlation Coefficient"},
                linewidths=0.5,               # 网格线宽度
                ax=ax)
    
    # 如果有显著性信息，添加显著性标记
    if p_matrix is not None:
        # 添加显著性标记 (* p<0.05, ** p<0.01, *** p<0.001)
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                p_val = p_matrix.iloc[i, j]
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker and i != j:  # 不在对角线上添加标记
                    ax.text(j + 0.5, i + 0.8, marker, 
                           ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='black')
    
    # 设置标题和标签
    method_name_cn = "皮尔逊" if method == 'pearson' else "斯皮尔曼"
    ax.set_title(f'动态性能指标{method_name_cn}相关性热力图\n'
                f'Dynamic Performance Metrics {method.capitalize()} Correlation Heatmap\n'
                f'(n=81 layouts)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 旋转x轴标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ 热力图保存至: {output_path}")
    
    return fig, ax

def create_clustered_heatmap(corr_matrix, method='pearson', output_path=None, figsize=(18, 16)):
    """创建带聚类的相关性热力图"""
    print(f"🌲 生成带层次聚类的{method}相关性热力图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用 clustermap 创建带聚类的热力图
    g = sns.clustermap(corr_matrix,
                       annot=True,
                       fmt='.3f',
                       cmap='RdBu_r',
                       center=0,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={"shrink": .8, "label": f"{method.capitalize()} Correlation Coefficient"},
                       figsize=figsize,
                       dendrogram_ratio=0.15,    # 树状图比例
                       cbar_pos=(0.02, 0.83, 0.03, 0.15))  # 调整colorbar位置
    
    # 设置标题
    method_name_cn = "皮尔逊" if method == 'pearson' else "斯皮尔曼"
    g.fig.suptitle(f'动态性能指标{method_name_cn}相关性热力图（带层次聚类）\n'
                   f'Dynamic Performance Metrics {method.capitalize()} Correlation Heatmap with Hierarchical Clustering\n'
                   f'(n=81 layouts)', 
                   fontsize=16, fontweight='bold', y=0.98)
    
    # 旋转标签
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    
    # 保存图形
    if output_path:
        g.savefig(output_path, dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
        print(f"✅ 聚类热力图保存至: {output_path}")
    
    return g

def analyze_correlation_patterns(corr_matrix, threshold=0.7):
    """分析相关性模式"""
    print(f"\n🔍 相关性模式分析（阈值: |r| ≥ {threshold}）")
    print("=" * 60)
    
    # 提取上三角矩阵（避免重复和对角线）
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix.where(mask)
    
    # 找出高相关性的指标对
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'metric1': corr_matrix.index[i],
                    'metric2': corr_matrix.columns[j],
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val)
                })
    
    # 按绝对相关性大小排序
    high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    print(f"🔥 发现 {len(high_corr_pairs)} 对高相关性指标（|r| ≥ {threshold}）：")
    if high_corr_pairs:
        print(f"{'序号':<4} {'指标1':<25} {'指标2':<25} {'相关系数':<10} {'类型'}")
        print("-" * 80)
        for i, pair in enumerate(high_corr_pairs, 1):
            corr_type = "正相关" if pair['correlation'] > 0 else "负相关"
            print(f"{i:<4} {pair['metric1']:<25} {pair['metric2']:<25} "
                  f"{pair['correlation']:<10.3f} {corr_type}")
    else:
        print(f"   未发现绝对相关性 ≥ {threshold} 的指标对")
    
    # 统计相关性分布
    all_corr_values = corr_values.values.flatten()
    all_corr_values = all_corr_values[~np.isnan(all_corr_values)]
    
    print(f"\n📊 相关性分布统计：")
    print(f"   总指标对数: {len(all_corr_values)}")
    print(f"   平均相关性: {np.mean(np.abs(all_corr_values)):.3f}")
    print(f"   最大正相关: {np.max(all_corr_values):.3f}")
    print(f"   最大负相关: {np.min(all_corr_values):.3f}")
    print(f"   强相关对数 (|r|≥0.7): {np.sum(np.abs(all_corr_values) >= 0.7)}")
    print(f"   中等相关对数 (0.3≤|r|<0.7): {np.sum((np.abs(all_corr_values) >= 0.3) & (np.abs(all_corr_values) < 0.7))}")
    print(f"   弱相关对数 (|r|<0.3): {np.sum(np.abs(all_corr_values) < 0.3)}")
    
    return high_corr_pairs

def save_correlation_results(corr_matrix, p_matrix, metrics_data, output_dir, method='pearson'):
    """保存相关性分析结果"""
    print(f"\n💾 保存{method}相关性分析结果...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存相关性矩阵
    corr_file = os.path.join(output_dir, f"correlation_matrix_{method}.csv")
    corr_matrix.to_csv(corr_file)
    print(f"✅ 相关性矩阵保存至: {corr_file}")
    
    # 保存显著性矩阵
    if p_matrix is not None:
        p_file = os.path.join(output_dir, f"significance_matrix_{method}.csv")
        p_matrix.to_csv(p_file)
        print(f"✅ 显著性矩阵保存至: {p_file}")
    
    # 保存描述性统计
    desc_file = os.path.join(output_dir, f"descriptive_statistics_{method}.csv")
    desc_stats = metrics_data.describe()
    desc_stats.to_csv(desc_file)
    print(f"✅ 描述性统计保存至: {desc_file}")
    
    # 创建分析报告
    report_file = os.path.join(output_dir, f"correlation_analysis_report_{method}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"动态性能指标{method.upper()}相关性分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析方法: {method.capitalize()} Correlation\n")
        f.write(f"样本数量: {len(metrics_data)}\n")
        f.write(f"指标数量: {len(metrics_data.columns)}\n\n")
        
        f.write("指标列表:\n")
        for i, metric in enumerate(metrics_data.columns, 1):
            f.write(f"   {i:2d}. {metric}\n")
        
        f.write("\n相关性矩阵摘要:\n")
        # 提取上三角矩阵
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_values = corr_matrix.where(mask).values.flatten()
        corr_values = corr_values[~np.isnan(corr_values)]
        
        f.write(f"   最大正相关: {np.max(corr_values):.3f}\n")
        f.write(f"   最大负相关: {np.min(corr_values):.3f}\n")
        f.write(f"   平均绝对相关性: {np.mean(np.abs(corr_values)):.3f}\n")
        f.write(f"   强相关对数 (|r|≥0.7): {np.sum(np.abs(corr_values) >= 0.7)}\n")
        f.write(f"   中等相关对数 (0.3≤|r|<0.7): {np.sum((np.abs(corr_values) >= 0.3) & (np.abs(corr_values) < 0.7))}\n")
        f.write(f"   弱相关对数 (|r|<0.3): {np.sum(np.abs(corr_values) < 0.3)}\n")
    
    print(f"✅ 分析报告保存至: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='动态性能指标相关性分析')
    parser.add_argument('--input', type=str, 
                       default='models/input_1-100/merged_dataset.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='analysis/charts/correlation',
                       help='输出目录')
    parser.add_argument('--method', type=str, choices=['pearson', 'spearman', 'both'],
                       default='both', help='相关性计算方法')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='高相关性阈值')
    parser.add_argument('--figsize', type=str, default='16,14',
                       help='图形尺寸 (宽,高)')
    
    args = parser.parse_args()
    
    # 解析图形尺寸
    figsize = tuple(map(int, args.figsize.split(',')))
    
    print("🔍 动态性能指标相关性分析")
    print("=" * 50)
    
    # 加载数据
    df = load_and_validate_data(args.input)
    
    # 识别性能指标
    performance_metrics = identify_performance_metrics(df)
    
    if len(performance_metrics) != 20:
        print(f"⚠️ 警告: 检测到 {len(performance_metrics)} 个动态性能指标，期望20个")
        print("请检查数据文件格式是否正确")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 选择分析方法
    methods = ['pearson', 'spearman'] if args.method == 'both' else [args.method]
    
    for method in methods:
        print(f"\n{'='*20} {method.upper()} 相关性分析 {'='*20}")
        
        # 计算相关性矩阵
        corr_matrix, metrics_data = calculate_correlation_matrix(df, performance_metrics, method)
        
        # 计算显著性矩阵
        p_matrix = calculate_significance_matrix(metrics_data, method)
        
        # 生成标准热力图
        heatmap_path = os.path.join(args.output_dir, f"correlation_heatmap_{method}.png")
        fig1, ax1 = create_correlation_heatmap(corr_matrix, p_matrix, method, 
                                              heatmap_path, figsize)
        plt.close(fig1)
        
        # 生成聚类热力图
        clustered_path = os.path.join(args.output_dir, f"correlation_heatmap_clustered_{method}.png")
        g = create_clustered_heatmap(corr_matrix, method, clustered_path, figsize)
        plt.close(g.fig)
        
        # 分析相关性模式
        high_corr_pairs = analyze_correlation_patterns(corr_matrix, args.threshold)
        
        # 保存结果
        save_correlation_results(corr_matrix, p_matrix, metrics_data, args.output_dir, method)
    
    print(f"\n🎉 相关性分析完成！")
    print(f"📁 所有结果保存在: {args.output_dir}")
    print(f"📊 生成的文件包括:")
    print(f"   • 相关性热力图 (标准版和聚类版)")
    print(f"   • 相关性矩阵 CSV 文件")
    print(f"   • 显著性检验 p值矩阵")
    print(f"   • 描述性统计")
    print(f"   • 分析报告")

if __name__ == '__main__':
    main()
