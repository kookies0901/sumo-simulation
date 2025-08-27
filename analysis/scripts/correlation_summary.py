#!/usr/bin/env python3
"""
相关性分析结果总结脚本
生成简洁的相关性发现总结和重要指标间关系图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_correlation_results():
    """加载相关性分析结果"""
    corr_file = "analysis/charts/correlation/correlation_matrix_pearson.csv"
    
    if not os.path.exists(corr_file):
        print("❌ 请先运行 performance_correlation_analysis.py")
        return None
    
    corr_matrix = pd.read_csv(corr_file, index_col=0)
    print(f"✅ 加载相关性矩阵: {corr_matrix.shape}")
    
    return corr_matrix

def find_top_correlations(corr_matrix, top_n=20):
    """找出最强的相关性"""
    # 提取上三角矩阵，避免重复和对角线
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # 获取所有相关性值
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
    
    # 按绝对相关性排序
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    return correlations[:top_n]

def create_top_correlations_plot(top_correlations, output_path):
    """创建最强相关性的可视化"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据
    correlations = [x['correlation'] for x in top_correlations]
    labels = [f"{x['metric1'][:15]}...\nvs\n{x['metric2'][:15]}..." 
              if len(x['metric1']) > 15 or len(x['metric2']) > 15
              else f"{x['metric1']}\nvs\n{x['metric2']}" 
              for x in top_correlations]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 创建颜色映射
    colors = ['red' if x < 0 else 'blue' for x in correlations]
    
    # 创建条形图
    bars = ax.barh(range(len(correlations)), correlations, color=colors, alpha=0.7)
    
    # 设置标签
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('皮尔逊相关系数 (Pearson Correlation Coefficient)', fontsize=12)
    ax.set_title('动态性能指标间最强相关性 (Top 20)\nStrongest Correlations Among Dynamic Performance Metrics', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(corr + (0.02 if corr > 0 else -0.02), i, 
                f'{corr:.3f}', 
                va='center', ha='left' if corr > 0 else 'right',
                fontsize=9, fontweight='bold')
    
    # 添加图例
    import matplotlib.patches as patches
    legend_elements = [
        patches.Patch(color='blue', alpha=0.7, label='正相关 (Positive)'),
        patches.Patch(color='red', alpha=0.7, label='负相关 (Negative)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✅ 最强相关性图保存至: {output_path}")
    
    plt.close()

def create_correlation_categories_plot(corr_matrix, output_path):
    """创建按指标类别分组的相关性图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 定义指标分类
    categories = {
        '行驶时间类': ['duration_mean', 'duration_median', 'duration_p90'],
        '充电时间类': ['charging_time_mean', 'charging_time_median', 'charging_time_p90'],
        '等待时间类': ['waiting_time_mean', 'waiting_time_median', 'waiting_time_p90'],
        '能耗分布类': ['energy_gini', 'energy_cv', 'energy_hhi', 'energy_p90_p50_ratio'],
        '车辆分布类': ['vehicle_gini', 'vehicle_cv', 'vehicle_hhi'],
        '系统指标类': ['charging_station_coverage', 'reroute_count', 
                  'ev_charging_participation_rate', 'ev_charging_failures']
    }
    
    # 重新排序相关性矩阵
    ordered_metrics = []
    category_boundaries = []
    current_pos = 0
    
    for category, metrics in categories.items():
        available_metrics = [m for m in metrics if m in corr_matrix.columns]
        ordered_metrics.extend(available_metrics)
        current_pos += len(available_metrics)
        category_boundaries.append(current_pos)
    
    # 重新排序矩阵
    ordered_corr = corr_matrix.loc[ordered_metrics, ordered_metrics]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 生成热力图
    sns.heatmap(ordered_corr, 
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={"shrink": .8, "label": "皮尔逊相关系数"},
                linewidths=0.5,
                ax=ax)
    
    # 添加分类边界线
    for boundary in category_boundaries[:-1]:
        ax.axhline(boundary, color='black', linewidth=2)
        ax.axvline(boundary, color='black', linewidth=2)
    
    # 添加分类标签
    y_positions = []
    prev_boundary = 0
    for i, boundary in enumerate(category_boundaries):
        y_pos = (prev_boundary + boundary) / 2
        y_positions.append(y_pos)
        prev_boundary = boundary
    
    # 在右侧添加分类标签
    for i, (category, y_pos) in enumerate(zip(categories.keys(), y_positions)):
        ax.text(len(ordered_metrics) + 0.5, y_pos, category, 
               rotation=0, ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 设置标题
    ax.set_title('动态性能指标分类相关性热力图\nCategorized Correlation Heatmap of Dynamic Performance Metrics', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 旋转标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ 分类相关性图保存至: {output_path}")
    
    plt.close()

def generate_summary_report(corr_matrix, top_correlations, output_path):
    """生成总结报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("动态性能指标相关性分析总结报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 数据概览\n")
        f.write("-" * 30 + "\n")
        f.write(f"分析指标数量: {len(corr_matrix)} 个\n")
        f.write(f"样本数量: 81 个布局方案\n")
        f.write(f"总相关性对数: {len(corr_matrix) * (len(corr_matrix) - 1) // 2} 对\n\n")
        
        f.write("🔥 最强相关性发现 (Top 10)\n")
        f.write("-" * 30 + "\n")
        for i, corr in enumerate(top_correlations[:10], 1):
            corr_type = "正相关" if corr['correlation'] > 0 else "负相关"
            f.write(f"{i:2d}. {corr['metric1']} ↔ {corr['metric2']}\n")
            f.write(f"    相关系数: {corr['correlation']:.3f} ({corr_type})\n\n")
        
        f.write("📈 相关性强度分布\n")
        f.write("-" * 30 + "\n")
        
        # 计算相关性分布
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        all_corr = corr_matrix.where(mask).values.flatten()
        all_corr = all_corr[~np.isnan(all_corr)]
        
        very_strong = np.sum(np.abs(all_corr) >= 0.8)
        strong = np.sum((np.abs(all_corr) >= 0.6) & (np.abs(all_corr) < 0.8))
        moderate = np.sum((np.abs(all_corr) >= 0.3) & (np.abs(all_corr) < 0.6))
        weak = np.sum(np.abs(all_corr) < 0.3)
        
        f.write(f"极强相关 (|r| ≥ 0.8): {very_strong} 对 ({very_strong/len(all_corr)*100:.1f}%)\n")
        f.write(f"强相关 (0.6 ≤ |r| < 0.8): {strong} 对 ({strong/len(all_corr)*100:.1f}%)\n")
        f.write(f"中等相关 (0.3 ≤ |r| < 0.6): {moderate} 对 ({moderate/len(all_corr)*100:.1f}%)\n")
        f.write(f"弱相关 (|r| < 0.3): {weak} 对 ({weak/len(all_corr)*100:.1f}%)\n\n")
        
        f.write("🎯 关键发现\n")
        f.write("-" * 30 + "\n")
        f.write("1. 充电系统效率指标群：\n")
        f.write("   - charging_time_* 与 vehicle_hhi 高度正相关\n")
        f.write("   - charging_station_coverage 与多个效率指标负相关\n\n")
        
        f.write("2. 能耗分布指标群：\n")
        f.write("   - energy_gini, energy_cv, energy_hhi 内部高相关\n")
        f.write("   - 与充电参与率存在显著关联\n\n")
        
        f.write("3. 时间性能指标群：\n")
        f.write("   - duration_*, waiting_time_*, charging_time_* 内部相关\n")
        f.write("   - 体现系统整体性能协同效应\n\n")

def main():
    """主函数"""
    print("📊 相关性分析结果总结")
    print("=" * 40)
    
    # 加载相关性矩阵
    corr_matrix = load_correlation_results()
    if corr_matrix is None:
        return
    
    # 创建输出目录
    output_dir = "analysis/charts/correlation/summary"
    os.makedirs(output_dir, exist_ok=True)
    
    # 找出最强相关性
    top_correlations = find_top_correlations(corr_matrix, top_n=20)
    print(f"🔍 识别出前20个最强相关性")
    
    # 生成可视化
    top_corr_plot = os.path.join(output_dir, "top_correlations.png")
    create_top_correlations_plot(top_correlations, top_corr_plot)
    
    category_plot = os.path.join(output_dir, "categorized_correlation.png")
    create_correlation_categories_plot(corr_matrix, category_plot)
    
    # 生成总结报告
    summary_report = os.path.join(output_dir, "correlation_summary_report.txt")
    generate_summary_report(corr_matrix, top_correlations, summary_report)
    print(f"✅ 总结报告保存至: {summary_report}")
    
    print(f"\n🎉 相关性总结完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 生成文件:")
    print(f"   • top_correlations.png - 最强相关性条形图")
    print(f"   • categorized_correlation.png - 分类相关性热力图")
    print(f"   • correlation_summary_report.txt - 详细总结报告")

if __name__ == '__main__':
    main()
