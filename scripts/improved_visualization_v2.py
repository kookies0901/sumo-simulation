#!/usr/bin/env python3
"""
改进的可视化脚本 v2 - 基于generate_graphs_simple.py的拟合方式
使用Linear和Polynomial回归，增加更多特征展示
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

# 设置matplotlib参数
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = ['DejaVu Sans']  # 使用系统默认字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def fit_simple_models(x, y):
    """使用与generate_graphs_simple.py相同的拟合方式"""
    try:
        # 移除NaN值
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 5:
            return None, None, 0.0, "insufficient_data", {}
        
        # 重塑数据为sklearn格式
        X = x_clean.reshape(-1, 1)
        
        # 定义两个回归模型（与generate_graphs_simple.py保持一致）
        models = {
            'Linear': LinearRegression(),
            'Polynomial': LinearRegression()  # 将与多项式特征一起使用
        }
        
        model_results = {}
        best_model = None
        best_r2 = -np.inf
        best_model_name = ""
        
        for name, model in models.items():
            try:
                if name == 'Polynomial':
                    # 多项式回归
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(X)
                    model.fit(X_poly, y_clean)
                    y_pred = model.predict(X_poly)
                    
                    # 生成预测曲线
                    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
                    X_fit_poly = poly_features.transform(x_fit)
                    y_fit = model.predict(X_fit_poly)
                else:
                    # 线性回归
                    model.fit(X, y_clean)
                    y_pred = model.predict(X)
                    
                    # 生成预测曲线
                    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
                    y_fit = model.predict(x_fit)
                
                # 计算指标
                r2 = r2_score(y_clean, y_pred)
                
                # 计算皮尔逊相关系数
                correlation = np.corrcoef(x_clean, y_clean)[0, 1]
                
                model_results[name] = {
                    'r2': r2,
                    'correlation': correlation,
                    'x_fit': x_fit.flatten(),
                    'y_fit': y_fit,
                    'model': model
                }
                
                # 更新最佳模型（使用R²作为标准，与generate_graphs_simple.py一致）
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name
                    best_model_name = name
                    
            except Exception as e:
                print(f"   ⚠️ 模型 {name} 训练失败: {e}")
                continue
        
        if best_model is None:
            return None, None, 0.0, "no_valid_model", {}
        
        # 返回最佳模型的结果
        best_result = model_results[best_model]
        return (best_result['x_fit'], best_result['y_fit'], 
                best_result['r2'], best_model_name, model_results)
        
    except Exception as e:
        print(f"⚠️ 模型拟合失败: {e}")
        return None, None, 0.0, "error", {}

def get_feature_performance_columns():
    """获取所有特征和性能指标列（扩展版本）"""
    
    # 布局特征变量（完整列表）
    feature_columns = [
        'avg_dist_to_center',
        'std_nearest_neighbor', 
        'min_distance',
        'max_pairwise_distance',
        'cs_density_std',
        'cluster_count',
        'coverage_ratio',
        'max_gap_distance',
        'gini_coefficient',
        'avg_betweenness_centrality'
    ]
    
    # 性能指标（完整列表）
    performance_columns = [
        'duration_mean',
        'duration_median', 
        'duration_p90',
        'charging_time_mean',
        'charging_time_median',
        'charging_time_p90',
        'waiting_time_mean',
        'waiting_time_median',
        'waiting_time_p90',
        'energy_gini',
        'energy_cv',
        'energy_hhi',
        'energy_p90_p50_ratio',
        'vehicle_gini',
        'vehicle_cv', 
        'vehicle_hhi',
        'charging_station_coverage',
        'reroute_count',
        'ev_charging_participation_rate',
        'ev_charging_failures'
    ]
    
    return feature_columns, performance_columns

def get_column_display_info():
    """获取列的显示名称和单位信息"""
    
    # 特征变量的显示名称和单位
    feature_display = {
        'avg_dist_to_center': ('Average Distance to Center', 'meters'),
        'std_nearest_neighbor': ('Std of Nearest Neighbor Distance', 'meters'),
        'min_distance': ('Minimum Distance', 'meters'),
        'max_pairwise_distance': ('Maximum Pairwise Distance', 'meters'),
        'cs_density_std': ('Charging Station Density Std', 'stations/km²'),
        'cluster_count': ('Cluster Count', 'clusters'),
        'coverage_ratio': ('Coverage Ratio', 'ratio'),
        'max_gap_distance': ('Maximum Gap Distance', 'meters'),
        'gini_coefficient': ('Gini Coefficient', 'coefficient'),
        'avg_betweenness_centrality': ('Average Betweenness Centrality', 'centrality')
    }
    
    # 性能指标的显示名称和单位
    performance_display = {
        'duration_mean': ('Mean Trip Duration', 'seconds'),
        'duration_median': ('Median Trip Duration', 'seconds'),
        'duration_p90': ('90th Percentile Trip Duration', 'seconds'),
        'charging_time_mean': ('Mean Charging Time', 'seconds'),
        'charging_time_median': ('Median Charging Time', 'seconds'),
        'charging_time_p90': ('90th Percentile Charging Time', 'seconds'),
        'waiting_time_mean': ('Mean Waiting Time', 'seconds'),
        'waiting_time_median': ('Median Waiting Time', 'seconds'),
        'waiting_time_p90': ('90th Percentile Waiting Time', 'seconds'),
        'energy_gini': ('Energy Distribution Gini', 'coefficient'),
        'energy_cv': ('Energy Coefficient of Variation', 'coefficient'),
        'energy_hhi': ('Energy Herfindahl-Hirschman Index', 'index'),
        'energy_p90_p50_ratio': ('Energy P90/P50 Ratio', 'ratio'),
        'vehicle_gini': ('Vehicle Distribution Gini', 'coefficient'),
        'vehicle_cv': ('Vehicle Coefficient of Variation', 'coefficient'),
        'vehicle_hhi': ('Vehicle Herfindahl-Hirschman Index', 'index'),
        'charging_station_coverage': ('Charging Station Coverage', 'ratio'),
        'reroute_count': ('Reroute Count', 'count'),
        'ev_charging_participation_rate': ('EV Charging Participation Rate', 'rate'),
        'ev_charging_failures': ('EV Charging Failures', 'count')
    }
    
    return feature_display, performance_display

def format_axis_label(column_name, display_info):
    """格式化轴标签，包含单位"""
    if column_name in display_info:
        display_name, unit = display_info[column_name]
        return f"{display_name} ({unit})"
    else:
        return column_name

def create_improved_scatter_plot(df, x_col, y_col, output_dir):
    """创建改进的散点图，使用与generate_graphs_simple.py相同的拟合方式"""
    
    # 获取显示信息
    feature_display, performance_display = get_column_display_info()
    
    # 创建2x2的子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x = df[x_col].values
    y = df[y_col].values
    
    # 移除NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # 1. 全数据散点图（左上）- 使用最佳拟合模型
    ax1.scatter(x_clean, y_clean, alpha=0.7, s=80, color='steelblue', 
               edgecolors='black', linewidth=0.5)
    
    # 使用与generate_graphs_simple.py相同的拟合方法
    x_fit, y_fit, r2_full, best_model, model_results = fit_simple_models(x, y)
    
    if x_fit is not None and y_fit is not None:
        # 根据模型类型设置颜色
        color_map = {'Linear': 'darkred', 'Polynomial': 'red'}
        color = color_map.get(best_model, 'darkred')
        ax1.plot(x_fit, y_fit, color=color, linewidth=2.5,
                label=f'{best_model} (R² = {r2_full:.3f})')
        
        # 添加模型比较信息
        if len(model_results) > 1:
            best_result = model_results[best_model]
            info_text = f"Best: {best_model} (R² = {r2_full:.3f})\n"
            info_text += f"Correlation: {best_result['correlation']:.3f}\n"
            
            # 显示两个模型的R²比较
            for name, result in model_results.items():
                if name != best_model:
                    info_text += f"{name}: R² = {result['r2']:.3f}"
            
            # 创建文本框显示信息
            ax1.text(0.02, 0.98, info_text.strip(), transform=ax1.transAxes, 
                   fontsize=8, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax1.set_title(f'Full Dataset (N={len(x_clean)})', fontweight='bold')
    ax1.set_xlabel(format_axis_label(x_col, feature_display))
    ax1.set_ylabel(format_axis_label(y_col, performance_display))
    if x_fit is not None:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 中等密度区间（右上）
    x_min_central = np.percentile(x_clean, 25)
    x_max_central = np.percentile(x_clean, 75)
    
    mask_central = (x_clean >= x_min_central) & (x_clean <= x_max_central)
    x_central = x_clean[mask_central]
    y_central = y_clean[mask_central]
    
    ax2.scatter(x_central, y_central, alpha=0.7, s=80, color='green', 
               edgecolors='black', linewidth=0.5)
    
    # 拟合中等密度数据
    if len(x_central) > 5:
        x_fit_central, y_fit_central, r2_central, best_model_central, _ = fit_simple_models(x_central, y_central)
        
        if x_fit_central is not None:
            color_central = color_map.get(best_model_central, 'darkgreen')
            ax2.plot(x_fit_central, y_fit_central, color=color_central, linewidth=2.5,
                    label=f'{best_model_central} (R² = {r2_central:.3f})')
    else:
        r2_central = 0.0
        best_model_central = "insufficient_data"
    
    ax2.set_title(f'Central Density Range (N={len(x_central)})', fontweight='bold')
    ax2.set_xlabel(format_axis_label(x_col, feature_display))
    ax2.set_ylabel(format_axis_label(y_col, performance_display))
    if len(x_central) > 5:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 数据分布直方图（左下）
    ax3.hist(x_clean, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x_min_central, color='red', linestyle='--', alpha=0.7, 
               label='Central Range')
    ax3.axvline(x_max_central, color='red', linestyle='--', alpha=0.7)
    ax3.set_title('X-axis Data Distribution', fontweight='bold')
    ax3.set_xlabel(format_axis_label(x_col, feature_display))
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 分区间分析（右下）
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
        
        if len(x_region) > 5:
            ax4.scatter(x_region, y_region, alpha=0.7, s=80, color=color, 
                       label=f'{name} (N={len(x_region)})', edgecolors='black', linewidth=0.5)
            
            # 局部拟合
            x_fit_region, y_fit_region, r2_region, best_model_region, _ = fit_simple_models(x_region, y_region)
            r2_regions.append((name, r2_region, best_model_region))
            
            if x_fit_region is not None:
                ax4.plot(x_fit_region, y_fit_region, color=color, linewidth=2, alpha=0.7)
    
    ax4.set_title('Regional Analysis', fontweight='bold')
    ax4.set_xlabel(format_axis_label(x_col, feature_display))
    ax4.set_ylabel(format_axis_label(y_col, performance_display))
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加总体信息
    x_display = feature_display.get(x_col, (x_col, ''))[0]
    y_display = performance_display.get(y_col, (y_col, ''))[0]
    fig.suptitle(f'{x_display} vs {y_display} - Multi-perspective Analysis', fontsize=14, fontweight='bold')
    
    # 添加文本说明
    info_text = f"Full Data: {best_model} (R² = {r2_full:.3f})\\n"
    if len(x_central) > 5:
        info_text += f"Central Range: {best_model_central} (R² = {r2_central:.3f})\\n"
    for name, r2_val, model_name in r2_regions:
        info_text += f"{name} Region: {model_name} (R² = {r2_val:.3f})\\n"
    
    fig.text(0.02, 0.02, info_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存
    filename = f"{x_col}_{y_col}_improved_v2.png"
    filepath = f"{output_dir}/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath, r2_full, best_model

def generate_comprehensive_visualization(df, output_dir):
    """生成全面的可视化图表"""
    
    feature_columns, performance_columns = get_feature_performance_columns()
    
    # 验证列是否存在
    available_features = [col for col in feature_columns if col in df.columns]
    available_performance = [col for col in performance_columns if col in df.columns]
    
    print(f"📊 可用特征变量: {len(available_features)} 个")
    print(f"📈 可用性能指标: {len(available_performance)} 个")
    print(f"🎯 将生成图表数量: {len(available_features) * len(available_performance)} 张")
    
    # 创建PDF合集
    pdf_path = os.path.join(output_dir, "improved_visualization_comprehensive.pdf")
    
    results = []
    success_count = 0
    
    with PdfPages(pdf_path) as pdf:
        for i, feature in enumerate(available_features, 1):
            print(f"\n📊 处理特征变量 [{i}/{len(available_features)}]: {feature}")
            
            for j, performance in enumerate(available_performance, 1):
                print(f"   📈 [{j}/{len(available_performance)}] {performance}...", end="")
                
                try:
                    # 创建单独的图表保存为PNG
                    filepath, r2, best_model = create_improved_scatter_plot(df, feature, performance, output_dir)
                    
                    # 同时保存到PDF
                    # 重新生成图形用于PDF
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # 简化版本用于PDF（减少计算时间）
                    x = df[feature].values
                    y = df[performance].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    # 只在PDF中显示全数据图
                    ax1.scatter(x_clean, y_clean, alpha=0.7, s=60, color='steelblue', edgecolors='black')
                    
                    x_fit, y_fit, r2_val, best_model_name, model_results = fit_simple_models(x, y)
                    if x_fit is not None:
                        color_map = {'Linear': 'darkred', 'Polynomial': 'red'}
                        color = color_map.get(best_model_name, 'darkred')
                        ax1.plot(x_fit, y_fit, color=color, linewidth=2,
                                label=f'{best_model_name} (R² = {r2_val:.3f})')
                        ax1.legend()
                    
                    feature_display, performance_display = get_column_display_info()
                    ax1.set_xlabel(format_axis_label(feature, feature_display))
                    ax1.set_ylabel(format_axis_label(performance, performance_display))
                    
                    x_display = feature_display.get(feature, (feature, ''))[0]
                    y_display = performance_display.get(performance, (performance, ''))[0]
                    ax1.set_title(f'{x_display} vs {y_display}', fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    
                    # 隐藏其他子图
                    for ax in [ax2, ax3, ax4]:
                        ax.set_visible(False)
                    
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=200, bbox_inches='tight')
                    plt.close()
                    
                    # 记录结果
                    results.append({
                        'feature': feature,
                        'performance': performance,
                        'r2': r2,
                        'best_model': best_model,
                        'filename': filepath
                    })
                    
                    success_count += 1
                    print(f" ✅ (R²={r2:.3f}, {best_model})")
                    
                except Exception as e:
                    print(f" ❌ 失败: {e}")
                    continue
    
    # 保存结果统计
    if results:
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, "improved_visualization_results.csv")
        results_df.to_csv(results_csv, index=False)
        
        print(f"\n🎉 改进可视化生成完成！")
        print(f"✅ 成功生成: {success_count} 张图表")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 PDF合集: {pdf_path}")
        print(f"📊 结果统计: {results_csv}")
        
        # 显示统计信息
        print(f"\n📈 拟合质量统计:")
        print(f"   - 平均 R²: {results_df['r2'].mean():.3f}")
        print(f"   - 最高 R²: {results_df['r2'].max():.3f}")
        print(f"   - R² > 0.5 的图表: {len(results_df[results_df['r2'] > 0.5])} 张")
        print(f"   - R² > 0.3 的图表: {len(results_df[results_df['r2'] > 0.3])} 张")
        
        print(f"\n🎯 最佳模型分布:")
        model_counts = results_df['best_model'].value_counts()
        for model, count in model_counts.items():
            print(f"   - {model}: {count} 张")
        
        # 显示最佳关系
        print(f"\n🌟 最佳关系（按R²排序）:")
        top_results = results_df.nlargest(10, 'r2')
        for _, row in top_results.iterrows():
            print(f"   {row['feature']} -> {row['performance']}: "
                  f"R²={row['r2']:.3f}, {row['best_model']}")
    
    return results_df if results else pd.DataFrame()

def main():
    print("🚀 开始改进的可视化分析 v2 - 基于generate_graphs_simple.py的拟合方式")
    
    # 设置路径
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv"
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/models/plots_improved_v2"
    
    print(f"📊 数据文件: {data_file}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    try:
        df = pd.read_csv(data_file)
        print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return 1
    
    # 生成全面的可视化
    results_df = generate_comprehensive_visualization(df, output_dir)
    
    if len(results_df) > 0:
        print(f"\n🎓 改进可视化图表已生成完毕！")
        print(f"📁 所有图表保存在: {output_dir}")
        print(f"📝 图表命名规则: 特征变量_性能指标_improved_v2.png")
        print(f"📑 PDF合集可直接用于论文插图")
        print(f"💡 使用Linear和Polynomial回归，与generate_graphs_simple.py保持一致")
        print(f"🎨 多视角分析解决数据分布不均匀问题")
    else:
        print("❌ 没有成功生成任何图表")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
