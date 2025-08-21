#!/usr/bin/env python3
"""
生成充电桩布局特征与性能指标的散点图分析 - 简化版本
仅使用Linear和Polynomial回归模型，避免过拟合问题
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = ['DejaVu Sans']  # 使用系统默认字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def load_merged_dataset(file_path):
    """加载合并后的数据集"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
        return df
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def get_feature_performance_columns(df):
    """定义特征变量和性能指标列"""
    
    # 布局特征变量（移除已删除的列）
    feature_columns = [
        'avg_dist_to_center',
        'std_pairwise_distance',
        'min_pairwise_distance',
        'max_pairwise_distance',
        'cs_density_std',
        'cluster_count',
        'coverage_ratio',
        'max_gap_distance',
        'gini_coefficient',
        'avg_betweenness_centrality'
    ]
    
    # 性能指标（移除已删除的列）
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
    
    # 验证列是否存在
    missing_features = [col for col in feature_columns if col not in df.columns]
    missing_performance = [col for col in performance_columns if col not in df.columns]
    
    if missing_features:
        print(f"⚠️ 缺少特征列: {missing_features}")
    if missing_performance:
        print(f"⚠️ 缺少性能列: {missing_performance}")
    
    # 只使用存在的列
    available_features = [col for col in feature_columns if col in df.columns]
    available_performance = [col for col in performance_columns if col in df.columns]
    
    print(f"📊 可用特征变量: {len(available_features)} 个")
    print(f"📈 可用性能指标: {len(available_performance)} 个")
    print(f"🎯 将生成图表数量: {len(available_features) * len(available_performance)} 张")
    
    return available_features, available_performance

def get_column_display_info():
    """获取列的显示名称和单位信息"""
    
    # 特征变量的显示名称和单位
    feature_display = {
        'avg_dist_to_center': ('Average Distance to Center', 'meters'),
        'std_pairwise_distance': ('Std of Pairwise Distance', 'meters'),
        'min_pairwise_distance': ('Minimum Pairwise Distance', 'meters'),
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

def fit_simple_models(x, y):
    """训练Linear和Polynomial回归模型并返回最佳模型的结果"""
    try:
        # 移除NaN值
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 5:
            return None, None, 0.0, "insufficient_data", {}
        
        # 重塑数据为sklearn格式
        X = x_clean.reshape(-1, 1)
        
        # 定义两个回归模型
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
                mse = mean_squared_error(y_clean, y_pred)
                
                # 计算皮尔逊相关系数
                correlation, p_value = stats.pearsonr(x_clean, y_clean)
                
                model_results[name] = {
                    'r2': r2,
                    'mse': mse,
                    'correlation': correlation,
                    'p_value': p_value,
                    'x_fit': x_fit.flatten(),
                    'y_fit': y_fit,
                    'model': model
                }
                
                # 更新最佳模型
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

def create_scatter_plot(df, x_col, y_col, output_dir):
    """创建单个散点图"""
    try:
        # 获取显示信息
        feature_display, performance_display = get_column_display_info()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取数据
        x = df[x_col].values
        y = df[y_col].values
        
        # 创建散点图
        scatter = ax.scatter(x, y, alpha=0.7, s=80, color='steelblue', 
                           edgecolors='black', linewidth=0.5)
        
        # 使用简单模型拟合
        x_fit, y_fit, r2, best_model, model_results = fit_simple_models(x, y)
        
        # 绘制最佳拟合线
        if x_fit is not None and y_fit is not None:
            # 根据模型类型设置颜色
            color_map = {
                'Linear': 'darkred',
                'Polynomial': 'red'
            }
            color = color_map.get(best_model, 'darkred')
            ax.plot(x_fit, y_fit, color=color, linewidth=2.5,
                   label=f'{best_model} (R² = {r2:.3f})')
            
            # 添加模型比较信息（修复乱码）
            if len(model_results) > 1:
                best_result = model_results[best_model]
                info_text = f"Best Model: {best_model}\n"
                info_text += f"R²: {r2:.3f}\n"
                info_text += f"Correlation: {best_result['correlation']:.3f}\n"
                
                # 显示两个模型的R²比较
                for name, result in model_results.items():
                    if name != best_model:
                        info_text += f"{name} R²: {result['r2']:.3f}\n"
                
                # 创建文本框显示信息
                ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 格式化轴标签（包含单位）
        x_label = format_axis_label(x_col, feature_display)
        y_label = format_axis_label(y_col, performance_display)
        
        # 设置标签和标题
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        
        # 创建更简洁的标题
        x_display = feature_display.get(x_col, (x_col, ''))[0]
        y_display = performance_display.get(y_col, (y_col, ''))[0]
        ax.set_title(f'{x_display} vs {y_display}', fontsize=14, fontweight='bold', pad=20)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 添加图例
        if x_fit is not None:
            ax.legend(loc='best', framealpha=0.8)
        
        # 设置样式
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        filename = f"{x_col}_{y_col}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath, r2, best_model
        
    except Exception as e:
        print(f"❌ 创建图表失败 {x_col} vs {y_col}: {e}")
        plt.close()
        return None, 0.0, "error"

def generate_all_plots(df, feature_cols, performance_cols, output_dir):
    """生成所有散点图"""
    print(f"\n🎨 开始生成图表（仅Linear和Polynomial模型）...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储结果统计
    results = []
    success_count = 0
    total_count = len(feature_cols) * len(performance_cols)
    
    # 创建PDF合集
    pdf_path = os.path.join(output_dir, "simple_scatter_plots.pdf")
    
    # 获取显示信息
    feature_display, performance_display = get_column_display_info()
    
    with PdfPages(pdf_path) as pdf:
        for i, x_col in enumerate(feature_cols, 1):
            print(f"\n📊 处理特征变量 [{i}/{len(feature_cols)}]: {x_col}")
            
            for j, y_col in enumerate(performance_cols, 1):
                print(f"   📈 [{j}/{len(performance_cols)}] {y_col}...", end="")
                
                try:
                    # 创建图形用于PDF
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # 获取数据
                    x = df[x_col].values
                    y = df[y_col].values
                    
                    # 创建散点图
                    ax.scatter(x, y, alpha=0.7, s=80, color='steelblue', 
                             edgecolors='black', linewidth=0.5)
                    
                    # 使用简单模型拟合
                    x_fit, y_fit, r2, best_model, model_results = fit_simple_models(x, y)
                    
                    # 绘制最佳拟合线
                    if x_fit is not None and y_fit is not None:
                        # 根据模型类型设置颜色
                        color_map = {
                            'Linear': 'darkred',
                            'Polynomial': 'red'
                        }
                        color = color_map.get(best_model, 'darkred')
                        ax.plot(x_fit, y_fit, color=color, linewidth=2.5,
                               label=f'{best_model} (R² = {r2:.3f})')
                        
                        # 添加模型比较信息（修复乱码）
                        if len(model_results) > 1:
                            best_result = model_results[best_model]
                            info_text = f"Best: {best_model} (R² = {r2:.3f})\n"
                            info_text += f"Correlation: {best_result['correlation']:.3f}\n"
                            
                            # 显示两个模型的R²比较
                            for name, result in model_results.items():
                                if name != best_model:
                                    info_text += f"{name}: R² = {result['r2']:.3f}"
                            
                            # 创建文本框显示信息
                            ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes, 
                                   fontsize=9, verticalalignment='top', 
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    
                    # 格式化轴标签（包含单位）
                    x_label = format_axis_label(x_col, feature_display)
                    y_label = format_axis_label(y_col, performance_display)
                    
                    # 设置标签和标题
                    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
                    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
                    
                    # 创建更简洁的标题
                    x_display = feature_display.get(x_col, (x_col, ''))[0]
                    y_display = performance_display.get(y_col, (y_col, ''))[0]
                    ax.set_title(f'{x_display} vs {y_display}', fontsize=14, fontweight='bold', pad=20)
                    
                    # 添加网格
                    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    
                    # 添加图例
                    if x_fit is not None:
                        ax.legend(loc='best', framealpha=0.8)
                    
                    # 设置样式
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_linewidth(1)
                    ax.spines['bottom'].set_linewidth(1)
                    
                    # 调整布局
                    plt.tight_layout()
                    
                    # 保存到PDF
                    pdf.savefig(fig, dpi=300, bbox_inches='tight')
                    
                    # 保存单独的PNG文件
                    filename = f"{x_col}_{y_col}.png"
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    # 记录结果，包含相关系数信息
                    correlation = model_results[best_model]['correlation'] if model_results and best_model in model_results else 0.0
                    results.append({
                        'feature': x_col,
                        'performance': y_col,
                        'r2': r2,
                        'correlation': correlation,
                        'best_model': best_model,
                        'filename': filename
                    })
                    
                    success_count += 1
                    print(f" ✅ (R²={r2:.3f}, r={correlation:.3f})")
                    
                except Exception as e:
                    print(f" ❌ 失败: {e}")
                    plt.close()
                    continue
    
    # 保存结果统计
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "simple_plot_results_summary.csv")
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n🎉 图表生成完成！")
    print(f"✅ 成功生成: {success_count}/{total_count} 张图表")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 PDF合集: {pdf_path}")
    print(f"📊 结果统计: {results_csv}")
    
    # 显示统计信息
    if len(results_df) > 0:
        print(f"\n📈 拟合质量统计:")
        print(f"   - 平均 R²: {results_df['r2'].mean():.3f}")
        print(f"   - 最高 R²: {results_df['r2'].max():.3f}")
        print(f"   - 平均相关系数: {results_df['correlation'].mean():.3f}")
        print(f"   - R² > 0.5 的图表: {len(results_df[results_df['r2'] > 0.5])} 张")
        print(f"   - R² > 0.3 的图表: {len(results_df[results_df['r2'] > 0.3])} 张")
        print(f"   - |相关系数| > 0.3 的图表: {len(results_df[abs(results_df['correlation']) > 0.3])} 张")
        
        print(f"\n🎯 最佳模型分布:")
        model_counts = results_df['best_model'].value_counts()
        for model, count in model_counts.items():
            print(f"   - {model}: {count} 张")
        
        print(f"\n🏆 按模型类型统计平均R²:")
        model_r2_avg = results_df.groupby('best_model')['r2'].mean().sort_values(ascending=False)
        for model, avg_r2 in model_r2_avg.items():
            print(f"   - {model}: {avg_r2:.3f}")
        
        # 显示最佳关系
        print(f"\n🌟 最佳关系（按R²排序）:")
        top_results = results_df.nlargest(10, 'r2')
        for _, row in top_results.iterrows():
            print(f"   {row['feature']} -> {row['performance']}: "
                  f"R²={row['r2']:.3f}, r={row['correlation']:.3f}, {row['best_model']}")
    
    return results_df

def main():
    print("🚀 开始生成充电桩布局特征与性能指标散点图（简化版本）")
    
    # 设置路径
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv"
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/models/plots_simple"
    
    print(f"📊 数据文件: {data_file}")
    print(f"📁 输出目录: {output_dir}")
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return 1
    
    # 加载数据
    df = load_merged_dataset(data_file)
    if df is None:
        return 1
    
    # 获取特征和性能指标列
    feature_cols, performance_cols = get_feature_performance_columns(df)
    
    if not feature_cols or not performance_cols:
        print("❌ 没有找到有效的特征或性能指标列")
        return 1
    
    # 生成所有图表
    results_df = generate_all_plots(df, feature_cols, performance_cols, output_dir)
    
    if len(results_df) > 0:
        print(f"\n🎓 简化版论文用图表已生成完毕！")
        print(f"📁 所有图表保存在: {output_dir}")
        print(f"📝 图表命名规则: 特征变量_性能指标.png")
        print(f"📑 PDF合集可直接用于论文插图")
        print(f"💡 仅使用Linear和Polynomial回归，避免过拟合问题")
    else:
        print("❌ 没有成功生成任何图表")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
