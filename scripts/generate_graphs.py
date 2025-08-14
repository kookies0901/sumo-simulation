#!/usr/bin/env python3
"""
生成充电桩布局特征与性能指标的散点图分析
为硕士论文制作高质量的散点图 + 回归趋势线图表
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
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
    
    # 12个布局特征变量
    feature_columns = [
        'cs_count',
        'avg_dist_to_center',
        'avg_nearest_neighbor',
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
    
    # 22个性能指标
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
        'energy_zero_usage_rate',
        'vehicle_gini',
        'vehicle_cv',
        'vehicle_hhi',
        'vehicle_zero_usage_rate',
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

def calculate_correlation_and_fit(x, y):
    """计算相关系数和拟合曲线"""
    try:
        # 移除NaN值
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return None, None, 0.0, "insufficient_data"
        
        # 计算Pearson相关系数
        correlation, p_value = stats.pearsonr(x_clean, y_clean)
        
        # 决定拟合方法
        if len(x_clean) < 10:
            # 数据点少，使用线性拟合
            coeffs = np.polyfit(x_clean, y_clean, 1)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_fit = poly(x_fit)
            r2 = r2_score(y_clean, poly(x_clean))
            fit_type = "linear"
        else:
            # 尝试二阶多项式拟合
            try:
                coeffs = np.polyfit(x_clean, y_clean, 2)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_fit = poly(x_fit)
                r2_poly = r2_score(y_clean, poly(x_clean))
                
                # 比较线性拟合
                coeffs_linear = np.polyfit(x_clean, y_clean, 1)
                poly_linear = np.poly1d(coeffs_linear)
                r2_linear = r2_score(y_clean, poly_linear(x_clean))
                
                # 如果二阶多项式明显更好，使用它
                if r2_poly > r2_linear + 0.05:
                    r2 = r2_poly
                    fit_type = "polynomial"
                else:
                    # 否则使用线性拟合
                    coeffs = coeffs_linear
                    poly = poly_linear
                    y_fit = poly(x_fit)
                    r2 = r2_linear
                    fit_type = "linear"
                    
            except:
                # 如果多项式拟合失败，使用线性拟合
                coeffs = np.polyfit(x_clean, y_clean, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_fit = poly(x_fit)
                r2 = r2_score(y_clean, poly(x_clean))
                fit_type = "linear"
        
        return x_fit, y_fit, r2, fit_type
        
    except Exception as e:
        print(f"⚠️ 拟合计算失败: {e}")
        return None, None, 0.0, "error"

def create_scatter_plot(df, x_col, y_col, output_dir):
    """创建单个散点图"""
    try:
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取数据
        x = df[x_col].values
        y = df[y_col].values
        
        # 创建散点图
        scatter = ax.scatter(x, y, alpha=0.6, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
        
        # 计算拟合线
        x_fit, y_fit, r2, fit_type = calculate_correlation_and_fit(x, y)
        
        # 绘制拟合线
        if x_fit is not None and y_fit is not None:
            color = 'red' if fit_type == 'polynomial' else 'darkred'
            linestyle = '--' if fit_type == 'polynomial' else '-'
            ax.plot(x_fit, y_fit, color=color, linewidth=2, linestyle=linestyle,
                   label=f'{fit_type.title()} Fit (R² = {r2:.3f})')
        
        # 设置标签和标题
        ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
        ax.set_title(f'{x_col} vs {y_col}', fontsize=14, fontweight='bold', pad=20)
        
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
        
        return filepath, r2, fit_type
        
    except Exception as e:
        print(f"❌ 创建图表失败 {x_col} vs {y_col}: {e}")
        plt.close()
        return None, 0.0, "error"

def generate_all_plots(df, feature_cols, performance_cols, output_dir):
    """生成所有散点图"""
    print(f"\n🎨 开始生成图表...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储结果统计
    results = []
    success_count = 0
    total_count = len(feature_cols) * len(performance_cols)
    
    # 创建PDF合集
    pdf_path = os.path.join(output_dir, "all_scatter_plots.pdf")
    
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
                    ax.scatter(x, y, alpha=0.6, s=60, color='steelblue', 
                             edgecolors='black', linewidth=0.5)
                    
                    # 计算拟合线
                    x_fit, y_fit, r2, fit_type = calculate_correlation_and_fit(x, y)
                    
                    # 绘制拟合线
                    if x_fit is not None and y_fit is not None:
                        color = 'red' if fit_type == 'polynomial' else 'darkred'
                        linestyle = '--' if fit_type == 'polynomial' else '-'
                        ax.plot(x_fit, y_fit, color=color, linewidth=2, linestyle=linestyle,
                               label=f'{fit_type.title()} Fit (R² = {r2:.3f})')
                    
                    # 设置标签和标题
                    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
                    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
                    ax.set_title(f'{x_col} vs {y_col}', fontsize=14, fontweight='bold', pad=20)
                    
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
                    
                    # 记录结果
                    results.append({
                        'feature': x_col,
                        'performance': y_col,
                        'r2': r2,
                        'fit_type': fit_type,
                        'filename': filename
                    })
                    
                    success_count += 1
                    print(f" ✅ (R²={r2:.3f})")
                    
                except Exception as e:
                    print(f" ❌ 失败: {e}")
                    plt.close()
                    continue
    
    # 保存结果统计
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "plot_results_summary.csv")
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
        print(f"   - R² > 0.5 的图表: {len(results_df[results_df['r2'] > 0.5])} 张")
        print(f"   - R² > 0.3 的图表: {len(results_df[results_df['r2'] > 0.3])} 张")
        
        print(f"\n🎯 拟合方法分布:")
        fit_type_counts = results_df['fit_type'].value_counts()
        for fit_type, count in fit_type_counts.items():
            print(f"   - {fit_type}: {count} 张")
    
    return results_df

def main():
    print("🚀 开始生成充电桩布局特征与性能指标散点图")
    
    # 设置路径
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input/merged_dataset.csv"
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/models/plots"
    
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
        print(f"\n🎓 论文用图表已生成完毕！")
        print(f"📁 所有图表保存在: {output_dir}")
        print(f"📝 图表命名规则: 特征变量_性能指标.png")
        print(f"📑 PDF合集可直接用于论文插图")
    else:
        print("❌ 没有成功生成任何图表")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
