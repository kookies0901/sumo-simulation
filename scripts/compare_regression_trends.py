#!/usr/bin/env python3
"""
比较插值前后回归趋势的一致性
验证插值是否保持原有的回归关系
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import os

# 设置字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def fit_simple_models(x, y):
    """复制generate_graphs_simple.py的回归逻辑"""
    try:
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 5:
            return None, None, 0.0, "insufficient_data"
        
        X = x_clean.reshape(-1, 1)
        
        models = {
            'Linear': LinearRegression(),
            'Polynomial': LinearRegression()
        }
        
        best_model = None
        best_r2 = -np.inf
        best_model_name = ""
        best_x_fit = None
        best_y_fit = None
        
        for name, model in models.items():
            try:
                if name == 'Polynomial':
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(X)
                    model.fit(X_poly, y_clean)
                    y_pred = model.predict(X_poly)
                    
                    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
                    X_fit_poly = poly_features.transform(x_fit)
                    y_fit = model.predict(X_fit_poly)
                else:
                    model.fit(X, y_clean)
                    y_pred = model.predict(X)
                    
                    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
                    y_fit = model.predict(x_fit)
                
                r2 = r2_score(y_clean, y_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name
                    best_model_name = name
                    best_x_fit = x_fit.flatten()
                    best_y_fit = y_fit
                    
            except Exception as e:
                continue
        
        return best_x_fit, best_y_fit, best_r2, best_model_name
        
    except Exception as e:
        return None, None, 0.0, "fitting_error"

def compare_single_relationship(df_original, df_interpolated, x_col, y_col, output_dir):
    """比较单个变量对的回归关系"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始数据回归
    x_orig = df_original[x_col].values
    y_orig = df_original[y_col].values
    x_fit_orig, y_fit_orig, r2_orig, model_orig = fit_simple_models(x_orig, y_orig)
    
    # 插值数据回归
    x_interp = df_interpolated[x_col].values
    y_interp = df_interpolated[y_col].values
    x_fit_interp, y_fit_interp, r2_interp, model_interp = fit_simple_models(x_interp, y_interp)
    
    # 计算相关系数
    mask_orig = ~(np.isnan(x_orig) | np.isnan(y_orig))
    mask_interp = ~(np.isnan(x_interp) | np.isnan(y_interp))
    
    if np.sum(mask_orig) > 2:
        corr_orig, _ = stats.pearsonr(x_orig[mask_orig], y_orig[mask_orig])
    else:
        corr_orig = 0
        
    if np.sum(mask_interp) > 2:
        corr_interp, _ = stats.pearsonr(x_interp[mask_interp], y_interp[mask_interp])
    else:
        corr_interp = 0
    
    # 1. 原始数据图
    ax1.scatter(x_orig, y_orig, alpha=0.7, s=50, color='blue', label=f'Original Data (N={len(df_original)})')
    if x_fit_orig is not None:
        ax1.plot(x_fit_orig, y_fit_orig, 'r-', linewidth=2, 
                label=f'{model_orig}: R² = {r2_orig:.3f}, r = {corr_orig:.3f}')
    ax1.set_title('Original Data Regression', fontweight='bold')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 插值数据图
    # 分别显示原始点和插值点
    original_indices = df_interpolated['layout_id'].str.contains('cs_group_000', na=False) == False
    interpolated_indices = df_interpolated['layout_id'].str.contains('cs_group_000', na=False)
    
    ax2.scatter(x_interp[original_indices], y_interp[original_indices], 
               alpha=0.7, s=50, color='blue', label=f'Original Points ({np.sum(original_indices)})')
    ax2.scatter(x_interp[interpolated_indices], y_interp[interpolated_indices], 
               alpha=0.8, s=60, color='red', marker='^', 
               label=f'Interpolated Points ({np.sum(interpolated_indices)})')
    
    if x_fit_interp is not None:
        ax2.plot(x_fit_interp, y_fit_interp, 'g-', linewidth=2, 
                label=f'{model_interp}: R² = {r2_interp:.3f}, r = {corr_interp:.3f}')
    ax2.set_title('With Interpolated Data', fontweight='bold')
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 趋势对比图
    if x_fit_orig is not None and x_fit_interp is not None:
        # 统一X轴范围进行比较
        x_min = min(x_fit_orig.min(), x_fit_interp.min())
        x_max = max(x_fit_orig.max(), x_fit_interp.max())
        x_common = np.linspace(x_min, x_max, 100)
        
        # 重新拟合以获得相同X范围的预测
        _, y_common_orig, _, _ = fit_simple_models(x_orig, y_orig)
        _, y_common_interp, _, _ = fit_simple_models(x_interp, y_interp)
        
        if y_common_orig is not None:
            ax3.plot(x_fit_orig, y_fit_orig, 'b-', linewidth=3, alpha=0.7, 
                    label=f'Original Trend (R²={r2_orig:.3f})')
        if y_common_interp is not None:
            ax3.plot(x_fit_interp, y_fit_interp, 'r--', linewidth=2, 
                    label=f'Interpolated Trend (R²={r2_interp:.3f})')
    
    ax3.set_title('Trend Comparison', fontweight='bold')
    ax3.set_xlabel(x_col)
    ax3.set_ylabel(y_col)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    r2_diff = abs(r2_interp - r2_orig) if (r2_orig != 0 and r2_interp != 0) else 0
    corr_diff = abs(corr_interp - corr_orig)
    
    info_text = (f"R² Change: {r2_interp - r2_orig:+.3f}\n"
                f"Correlation Change: {corr_interp - corr_orig:+.3f}\n"
                f"Trend Consistency: {'Good' if r2_diff < 0.1 and corr_diff < 0.1 else 'Check'}")
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgreen' if r2_diff < 0.1 else 'lightyellow', alpha=0.8))
    
    plt.suptitle(f'{x_col} vs {y_col} - Interpolation Impact Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    safe_filename = f"{x_col}_vs_{y_col}_comparison".replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f"{safe_filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'x_col': x_col,
        'y_col': y_col,
        'r2_original': r2_orig,
        'r2_interpolated': r2_interp,
        'r2_change': r2_interp - r2_orig,
        'corr_original': corr_orig,
        'corr_interpolated': corr_interp,
        'corr_change': corr_interp - corr_orig,
        'model_original': model_orig,
        'model_interpolated': model_interp,
        'trend_consistent': r2_diff < 0.1 and corr_diff < 0.1
    }

def main():
    # 文件路径
    original_file = '/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv'
    interpolated_file = '/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/dataset_interpolated.csv'
    output_dir = '/home/ubuntu/project/MSC/Msc_Project/models/trend_comparison'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 趋势一致性分析开始...")
    print(f"原始数据: {original_file}")
    print(f"插值数据: {interpolated_file}")
    
    # 读取数据
    df_original = pd.read_csv(original_file)
    df_interpolated = pd.read_csv(interpolated_file)
    
    print(f"\n📈 数据概况:")
    print(f"原始数据点: {len(df_original)}")
    print(f"插值数据点: {len(df_interpolated)}")
    print(f"新增点数: {len(df_interpolated) - len(df_original)}")
    
    # 重要的变量对（与插值脚本保持一致）
    important_pairs = [
        ('avg_dist_to_center', 'duration_mean'),
        ('avg_dist_to_center', 'charging_time_mean'),
        ('avg_dist_to_center', 'waiting_time_mean'),
        ('gini_coefficient', 'energy_gini'),
        ('cluster_count', 'charging_station_coverage'),
        ('std_nearest_neighbor', 'duration_p90'),
        ('max_pairwise_distance', 'energy_cv'),
        ('coverage_ratio', 'vehicle_gini')
    ]
    
    results = []
    
    print(f"\n🔍 分析 {len(important_pairs)} 个重要变量对...")
    
    for i, (x_col, y_col) in enumerate(important_pairs, 1):
        if x_col in df_original.columns and y_col in df_original.columns:
            print(f"[{i}/{len(important_pairs)}] {x_col} vs {y_col}")
            
            result = compare_single_relationship(df_original, df_interpolated, x_col, y_col, output_dir)
            results.append(result)
            
            print(f"   Original R²: {result['r2_original']:.3f}, Interpolated R²: {result['r2_interpolated']:.3f}")
            print(f"   Change: {result['r2_change']:+.3f}, Consistent: {'✅' if result['trend_consistent'] else '⚠️'}")
        else:
            print(f"[{i}/{len(important_pairs)}] {x_col} vs {y_col} - 列不存在，跳过")
    
    # 生成总结报告
    if results:
        df_results = pd.DataFrame(results)
        
        print(f"\n📊 趋势一致性总结:")
        print(f"总分析变量对: {len(results)}")
        print(f"趋势一致的对数: {df_results['trend_consistent'].sum()}")
        print(f"一致性比例: {df_results['trend_consistent'].mean()*100:.1f}%")
        print(f"平均R²变化: {df_results['r2_change'].mean():+.3f}")
        print(f"平均相关系数变化: {df_results['corr_change'].mean():+.3f}")
        
        # 保存详细结果
        df_results.to_csv(os.path.join(output_dir, 'trend_consistency_analysis.csv'), index=False)
        
        print(f"\n✅ 分析完成!")
        print(f"📁 比较图表保存至: {output_dir}")
        print(f"📄 详细结果: {output_dir}/trend_consistency_analysis.csv")
        
        # 显示不一致的变量对
        inconsistent = df_results[~df_results['trend_consistent']]
        if len(inconsistent) > 0:
            print(f"\n⚠️ 需要关注的变量对 (趋势变化较大):")
            for _, row in inconsistent.iterrows():
                print(f"   {row['x_col']} vs {row['y_col']}: R²变化 {row['r2_change']:+.3f}")
        else:
            print(f"\n🎉 所有变量对的趋势都保持良好一致性!")
    
    print(f"\n💡 现在可以安心使用插值数据进行可视化，趋势保持一致！")

if __name__ == '__main__':
    main()


