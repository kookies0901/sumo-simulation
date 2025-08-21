#!/usr/bin/env python3
"""
简单回归分析 - 与generate_graphs_simple.py保持一致的方法
仅使用Linear和Polynomial回归模型，避免过拟合问题
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def fit_simple_models(x, y):
    """训练Linear和Polynomial回归模型并返回最佳模型的结果（与generate_graphs_simple.py保持一致）"""
    try:
        # 移除NaN值
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 5:
            return None, "insufficient_data", {}
        
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
                    
                    # 用于交叉验证的多项式管道
                    poly_pipeline = Pipeline([
                        ('poly', PolynomialFeatures(degree=2)),
                        ('linear', LinearRegression())
                    ])
                    cv_model = poly_pipeline
                else:
                    # 线性回归
                    model.fit(X, y_clean)
                    y_pred = model.predict(X)
                    cv_model = model
                
                # 计算训练R²
                r2 = r2_score(y_clean, y_pred)
                mse = mean_squared_error(y_clean, y_pred)
                
                # 交叉验证
                try:
                    cv_scores = cross_val_score(cv_model, X, y_clean, 
                                              cv=LeaveOneOut(), scoring='r2')
                    cv_scores = cv_scores[~np.isnan(cv_scores) & ~np.isinf(cv_scores)]
                    cv_r2 = cv_scores.mean() if len(cv_scores) > 0 else 0.0
                    cv_std = cv_scores.std() if len(cv_scores) > 0 else 0.0
                except Exception as e:
                    cv_r2 = 0.0
                    cv_std = 0.0
                    cv_scores = np.array([0.0])
                
                # 计算皮尔逊相关系数
                correlation, p_value = stats.pearsonr(x_clean, y_clean)
                
                model_results[name] = {
                    'r2': r2,
                    'cv_r2': cv_r2,
                    'cv_std': cv_std,
                    'cv_scores': cv_scores,
                    'overfitting': r2 - cv_r2,
                    'mse': mse,
                    'correlation': correlation,
                    'p_value': p_value,
                    'model': model
                }
                
                # 更新最佳模型（使用训练R²作为标准，与generate_graphs_simple.py一致）
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name
                    best_model_name = name
                    
            except Exception as e:
                print(f"   ⚠️ 模型 {name} 训练失败: {e}")
                continue
        
        if best_model is None:
            return None, "no_valid_model", {}
        
        return model_results, best_model_name, model_results[best_model]
        
    except Exception as e:
        print(f"⚠️ 模型拟合失败: {e}")
        return None, "error", {}

def analyze_relationship(x, y, feature_name, target_name):
    """分析两个变量之间的关系，使用Linear和Polynomial回归（与generate_graphs_simple.py保持一致）"""
    
    # 移除NaN值
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 5:
        return None
    
    print(f"\n📊 分析: {feature_name} vs {target_name}")
    print(f"   样本数量: {len(x_clean)}")
    
    # 1. 皮尔逊相关系数
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    print(f"   皮尔逊相关系数: r = {pearson_r:.4f}, p-value = {pearson_p:.4f}")
    
    # 2. 使用与generate_graphs_simple.py相同的模型拟合方式
    model_results, best_model_name, best_result = fit_simple_models(x, y)
    
    if model_results is None:
        print(f"   ⚠️ 模型拟合失败")
        return None
    
    # 输出各模型结果
    for name, result in model_results.items():
        print(f"   {name}回归:")
        print(f"     训练 R²: {result['r2']:.4f}")
        print(f"     交叉验证 R²: {result['cv_r2']:.4f} ± {result['cv_std']:.4f}")
        print(f"     过拟合程度: {result['overfitting']:.4f}")
    
    # 3. 数据特征分析
    x_range = x_clean.max() - x_clean.min()
    x_cv = np.std(x_clean) / np.mean(x_clean) if np.mean(x_clean) != 0 else 0
    y_range = y_clean.max() - y_clean.min()
    y_cv = np.std(y_clean) / np.mean(y_clean) if np.mean(y_clean) != 0 else 0
    
    print(f"   数据特征:")
    print(f"     X变异系数: {x_cv:.4f}")
    print(f"     Y变异系数: {y_cv:.4f}")
    print(f"     X范围/均值: {x_range/np.mean(x_clean):.4f}" if np.mean(x_clean) != 0 else "     X范围/均值: N/A")
    print(f"     Y范围/均值: {y_range/np.mean(y_clean):.4f}" if np.mean(y_clean) != 0 else "     Y范围/均值: N/A")
    
    # 4. 质量评估（基于最佳模型）
    best_r2 = best_result['r2']
    best_cv_r2 = best_result['cv_r2']
    best_overfitting = best_result['overfitting']
    
    if abs(pearson_r) < 0.2:
        quality = "弱相关"
    elif best_overfitting > 0.3:
        quality = "过拟合严重"
    elif best_cv_r2 < 0.1:
        quality = "模型效果差"
    elif best_cv_r2 > 0.3 and best_overfitting < 0.1:
        quality = "良好"
    else:
        quality = "一般"
    
    print(f"   最佳模型: {best_model_name}")
    print(f"   模型质量: {quality}")
    
    return {
        'feature': feature_name,
        'target': target_name,
        'sample_size': len(x_clean),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'linear_train_r2': model_results['Linear']['r2'],
        'linear_cv_r2': model_results['Linear']['cv_r2'],
        'linear_overfitting': model_results['Linear']['overfitting'],
        'polynomial_train_r2': model_results['Polynomial']['r2'],
        'polynomial_cv_r2': model_results['Polynomial']['cv_r2'],
        'polynomial_overfitting': model_results['Polynomial']['overfitting'],
        'best_model': best_model_name,
        'best_train_r2': best_r2,
        'best_cv_r2': best_cv_r2,
        'best_overfitting': best_overfitting,
        'quality': quality,
        'x_cv': x_cv,
        'y_cv': y_cv
    }

def create_diagnostic_plot(df, x_col, y_col, output_dir):
    """创建诊断图表"""
    
    x = df[x_col].values
    y = df[y_col].values
    
    # 移除NaN值
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 5:
        return None
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 原始散点图 + 最佳回归线
    ax1.scatter(x_clean, y_clean, alpha=0.7, s=60, color='steelblue', edgecolors='black')
    
    # 使用与generate_graphs_simple.py相同的拟合方法
    model_results, best_model_name, best_result = fit_simple_models(x, y)
    
    if model_results is not None:
        # 绘制最佳拟合线
        X = x_clean.reshape(-1, 1)
        if best_model_name == 'Polynomial':
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
            X_line_poly = poly_features.transform(x_line)
            y_line = best_result['model'].predict(X_line_poly)
        else:
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
            y_line = best_result['model'].predict(x_line)
        
        color = 'red' if best_model_name == 'Polynomial' else 'darkred'
        ax1.plot(x_line, y_line, color=color, linewidth=2, 
                label=f'{best_model_name} (R² = {best_result["r2"]:.3f})')
    
    ax1.set_title(f'{x_col} vs {y_col}')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    if model_results is not None:
        stats_text = f'训练R²: {best_result["r2"]:.3f}\n交叉验证R²: {best_result["cv_r2"]:.3f}\n皮尔逊r: {best_result["correlation"]:.3f}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. 残差图
    if model_results is not None:
        X = x_clean.reshape(-1, 1)
        if best_model_name == 'Polynomial':
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            y_pred = best_result['model'].predict(X_poly)
        else:
            y_pred = best_result['model'].predict(X)
        
        residuals = y_clean - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.7, s=60, color='green')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_title('残差图')
        ax2.set_xlabel('预测值')
        ax2.set_ylabel('残差')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q图检验残差正态性
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('残差Q-Q图')
        ax3.grid(True, alpha=0.3)
        
        # 4. 交叉验证分数分布
        cv_scores = best_result['cv_scores']
        cv_r2 = best_result['cv_r2']
        
        # 检查并清理cv_scores数据
        cv_scores_clean = cv_scores[~np.isnan(cv_scores) & ~np.isinf(cv_scores)]
        
        if len(cv_scores_clean) > 0 and np.std(cv_scores_clean) > 1e-10:
            # 动态确定bins数量
            n_bins = min(10, max(3, len(cv_scores_clean) // 2))
            ax4.hist(cv_scores_clean, bins=n_bins, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(cv_r2, color='red', linestyle='--', linewidth=2, label=f'平均: {cv_r2:.3f}')
            ax4.set_title('交叉验证R²分布')
            ax4.set_xlabel('R²')
            ax4.set_ylabel('频次')
            ax4.legend()
        else:
            # 如果数据无效，显示文本说明
            ax4.text(0.5, 0.5, f'交叉验证数据:\n平均R²: {cv_r2:.3f}\n样本数: {len(cv_scores)}', 
                    ha='center', va='center', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_title('交叉验证R²信息')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
        
        ax4.grid(True, alpha=0.3)
    else:
        # 如果模型拟合失败，显示错误信息
        for ax in [ax2, ax3, ax4]:
            ax.text(0.5, 0.5, '模型拟合失败', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"{x_col}_{y_col}_diagnostic.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    return filepath

def main():
    print("🔍 开始简单回归分析 - 与generate_graphs_simple.py保持一致的方法")
    
    # 设置路径
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv"
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/models/analysis_simple_v2"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 数据文件: {data_file}")
    print(f"📁 输出目录: {output_dir}")
    
    # 加载数据
    try:
        df = pd.read_csv(data_file)
        print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return 1
    
    # 定义特征和目标变量
    layout_features = [
        'avg_dist_to_center', 'std_pairwise_distance', 'min_pairwise_distance',
        'max_pairwise_distance', 'cs_density_std', 'cluster_count',
        'coverage_ratio', 'max_gap_distance', 'gini_coefficient',
        'avg_betweenness_centrality'
    ]
    
    performance_metrics = [
        'duration_mean', 'waiting_time_mean', 'charging_time_mean',
        'energy_gini', 'vehicle_gini', 'charging_station_coverage',
        'reroute_count', 'ev_charging_participation_rate'
    ]
    
    print(f"📊 布局特征: {len(layout_features)} 个")
    print(f"📈 性能指标: {len(performance_metrics)} 个")
    print(f"💡 使用Linear和Polynomial回归模型（与generate_graphs_simple.py一致）")
    
    # 分析所有组合
    results = []
    
    print(f"\n🔍 开始分析 {len(layout_features) * len(performance_metrics)} 个组合...")
    
    for i, feature in enumerate(layout_features):
        print(f"\n[{i+1}/{len(layout_features)}] 处理特征: {feature}")
        
        for j, target in enumerate(performance_metrics):
            if feature in df.columns and target in df.columns:
                # 分析关系
                result = analyze_relationship(
                    df[feature].values, 
                    df[target].values, 
                    feature, 
                    target
                )
                
                if result:
                    results.append(result)
                    
                    # 为质量好的关系创建诊断图
                    if result['quality'] in ['良好', '一般'] and abs(result['pearson_r']) > 0.15:
                        print(f"     创建诊断图...")
                        create_diagnostic_plot(df, feature, target, output_dir)
    
    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        results_file = os.path.join(output_dir, "regression_analysis_results_v2.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"\n🎉 分析完成！")
        print(f"📊 总共分析: {len(results)} 个关系")
        print(f"💾 结果保存到: {results_file}")
        
        # 汇总统计
        print(f"\n📈 质量分布:")
        quality_counts = results_df['quality'].value_counts()
        for quality, count in quality_counts.items():
            print(f"   {quality}: {count} 个")
        
        print(f"\n🏆 最佳关系 (按训练R²排序):")
        best_results = results_df.sort_values('best_train_r2', ascending=False).head(10)
        for _, row in best_results.iterrows():
            print(f"   {row['feature']} -> {row['target']}: "
                  f"训练 R² = {row['best_train_r2']:.4f}, "
                  f"CV R² = {row['best_cv_r2']:.4f}, "
                  f"过拟合 = {row['best_overfitting']:.4f}, "
                  f"模型 = {row['best_model']}, "
                  f"质量 = {row['quality']}")
        
        print(f"\n📊 模型分布:")
        model_counts = results_df['best_model'].value_counts()
        for model, count in model_counts.items():
            print(f"   {model}: {count} 个")
        
        print(f"\n⚠️ 过拟合严重的关系:")
        overfitting_issues = results_df[results_df['best_overfitting'] > 0.2]
        if len(overfitting_issues) > 0:
            for _, row in overfitting_issues.iterrows():
                print(f"   {row['feature']} -> {row['target']}: "
                      f"过拟合 = {row['best_overfitting']:.4f}, 模型 = {row['best_model']}")
        else:
            print(f"   ✅ 没有严重过拟合的关系")
        
        print(f"\n📊 建议:")
        good_quality = len(results_df[results_df['quality'] == '良好'])
        if good_quality > 0:
            print(f"   ✅ 发现 {good_quality} 个高质量关系，可用于论文")
        else:
            moderate_quality = len(results_df[results_df['quality'] == '一般'])
            print(f"   📊 发现 {moderate_quality} 个中等质量关系")
            print(f"   💡 建议考虑数据插值以改善模型泛化能力")
    
    return 0

if __name__ == '__main__':
    exit(main())
