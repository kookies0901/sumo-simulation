#!/usr/bin/env python3
"""
简单回归分析 - 专门针对小样本过拟合问题的解决方案
使用简单模型和严格的交叉验证
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, LeaveOneOut
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_relationship(x, y, feature_name, target_name):
    """分析两个变量之间的关系，使用多种方法验证"""
    
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
    
    # 2. 简单线性回归
    X = x_clean.reshape(-1, 1)
    
    # 训练模型
    linear_model = LinearRegression()
    linear_model.fit(X, y_clean)
    y_pred_linear = linear_model.predict(X)
    
    # 训练R²
    train_r2_linear = r2_score(y_clean, y_pred_linear)
    
    # 留一交叉验证
    try:
        cv_scores_linear = cross_val_score(linear_model, X, y_clean, 
                                          cv=LeaveOneOut(), scoring='r2')
        # 清理无效分数
        cv_scores_linear = cv_scores_linear[~np.isnan(cv_scores_linear) & ~np.isinf(cv_scores_linear)]
        cv_r2_linear = cv_scores_linear.mean() if len(cv_scores_linear) > 0 else 0.0
        cv_std_linear = cv_scores_linear.std() if len(cv_scores_linear) > 0 else 0.0
    except Exception as e:
        print(f"     交叉验证失败: {e}")
        cv_scores_linear = np.array([0.0])
        cv_r2_linear = 0.0
        cv_std_linear = 0.0
    
    # 过拟合程度
    overfitting_linear = train_r2_linear - cv_r2_linear
    
    print(f"   线性回归:")
    print(f"     训练 R²: {train_r2_linear:.4f}")
    print(f"     交叉验证 R²: {cv_r2_linear:.4f} ± {cv_std_linear:.4f}")
    print(f"     过拟合程度: {overfitting_linear:.4f}")
    
    # 3. Ridge回归 (正则化)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y_clean)
    y_pred_ridge = ridge_model.predict(X)
    
    train_r2_ridge = r2_score(y_clean, y_pred_ridge)
    try:
        cv_scores_ridge = cross_val_score(ridge_model, X, y_clean, 
                                         cv=LeaveOneOut(), scoring='r2')
        # 清理无效分数
        cv_scores_ridge = cv_scores_ridge[~np.isnan(cv_scores_ridge) & ~np.isinf(cv_scores_ridge)]
        cv_r2_ridge = cv_scores_ridge.mean() if len(cv_scores_ridge) > 0 else 0.0
    except Exception as e:
        print(f"     Ridge交叉验证失败: {e}")
        cv_scores_ridge = np.array([0.0])
        cv_r2_ridge = 0.0
    overfitting_ridge = train_r2_ridge - cv_r2_ridge
    
    print(f"   Ridge回归 (α=1.0):")
    print(f"     训练 R²: {train_r2_ridge:.4f}")
    print(f"     交叉验证 R²: {cv_r2_ridge:.4f}")
    print(f"     过拟合程度: {overfitting_ridge:.4f}")
    
    # 4. 检验数据范围和分布
    x_range = x_clean.max() - x_clean.min()
    x_cv = np.std(x_clean) / np.mean(x_clean)
    y_range = y_clean.max() - y_clean.min()
    y_cv = np.std(y_clean) / np.mean(y_clean)
    
    print(f"   数据特征:")
    print(f"     X变异系数: {x_cv:.4f}")
    print(f"     Y变异系数: {y_cv:.4f}")
    print(f"     X范围/均值: {x_range/np.mean(x_clean):.4f}")
    print(f"     Y范围/均值: {y_range/np.mean(y_clean):.4f}")
    
    # 选择最佳模型
    if cv_r2_ridge > cv_r2_linear and overfitting_ridge < overfitting_linear:
        best_model = "Ridge"
        best_r2 = cv_r2_ridge
        best_overfitting = overfitting_ridge
        best_model_obj = ridge_model
    else:
        best_model = "Linear"
        best_r2 = cv_r2_linear
        best_overfitting = overfitting_linear
        best_model_obj = linear_model
    
    # 质量评估
    if abs(pearson_r) < 0.2:
        quality = "弱相关"
    elif best_overfitting > 0.3:
        quality = "过拟合严重"
    elif best_r2 < 0.1:
        quality = "模型效果差"
    elif best_r2 > 0.3 and best_overfitting < 0.1:
        quality = "良好"
    else:
        quality = "一般"
    
    print(f"   最佳模型: {best_model}")
    print(f"   模型质量: {quality}")
    
    return {
        'feature': feature_name,
        'target': target_name,
        'sample_size': len(x_clean),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'linear_train_r2': train_r2_linear,
        'linear_cv_r2': cv_r2_linear,
        'linear_overfitting': overfitting_linear,
        'ridge_train_r2': train_r2_ridge,
        'ridge_cv_r2': cv_r2_ridge,
        'ridge_overfitting': overfitting_ridge,
        'best_model': best_model,
        'best_cv_r2': best_r2,
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
    
    # 1. 原始散点图 + 线性回归
    ax1.scatter(x_clean, y_clean, alpha=0.7, s=60, color='steelblue', edgecolors='black')
    
    # 拟合线性回归
    X = x_clean.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y_clean)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax1.plot(x_line, y_line, 'r-', linewidth=2, label='线性回归')
    
    # 计算统计信息
    train_r2 = r2_score(y_clean, model.predict(X))
    try:
        cv_scores = cross_val_score(model, X, y_clean, cv=LeaveOneOut(), scoring='r2')
        # 清理无效分数
        cv_scores = cv_scores[~np.isnan(cv_scores) & ~np.isinf(cv_scores)]
        cv_r2 = cv_scores.mean() if len(cv_scores) > 0 else 0.0
    except Exception as e:
        print(f"     诊断图交叉验证失败: {e}")
        cv_scores = np.array([0.0])
        cv_r2 = 0.0
    pearson_r, _ = stats.pearsonr(x_clean, y_clean)
    
    ax1.set_title(f'{x_col} vs {y_col}')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'训练R²: {train_r2:.3f}\n交叉验证R²: {cv_r2:.3f}\n皮尔逊r: {pearson_r:.3f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. 残差图
    residuals = y_clean - model.predict(X)
    ax2.scatter(model.predict(X), residuals, alpha=0.7, s=60, color='green')
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
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"{x_col}_{y_col}_diagnostic.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    return filepath

def main():
    print("🔍 开始简单回归分析 - 防过拟合版本")
    
    # 设置路径
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv"
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/models/analysis_simple"
    
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
        'avg_dist_to_center', 'std_nearest_neighbor', 'min_distance',
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
        results_file = os.path.join(output_dir, "regression_analysis_results.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"\n🎉 分析完成！")
        print(f"📊 总共分析: {len(results)} 个关系")
        print(f"💾 结果保存到: {results_file}")
        
        # 汇总统计
        print(f"\n📈 质量分布:")
        quality_counts = results_df['quality'].value_counts()
        for quality, count in quality_counts.items():
            print(f"   {quality}: {count} 个")
        
        print(f"\n🏆 最佳关系 (按交叉验证R²排序):")
        best_results = results_df.sort_values('best_cv_r2', ascending=False).head(10)
        for _, row in best_results.iterrows():
            print(f"   {row['feature']} -> {row['target']}: "
                  f"CV R² = {row['best_cv_r2']:.4f}, "
                  f"过拟合 = {row['best_overfitting']:.4f}, "
                  f"质量 = {row['quality']}")
        
        print(f"\n⚠️ 过拟合严重的关系:")
        overfitting_issues = results_df[results_df['best_overfitting'] > 0.2]
        for _, row in overfitting_issues.iterrows():
            print(f"   {row['feature']} -> {row['target']}: "
                  f"过拟合 = {row['best_overfitting']:.4f}")
        
        print(f"\n📊 建议:")
        good_quality = len(results_df[results_df['quality'] == '良好'])
        if good_quality > 0:
            print(f"   ✅ 发现 {good_quality} 个高质量关系，可用于论文")
        else:
            print(f"   ⚠️ 没有发现高质量关系，数据可能存在以下问题:")
            print(f"      - 样本量太小 (N={len(df)})")
            print(f"      - 变量之间真实相关性较弱")
            print(f"      - 需要更多数据或特征工程")
    
    return 0

if __name__ == '__main__':
    exit(main())
