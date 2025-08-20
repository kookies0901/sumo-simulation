#!/usr/bin/env python3
"""
过拟合问题分析 - 简化版本
专门针对小样本数据的回归分析，提供清晰的过拟合诊断
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, LeaveOneOut
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def safe_cross_validation(model, X, y):
    """安全的交叉验证，处理可能的数值问题"""
    try:
        cv = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        # 过滤掉无效值
        valid_scores = scores[~np.isnan(scores)]
        if len(valid_scores) > 0:
            return valid_scores.mean(), valid_scores.std(), len(valid_scores)
        else:
            return 0.0, 0.0, 0
    except:
        return 0.0, 0.0, 0

def analyze_single_relationship(df, x_col, y_col):
    """分析单个特征与目标变量的关系"""
    
    # 获取数据并清理
    x = df[x_col].values
    y = df[y_col].values
    
    # 移除NaN值
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 5:
        return None
    
    # 基本统计
    sample_size = len(x_clean)
    
    # 皮尔逊相关系数
    try:
        pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    except:
        pearson_r, pearson_p = 0.0, 1.0
    
    # 准备数据
    X = x_clean.reshape(-1, 1)
    
    # 1. 线性回归
    linear_model = LinearRegression()
    linear_model.fit(X, y_clean)
    y_pred_linear = linear_model.predict(X)
    train_r2_linear = r2_score(y_clean, y_pred_linear)
    
    # 交叉验证
    cv_r2_linear, cv_std_linear, cv_count = safe_cross_validation(linear_model, X, y_clean)
    overfitting_linear = train_r2_linear - cv_r2_linear
    
    # 2. Ridge回归（正则化）
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y_clean)
    y_pred_ridge = ridge_model.predict(X)
    train_r2_ridge = r2_score(y_clean, y_pred_ridge)
    
    cv_r2_ridge, cv_std_ridge, _ = safe_cross_validation(ridge_model, X, y_clean)
    overfitting_ridge = train_r2_ridge - cv_r2_ridge
    
    # 数据变异性分析
    x_cv = np.std(x_clean) / np.mean(x_clean) if np.mean(x_clean) != 0 else 0
    y_cv = np.std(y_clean) / np.mean(y_clean) if np.mean(y_clean) != 0 else 0
    
    # 选择最佳模型
    if cv_r2_ridge > cv_r2_linear:
        best_model = "Ridge"
        best_cv_r2 = cv_r2_ridge
        best_overfitting = overfitting_ridge
    else:
        best_model = "Linear"
        best_cv_r2 = cv_r2_linear
        best_overfitting = overfitting_linear
    
    # 关系质量评估
    if abs(pearson_r) < 0.15:
        quality = "相关性极弱"
        recommendation = "考虑舍弃"
    elif best_overfitting > 0.5:
        quality = "严重过拟合"
        recommendation = "数据不足，谨慎使用"
    elif best_overfitting > 0.3:
        quality = "中度过拟合"
        recommendation = "需要更多数据"
    elif best_cv_r2 < 0.1 and abs(pearson_r) > 0.2:
        quality = "模型不当"
        recommendation = "尝试非线性模型"
    elif best_cv_r2 > 0.2 and best_overfitting < 0.2:
        quality = "良好"
        recommendation = "可用于分析"
    else:
        quality = "一般"
        recommendation = "可参考使用"
    
    return {
        'feature': x_col,
        'target': y_col,
        'sample_size': sample_size,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'linear_train_r2': train_r2_linear,
        'linear_cv_r2': cv_r2_linear,
        'linear_overfitting': overfitting_linear,
        'ridge_train_r2': train_r2_ridge,
        'ridge_cv_r2': cv_r2_ridge,
        'ridge_overfitting': overfitting_ridge,
        'best_model': best_model,
        'best_cv_r2': best_cv_r2,
        'best_overfitting': best_overfitting,
        'x_variability': x_cv,
        'y_variability': y_cv,
        'quality': quality,
        'recommendation': recommendation
    }

def main():
    print("🔍 过拟合问题分析")
    print("=" * 50)
    
    # 加载数据
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input/merged_dataset.csv"
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return 1
    
    # 定义变量
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
    
    print(f"📊 分析 {len(layout_features)} 个布局特征 vs {len(performance_metrics)} 个性能指标")
    print(f"🔢 总共 {len(layout_features) * len(performance_metrics)} 个关系")
    
    results = []
    
    # 逐一分析
    for i, feature in enumerate(layout_features):
        print(f"\n[{i+1}/{len(layout_features)}] 分析特征: {feature}")
        
        for target in performance_metrics:
            if feature in df.columns and target in df.columns:
                result = analyze_single_relationship(df, feature, target)
                if result:
                    results.append(result)
                    print(f"  {target}: r={result['pearson_r']:.3f}, "
                          f"CV R²={result['best_cv_r2']:.3f}, "
                          f"过拟合={result['best_overfitting']:.3f}, "
                          f"{result['quality']}")
    
    # 保存和分析结果
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n" + "=" * 60)
        print(f"📊 过拟合分析汇总报告")
        print(f"=" * 60)
        
        print(f"✅ 成功分析了 {len(results)} 个关系")
        
        # 质量分布
        print(f"\n🎯 关系质量分布:")
        quality_counts = results_df['quality'].value_counts()
        for quality, count in quality_counts.items():
            percentage = count / len(results) * 100
            print(f"   {quality}: {count} 个 ({percentage:.1f}%)")
        
        # 过拟合严重程度统计
        print(f"\n⚠️ 过拟合严重程度:")
        overfitting_severe = len(results_df[results_df['best_overfitting'] > 0.5])
        overfitting_moderate = len(results_df[(results_df['best_overfitting'] > 0.3) & (results_df['best_overfitting'] <= 0.5)])
        overfitting_mild = len(results_df[(results_df['best_overfitting'] > 0.1) & (results_df['best_overfitting'] <= 0.3)])
        overfitting_none = len(results_df[results_df['best_overfitting'] <= 0.1])
        
        print(f"   严重过拟合 (>0.5): {overfitting_severe} 个")
        print(f"   中度过拟合 (0.3-0.5): {overfitting_moderate} 个") 
        print(f"   轻度过拟合 (0.1-0.3): {overfitting_mild} 个")
        print(f"   无过拟合 (≤0.1): {overfitting_none} 个")
        
        # 最佳关系（可用于论文）
        print(f"\n🏆 推荐用于论文的关系 (质量='良好'):")
        good_relationships = results_df[results_df['quality'] == '良好'].sort_values('best_cv_r2', ascending=False)
        
        if len(good_relationships) > 0:
            for _, row in good_relationships.iterrows():
                print(f"   ✅ {row['feature']} -> {row['target']}")
                print(f"      相关系数: {row['pearson_r']:.4f}, CV R²: {row['best_cv_r2']:.4f}, 过拟合: {row['best_overfitting']:.4f}")
        else:
            print(f"   ❌ 没有发现质量为'良好'的关系")
        
        # 需要谨慎处理的关系
        print(f"\n⚠️ 需要谨慎处理的关系:")
        problematic = results_df[results_df['quality'].isin(['严重过拟合', '中度过拟合'])]
        if len(problematic) > 0:
            for _, row in problematic.head(5).iterrows():
                print(f"   ⚠️ {row['feature']} -> {row['target']}: {row['quality']}")
                print(f"      过拟合程度: {row['best_overfitting']:.4f}, 建议: {row['recommendation']}")
        
        # 数据问题诊断
        print(f"\n🔍 数据问题诊断:")
        avg_overfitting = results_df['best_overfitting'].mean()
        avg_cv_r2 = results_df['best_cv_r2'].mean()
        
        print(f"   平均过拟合程度: {avg_overfitting:.4f}")
        print(f"   平均交叉验证R²: {avg_cv_r2:.4f}")
        print(f"   样本量: {results_df['sample_size'].iloc[0]}")
        
        if avg_overfitting > 0.3:
            print(f"   🚨 整体过拟合严重，主要原因:")
            print(f"      1. 样本量太小 (N={len(df)})")
            print(f"      2. 特征与目标变量真实相关性可能较弱")
            print(f"      3. 需要收集更多数据")
        
        # 保存结果
        output_file = "/home/ubuntu/project/MSC/Msc_Project/models/analysis_simple/overfitting_analysis.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 详细结果已保存到: {output_file}")
        
        # 总结建议
        print(f"\n📝 总结建议:")
        good_count = len(good_relationships)
        if good_count > 0:
            print(f"   ✅ 发现 {good_count} 个可用关系，建议在论文中重点展示")
            print(f"   📊 使用简单线性回归或Ridge回归")
            print(f"   📈 报告交叉验证R²而非训练R²")
        else:
            print(f"   ⚠️ 当前数据难以建立稳定的回归关系")
            print(f"   💡 建议:")
            print(f"      - 增加样本量（目标>100个布局）")
            print(f"      - 考虑描述性统计分析而非回归")
            print(f"      - 使用相关性分析代替回归分析")
    
    return 0

if __name__ == '__main__':
    exit(main())
