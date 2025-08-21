#!/usr/bin/env python3
"""
详细的模型比较分析 - 解释为什么选择Linear和Polynomial而不是复杂模型
关键问题：即使Linear/Polynomial也有高R²，为什么它们更好？
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def comprehensive_model_comparison(x, y, feature_name, target_name):
    """全面比较所有模型的表现，重点分析泛化能力差异"""
    
    # 移除NaN值
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        return None
    
    X = x_clean.reshape(-1, 1)
    
    print(f"\n🔍 详细分析: {feature_name} vs {target_name}")
    print(f"📊 样本数量: {len(x_clean)}")
    
    # 定义所有模型
    models = {
        'Linear': {
            'model': LinearRegression(),
            'is_complex': False,
            'params': 'N/A'
        },
        'Polynomial': {
            'model': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ]),
            'is_complex': False,
            'params': 'degree=2'
        },
        'RandomForest': {
            'model': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'is_complex': True,
            'params': 'n_estimators=50, max_depth=5'
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
            'is_complex': True,
            'params': 'n_estimators=50, max_depth=3'
        },
        'SVR': {
            'model': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'is_complex': True,
            'params': 'kernel=rbf, C=1.0'
        }
    }
    
    results = []
    
    # 皮尔逊相关系数
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    print(f"📈 皮尔逊相关系数: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    
    for name, model_info in models.items():
        try:
            model = model_info['model']
            
            # 训练模型
            model.fit(X, y_clean)
            y_pred_train = model.predict(X)
            train_r2 = r2_score(y_clean, y_pred_train)
            train_mse = mean_squared_error(y_clean, y_pred_train)
            
            # 交叉验证 (关键指标!)
            cv_scores = cross_val_score(model, X, y_clean, cv=LeaveOneOut(), scoring='r2')
            cv_scores_clean = cv_scores[~np.isnan(cv_scores) & ~np.isinf(cv_scores)]
            cv_r2 = cv_scores_clean.mean() if len(cv_scores_clean) > 0 else 0.0
            cv_std = cv_scores_clean.std() if len(cv_scores_clean) > 0 else 0.0
            
            # 计算关键指标
            overfitting = train_r2 - cv_r2
            generalization_ratio = cv_r2 / train_r2 if train_r2 > 0 else 0
            
            # 稳定性分析：交叉验证分数的变异系数
            cv_stability = cv_std / abs(cv_r2) if cv_r2 != 0 else float('inf')
            
            # 模型复杂度评估
            if name == 'Linear':
                complexity_score = 1  # 最简单
            elif name == 'Polynomial':
                complexity_score = 2  # 中等
            else:
                complexity_score = 3  # 复杂
            
            # 计算综合评分 (重点：平衡拟合能力与泛化能力)
            if cv_r2 > 0:
                # 综合评分：交叉验证R² - 过拟合惩罚 - 复杂度惩罚
                composite_score = cv_r2 - 0.5 * max(0, overfitting) - 0.1 * (complexity_score - 1)
            else:
                composite_score = -1
            
            results.append({
                'model': name,
                'is_complex': model_info['is_complex'],
                'params': model_info['params'],
                'train_r2': train_r2,
                'cv_r2': cv_r2,
                'cv_std': cv_std,
                'overfitting': overfitting,
                'generalization_ratio': generalization_ratio,
                'cv_stability': cv_stability,
                'complexity_score': complexity_score,
                'composite_score': composite_score
            })
            
            print(f"\n🔧 {name} ({model_info['params']}):")
            print(f"   训练 R²: {train_r2:.4f}")
            print(f"   交叉验证 R²: {cv_r2:.4f} ± {cv_std:.4f}")
            print(f"   过拟合程度: {overfitting:.4f}")
            print(f"   泛化比率: {generalization_ratio:.4f} (越接近1越好)")
            print(f"   CV稳定性: {cv_stability:.4f} (越小越稳定)")
            print(f"   综合评分: {composite_score:.4f}")
            
        except Exception as e:
            print(f"❌ {name} 模型失败: {e}")
            continue
    
    if not results:
        return None
    
    # 分析结果
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 模型排名分析:")
    print("="*60)
    
    # 按综合评分排序
    results_sorted = results_df.sort_values('composite_score', ascending=False)
    
    for i, (_, row) in enumerate(results_sorted.iterrows(), 1):
        model_type = "复杂模型" if row['is_complex'] else "简单模型"
        print(f"{i}. {row['model']} ({model_type})")
        print(f"   综合评分: {row['composite_score']:.4f}")
        print(f"   交叉验证R²: {row['cv_r2']:.4f}")
        print(f"   过拟合程度: {row['overfitting']:.4f}")
        print(f"   泛化比率: {row['generalization_ratio']:.4f}")
    
    # 关键分析：为什么简单模型可能更好？
    simple_models = results_df[~results_df['is_complex']]
    complex_models = results_df[results_df['is_complex']]
    
    if len(simple_models) > 0 and len(complex_models) > 0:
        print(f"\n🎯 关键发现 - 为什么选择简单模型:")
        print("="*60)
        
        # 平均泛化能力比较
        simple_avg_gen = simple_models['generalization_ratio'].mean()
        complex_avg_gen = complex_models['generalization_ratio'].mean()
        
        print(f"📈 平均泛化比率:")
        print(f"   简单模型: {simple_avg_gen:.4f}")
        print(f"   复杂模型: {complex_avg_gen:.4f}")
        print(f"   差异: {simple_avg_gen - complex_avg_gen:.4f}")
        
        # 过拟合比较
        simple_avg_overfit = simple_models['overfitting'].mean()
        complex_avg_overfit = complex_models['overfitting'].mean()
        
        print(f"\n⚠️ 平均过拟合程度:")
        print(f"   简单模型: {simple_avg_overfit:.4f}")
        print(f"   复杂模型: {complex_avg_overfit:.4f}")
        print(f"   差异: {complex_avg_overfit - simple_avg_overfit:.4f}")
        
        # 稳定性比较
        simple_avg_stability = simple_models['cv_stability'].mean()
        complex_avg_stability = complex_models['cv_stability'].mean()
        
        print(f"\n🎯 平均CV稳定性 (越小越好):")
        print(f"   简单模型: {simple_avg_stability:.4f}")
        print(f"   复杂模型: {complex_avg_stability:.4f}")
        
        # 总结推荐
        best_model = results_sorted.iloc[0]
        print(f"\n🏆 推荐模型: {best_model['model']}")
        print(f"   理由: 综合评分最高 ({best_model['composite_score']:.4f})")
        print(f"   优势: 交叉验证R²={best_model['cv_r2']:.4f}, 过拟合={best_model['overfitting']:.4f}")
    
    return results_df, pearson_r

def analyze_key_relationships():
    """分析几个关键的特征-性能关系"""
    
    # 加载数据
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv"
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ 数据加载成功: {len(df)} 行")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 选择几个代表性的关系进行深入分析
    key_relationships = [
        ('cs_density_std', 'charging_time_mean'),
        ('cluster_count', 'charging_station_coverage'),
        ('max_gap_distance', 'energy_gini'),
        ('coverage_ratio', 'vehicle_gini')
    ]
    
    all_results = []
    
    for feature, target in key_relationships:
        if feature in df.columns and target in df.columns:
            result = comprehensive_model_comparison(
                df[feature].values, 
                df[target].values, 
                feature, 
                target
            )
            if result is not None:
                results_df, pearson_r = result
                results_df['feature'] = feature
                results_df['target'] = target
                results_df['pearson_r'] = pearson_r
                all_results.append(results_df)
    
    if all_results:
        # 合并所有结果
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # 保存详细结果
        output_file = "/home/ubuntu/project/MSC/Msc_Project/models/detailed_model_comparison.csv"
        combined_results.to_csv(output_file, index=False)
        
        print(f"\n📋 总体分析结果:")
        print("="*80)
        
        # 按模型类型汇总
        summary = combined_results.groupby('model').agg({
            'cv_r2': ['mean', 'std', 'count'],
            'overfitting': ['mean', 'std'],
            'generalization_ratio': ['mean', 'std'],
            'composite_score': ['mean', 'std']
        }).round(4)
        
        print("平均表现汇总:")
        print(summary)
        
        # 关键结论
        simple_performance = combined_results[~combined_results['is_complex']]['composite_score'].mean()
        complex_performance = combined_results[combined_results['is_complex']]['composite_score'].mean()
        
        print(f"\n🎯 关键结论:")
        print(f"📊 简单模型平均综合评分: {simple_performance:.4f}")
        print(f"🔧 复杂模型平均综合评分: {complex_performance:.4f}")
        
        if simple_performance > complex_performance:
            print(f"✅ 结论: 简单模型总体表现更好 (优势: {simple_performance - complex_performance:.4f})")
            print(f"📝 原因: 在小样本数据中，简单模型的泛化能力更强，过拟合风险更低")
        else:
            print(f"⚠️ 结论: 复杂模型表现略好，但需要考虑解释性和稳定性")
        
        print(f"\n💾 详细结果保存至: {output_file}")

if __name__ == '__main__':
    analyze_key_relationships()
