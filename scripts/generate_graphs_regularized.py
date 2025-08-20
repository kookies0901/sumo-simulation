#!/usr/bin/env python3
"""
生成充电桩布局特征与性能指标的散点图分析 - 防过拟合版本
针对小样本数据集优化，减少过拟合风险
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, LeaveOneOut
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

def fit_multiple_models_with_cv(x, y):
    """训练多个回归模型并使用交叉验证防止过拟合"""
    try:
        # 移除NaN值
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 5:
            return None, None, 0.0, "insufficient_data", {}
        
        # 重塑数据为sklearn格式
        X = x_clean.reshape(-1, 1)
        
        # 定义针对小样本优化的模型
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),  # 添加正则化
            'Lasso': Lasso(alpha=0.1),  # 添加特征选择
            'Polynomial': Ridge(alpha=1.0),  # 多项式+正则化
            'RandomForest_Small': RandomForestRegressor(
                n_estimators=10,  # 减少树的数量
                max_depth=3,      # 限制深度
                min_samples_split=5,  # 增加分裂要求
                random_state=42
            ),
            'GradientBoosting_Small': GradientBoostingRegressor(
                n_estimators=10,  # 大幅减少树的数量
                max_depth=2,      # 限制深度
                learning_rate=0.3,  # 增加学习率补偿树数减少
                min_samples_split=5,
                random_state=42
            ),
            'SVR_Simple': SVR(kernel='linear', C=1.0)  # 使用线性核
        }
        
        model_results = {}
        best_model = None
        best_cv_score = -np.inf
        best_model_name = ""
        
        # 使用留一交叉验证
        cv = LeaveOneOut()
        
        for name, model in models.items():
            try:
                if name == 'Polynomial':
                    # 多项式回归
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(X)
                    
                    # 交叉验证
                    cv_scores = cross_val_score(model, X_poly, y_clean, cv=cv, scoring='r2')
                    cv_r2 = cv_scores.mean()
                    
                    # 训练完整模型用于可视化
                    model.fit(X_poly, y_clean)
                    y_pred = model.predict(X_poly)
                    train_r2 = r2_score(y_clean, y_pred)
                    
                    # 生成预测曲线
                    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
                    X_fit_poly = poly_features.transform(x_fit)
                    y_fit = model.predict(X_fit_poly)
                else:
                    # 其他模型
                    # 交叉验证
                    cv_scores = cross_val_score(model, X, y_clean, cv=cv, scoring='r2')
                    cv_r2 = cv_scores.mean()
                    
                    # 训练完整模型用于可视化
                    model.fit(X, y_clean)
                    y_pred = model.predict(X)
                    train_r2 = r2_score(y_clean, y_pred)
                    
                    # 生成预测曲线
                    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
                    y_fit = model.predict(x_fit)
                
                # 计算过拟合程度
                overfitting = train_r2 - cv_r2
                
                model_results[name] = {
                    'train_r2': train_r2,
                    'cv_r2': cv_r2,
                    'overfitting': overfitting,
                    'x_fit': x_fit.flatten(),
                    'y_fit': y_fit,
                    'model': model
                }
                
                # 选择最佳模型：优先考虑交叉验证分数，同时惩罚过拟合
                score = cv_r2 - 0.5 * max(0, overfitting)  # 惩罚过拟合
                
                if score > best_cv_score:
                    best_cv_score = score
                    best_model = name
                    best_model_name = name
                    
            except Exception as e:
                print(f"   ⚠️ 模型 {name} 训练失败: {e}")
                continue
        
        if best_model is None:
            return None, None, 0.0, "no_valid_model", {}
        
        # 返回最佳模型的结果，但使用交叉验证R²
        best_result = model_results[best_model]
        return (best_result['x_fit'], best_result['y_fit'], 
                best_result['cv_r2'], best_model_name, model_results)
        
    except Exception as e:
        print(f"⚠️ 多模型拟合失败: {e}")
        return None, None, 0.0, "error", {}

def create_scatter_plot_with_validation(df, x_col, y_col, output_dir):
    """创建带交叉验证信息的散点图"""
    try:
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 获取数据
        x = df[x_col].values
        y = df[y_col].values
        
        # 创建散点图
        scatter = ax.scatter(x, y, alpha=0.7, s=80, color='steelblue', 
                           edgecolors='black', linewidth=0.5)
        
        # 使用改进的多模型拟合
        x_fit, y_fit, cv_r2, best_model, model_results = fit_multiple_models_with_cv(x, y)
        
        # 绘制最佳拟合线
        if x_fit is not None and y_fit is not None:
            # 根据模型类型设置颜色
            color_map = {
                'Linear': 'darkred',
                'Ridge': 'darkgreen',
                'Lasso': 'darkorange', 
                'Polynomial': 'red',
                'RandomForest_Small': 'green',
                'GradientBoosting_Small': 'blue',
                'SVR_Simple': 'purple'
            }
            color = color_map.get(best_model, 'darkred')
            ax.plot(x_fit, y_fit, color=color, linewidth=2.5,
                   label=f'{best_model} (CV R² = {cv_r2:.3f})')
            
            # 添加详细的模型比较信息
            if len(model_results) > 1:
                info_text = f"最佳模型: {best_model}\\n"
                info_text += f"交叉验证 R²: {cv_r2:.3f}\\n"
                
                best_result = model_results[best_model]
                info_text += f"训练 R²: {best_result['train_r2']:.3f}\\n"
                info_text += f"过拟合程度: {best_result['overfitting']:.3f}\\n"
                
                # 显示前3个模型的交叉验证结果
                sorted_models = sorted(model_results.items(), 
                                     key=lambda x: x[1]['cv_r2'], reverse=True)
                info_text += "\\nTop 3 (CV R²):\\n"
                for i, (name, result) in enumerate(sorted_models[:3]):
                    info_text += f"{i+1}. {name}: {result['cv_r2']:.3f}\\n"
                
                # 创建文本框
                ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 设置标签和标题
        ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
        ax.set_title(f'{x_col} vs {y_col}\\n(防过拟合版本)', fontsize=14, fontweight='bold', pad=20)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 添加图例
        if x_fit is not None:
            ax.legend(loc='upper right', framealpha=0.8)
        
        # 设置样式
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        filename = f"{x_col}_{y_col}_regularized.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath, cv_r2, best_model
        
    except Exception as e:
        print(f"❌ 创建图表失败 {x_col} vs {y_col}: {e}")
        plt.close()
        return None, 0.0, "error"

def generate_regularized_plots(df, feature_cols, performance_cols, output_dir):
    """生成防过拟合的散点图"""
    print(f"\\n🎨 开始生成防过拟合图表...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储结果统计
    results = []
    success_count = 0
    total_count = len(feature_cols) * len(performance_cols)
    
    # 创建PDF合集
    pdf_path = os.path.join(output_dir, "regularized_scatter_plots.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for i, x_col in enumerate(feature_cols, 1):
            print(f"\\n📊 处理特征变量 [{i}/{len(feature_cols)}]: {x_col}")
            
            for j, y_col in enumerate(performance_cols[:5]):  # 先处理前5个性能指标作为测试
                print(f"   📈 [{j+1}/5] {y_col}...", end="")
                
                try:
                    # 创建图形用于PDF
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # 获取数据
                    x = df[x_col].values
                    y = df[y_col].values
                    
                    # 创建散点图
                    ax.scatter(x, y, alpha=0.7, s=80, color='steelblue', 
                             edgecolors='black', linewidth=0.5)
                    
                    # 使用改进的多模型拟合
                    x_fit, y_fit, cv_r2, best_model, model_results = fit_multiple_models_with_cv(x, y)
                    
                    # 绘制最佳拟合线
                    if x_fit is not None and y_fit is not None:
                        color_map = {
                            'Linear': 'darkred',
                            'Ridge': 'darkgreen',
                            'Lasso': 'darkorange', 
                            'Polynomial': 'red',
                            'RandomForest_Small': 'green',
                            'GradientBoosting_Small': 'blue',
                            'SVR_Simple': 'purple'
                        }
                        color = color_map.get(best_model, 'darkred')
                        ax.plot(x_fit, y_fit, color=color, linewidth=2.5,
                               label=f'{best_model} (CV R² = {cv_r2:.3f})')
                        
                        # 添加详细信息
                        if len(model_results) > 1:
                            best_result = model_results[best_model]
                            info_text = f"Best: {best_model}\\n"
                            info_text += f"CV R²: {cv_r2:.3f}\\n"
                            info_text += f"Train R²: {best_result['train_r2']:.3f}\\n"
                            info_text += f"Overfitting: {best_result['overfitting']:.3f}"
                            
                            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                                   fontsize=9, verticalalignment='top', 
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    
                    # 设置标签和标题
                    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
                    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
                    ax.set_title(f'{x_col} vs {y_col} (Regularized)', fontsize=14, fontweight='bold', pad=20)
                    
                    # 添加网格和图例
                    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    if x_fit is not None:
                        ax.legend(loc='upper right', framealpha=0.8)
                    
                    # 设置样式
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # 调整布局
                    plt.tight_layout()
                    
                    # 保存到PDF
                    pdf.savefig(fig, dpi=300, bbox_inches='tight')
                    
                    # 保存单独的PNG文件
                    filename = f"{x_col}_{y_col}_regularized.png"
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    # 记录结果
                    if len(model_results) > 0 and best_model in model_results:
                        best_result = model_results[best_model]
                        results.append({
                            'feature': x_col,
                            'performance': y_col,
                            'cv_r2': cv_r2,
                            'train_r2': best_result.get('train_r2', 0),
                            'overfitting': best_result.get('overfitting', 0),
                            'best_model': best_model,
                            'filename': filename
                        })
                    
                    success_count += 1
                    print(f" ✅ (CV R²={cv_r2:.3f})")
                    
                except Exception as e:
                    print(f" ❌ 失败: {e}")
                    plt.close()
                    continue
    
    # 保存结果统计
    if results:
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, "regularized_results_summary.csv")
        results_df.to_csv(results_csv, index=False)
        
        print(f"\\n🎉 防过拟合图表生成完成！")
        print(f"✅ 成功生成: {success_count} 张图表")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 PDF合集: {pdf_path}")
        print(f"📊 结果统计: {results_csv}")
        
        # 显示改进的统计信息
        if len(results_df) > 0:
            print(f"\\n📈 交叉验证质量统计:")
            print(f"   - 平均 CV R²: {results_df['cv_r2'].mean():.3f}")
            print(f"   - 最高 CV R²: {results_df['cv_r2'].max():.3f}")
            print(f"   - CV R² > 0.3: {len(results_df[results_df['cv_r2'] > 0.3])} 张")
            print(f"   - 平均过拟合程度: {results_df['overfitting'].mean():.3f}")
            
            print(f"\\n🎯 最佳模型分布:")
            model_counts = results_df['best_model'].value_counts()
            for model, count in model_counts.items():
                avg_cv_r2 = results_df[results_df['best_model'] == model]['cv_r2'].mean()
                avg_overfitting = results_df[results_df['best_model'] == model]['overfitting'].mean()
                print(f"   - {model}: {count} 张 (CV R²={avg_cv_r2:.3f}, 过拟合={avg_overfitting:.3f})")
    
    return results_df if results else pd.DataFrame()

def main():
    print("🚀 开始生成防过拟合的充电桩布局特征与性能指标散点图")
    
    # 设置路径
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input/merged_dataset.csv"
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/models/plots_regularized"
    
    print(f"📊 数据文件: {data_file}")
    print(f"📁 输出目录: {output_dir}")
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return 1
    
    # 加载数据
    try:
        df = pd.read_csv(data_file)
        print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return 1
    
    # 定义特征和性能指标
    layout_features = [col for col in df.columns 
                      if col not in ['layout_id'] and 
                      not col.startswith(('duration_', 'charging_', 'waiting_', 'energy_', 'vehicle_', 'reroute_', 'ev_'))]
    
    performance_metrics = [col for col in df.columns 
                          if col.startswith(('duration_', 'charging_', 'waiting_', 'energy_', 'vehicle_', 'reroute_', 'ev_'))]
    
    if not layout_features or not performance_metrics:
        print("❌ 没有找到有效的特征或性能指标列")
        return 1
    
    print(f"📊 找到 {len(layout_features)} 个布局特征")
    print(f"📈 找到 {len(performance_metrics)} 个性能指标")
    
    # 生成防过拟合图表
    results_df = generate_regularized_plots(df, layout_features, performance_metrics, output_dir)
    
    if len(results_df) > 0:
        print(f"\\n🎓 防过拟合图表已生成完毕！")
        print(f"📁 所有图表保存在: {output_dir}")
        print(f"📝 图表命名规则: 特征变量_性能指标_regularized.png")
        print(f"📑 PDF合集和结果统计可直接用于论文")
    else:
        print("❌ 没有成功生成任何图表")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
