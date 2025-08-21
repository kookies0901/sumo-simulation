#!/usr/bin/env python3
"""
趋势展示 vs 预测建模的区别分析
解释为什么用于趋势展示时，过拟合的重要性会降低
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = ['DejaVu Sans']  # 使用系统默认字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def demonstrate_trend_vs_prediction():
    """演示趋势展示与预测建模的不同需求"""
    
    print("🎯 核心问题：趋势展示 vs 预测建模 - 是否需要在意过拟合？")
    print("="*80)
    
    # 加载实际数据
    data_file = "/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv"
    df = pd.read_csv(data_file)
    
    feature = 'cs_density_std'
    target = 'charging_time_mean'
    
    x = df[feature].values
    y = df[target].values
    
    # 移除NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    X = x_clean.reshape(-1, 1)
    
    print(f"📊 分析案例: {feature} vs {target}")
    print(f"📈 样本数量: {len(x_clean)}")
    
    # 计算皮尔逊相关系数
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    print(f"📈 皮尔逊相关系数: {correlation:.4f}")
    
    # 定义模型
    models = {
        'Linear': LinearRegression(),
        'Polynomial': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    }
    
    # 创建图形比较
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trend Display vs Prediction Modeling: Different Meanings of Overfitting', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green']
    
    for i, (name, model) in enumerate(models.items()):
        # 训练模型
        model.fit(X, y_clean)
        
        # 训练R²
        y_pred_train = model.predict(X)
        train_r2 = r2_score(y_clean, y_pred_train)
        
        # 交叉验证R²
        cv_scores = cross_val_score(model, X, y_clean, cv=LeaveOneOut(), scoring='r2')
        cv_r2 = cv_scores.mean()
        overfitting = train_r2 - cv_r2
        
        # 生成平滑曲线用于绘图
        x_smooth = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
        y_smooth = model.predict(x_smooth)
        
        # 第一行：趋势展示视角
        ax1 = axes[0, i]
        ax1.scatter(x_clean, y_clean, alpha=0.7, s=60, color='gray', edgecolors='black')
        ax1.plot(x_smooth, y_smooth, color=colors[i], linewidth=3, label=f'{name} Fit')
        ax1.set_title(f'Trend Display: {name}\nTrain R²={train_r2:.3f}', fontweight='bold')
        ax1.set_xlabel(feature)
        ax1.set_ylabel(target)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加趋势判断
        if correlation > 0.3:
            trend_text = "Strong Positive Trend"
            color_trend = 'green'
        elif correlation < -0.3:
            trend_text = "Strong Negative Trend"
            color_trend = 'green'
        else:
            trend_text = "Weak Correlation"
            color_trend = 'orange'
        
        ax1.text(0.05, 0.95, trend_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor=color_trend, alpha=0.3),
                verticalalignment='top')
        
        # 第二行：预测建模视角
        ax2 = axes[1, i]
        ax2.scatter(x_clean, y_clean, alpha=0.7, s=60, color='gray', edgecolors='black')
        ax2.plot(x_smooth, y_smooth, color=colors[i], linewidth=3, label=f'{name} Fit')
        ax2.set_title(f'Prediction Modeling: {name}\nCV R²={cv_r2:.3f}, Overfitting={overfitting:.3f}', fontweight='bold')
        ax2.set_xlabel(feature)
        ax2.set_ylabel(target)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加预测能力判断
        if cv_r2 > 0.3:
            pred_text = "Strong Prediction"
            color_pred = 'green'
        elif cv_r2 > 0.1:
            pred_text = "Moderate Prediction"
            color_pred = 'orange'
        else:
            pred_text = "Poor Prediction"
            color_pred = 'red'
        
        ax2.text(0.05, 0.95, pred_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor=color_pred, alpha=0.3),
                verticalalignment='top')
        
        print(f"\n{name} 模型:")
        print(f"  训练 R²: {train_r2:.4f}")
        print(f"  交叉验证 R²: {cv_r2:.4f}")
        print(f"  过拟合程度: {overfitting:.4f}")
        print(f"  趋势展示评价: {'适合' if train_r2 > 0.3 else '不太适合'}")
        print(f"  预测建模评价: {'适合' if cv_r2 > 0.3 else '不适合'}")
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/trend_vs_prediction_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation

def analyze_trend_requirements():
    """分析趋势展示的具体要求"""
    
    print(f"\n1️⃣ 趋势展示的核心要求:")
    print("-" * 60)
    
    requirements = {
        "视觉清晰度": {
            "描述": "读者能否清楚看到变量间的关系方向？",
            "评判标准": "拟合线的斜率方向与数据点分布一致",
            "过拟合影响": "影响很小，只要方向正确"
        },
        "相关性强度": {
            "描述": "关系的强弱程度是否准确反映？",
            "评判标准": "R²值反映关系紧密程度",
            "过拟合影响": "训练R²已足够反映关系强度"
        },
        "数据覆盖": {
            "描述": "拟合线是否覆盖所有数据范围？",
            "评判标准": "曲线通过数据点的分布区域",
            "过拟合影响": "无影响，所有模型都能覆盖"
        },
        "平滑性": {
            "描述": "拟合线是否平滑美观？",
            "评判标准": "无异常波动，视觉效果好",
            "过拟合影响": "轻微影响，过拟合可能产生局部波动"
        }
    }
    
    for req, details in requirements.items():
        print(f"📊 {req}:")
        print(f"   {details['描述']}")
        print(f"   标准: {details['评判标准']}")
        print(f"   过拟合影响: {details['过拟合影响']}")
        print()

def analyze_prediction_requirements():
    """分析预测建模的具体要求"""
    
    print(f"\n2️⃣ 预测建模的核心要求:")
    print("-" * 60)
    
    requirements = {
        "泛化能力": {
            "描述": "模型在新数据上的预测准确性",
            "评判标准": "交叉验证R²，测试集表现",
            "过拟合影响": "严重影响！过拟合直接降低泛化能力"
        },
        "预测稳定性": {
            "描述": "预测结果的一致性和可靠性",
            "评判标准": "交叉验证分数的标准差",
            "过拟合影响": "严重影响！过拟合导致预测不稳定"
        },
        "外推能力": {
            "描述": "在数据范围外的预测合理性",
            "评判标准": "外推预测与物理定律一致",
            "过拟合影响": "严重影响！过拟合模型外推不可靠"
        },
        "模型选择": {
            "描述": "选择最适合数据的模型复杂度",
            "评判标准": "偏差-方差平衡",
            "过拟合影响": "核心问题！需要严格控制过拟合"
        }
    }
    
    for req, details in requirements.items():
        print(f"🎯 {req}:")
        print(f"   {details['描述']}")
        print(f"   标准: {details['评判标准']}")
        print(f"   过拟合影响: {details['过拟合影响']}")
        print()

def provide_recommendations():
    """提供针对性建议"""
    
    print(f"\n3️⃣ 针对您研究的建议:")
    print("="*80)
    
    print("🎯 **如果您的目标是趋势展示**：")
    trend_recommendations = [
        "✅ 主要关注训练R²和相关系数",
        "✅ 选择拟合效果好、曲线平滑的模型",
        "✅ RandomForest等复杂模型也可以考虑",
        "✅ 重点是视觉效果和趋势方向的准确性",
        "⚠️ 过拟合不是主要问题，但仍要适度"
    ]
    
    for rec in trend_recommendations:
        print(f"   {rec}")
    
    print(f"\n🔍 **如果您的目标是科学发现**：")
    science_recommendations = [
        "✅ 仍建议使用Linear/Polynomial模型",
        "✅ 关注交叉验证结果和统计显著性",
        "✅ 模型可解释性对科学洞察很重要",
        "✅ 简单模型更容易被学术界接受",
        "⚠️ 需要平衡趋势展示和科学严谨性"
    ]
    
    for rec in science_recommendations:
        print(f"   {rec}")
    
    print(f"\n💡 **最佳实践建议**：")
    best_practices = [
        "📊 在论文中可以这样表述：",
        "   '为了清晰展示变量间的关系趋势，选择了拟合效果较好的XXX模型'",
        "   '模型的主要目的是可视化数据中的模式，而非建立预测工具'",
        "",
        "📈 可以同时展示多个模型的拟合结果：",
        "   '比较了Linear、Polynomial和RandomForest模型的拟合效果'",
        "   '选择视觉效果最佳的模型用于趋势展示'",
        "",
        "🔬 保持学术诚实：",
        "   '明确说明模型用途是趋势展示而非预测'",
        "   '在讨论部分承认模型的局限性'"
    ]
    
    for practice in best_practices:
        print(practice)

if __name__ == '__main__':
    correlation = demonstrate_trend_vs_prediction()
    analyze_trend_requirements()
    analyze_prediction_requirements()
    provide_recommendations()
    
    print(f"\n🎉 结论: 您的观点是正确的！")
    print(f"如果目标是趋势展示，过拟合确实不是核心问题。")
    print(f"关键是诚实地表达您的研究目标和模型用途。")
    print(f"\n📈 图表保存为: trend_vs_prediction_comparison.png")
