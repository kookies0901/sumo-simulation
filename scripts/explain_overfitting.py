#!/usr/bin/env python3
"""
过拟合现象解释 - 为什么高R²可能是问题
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, LeaveOneOut

def demonstrate_overfitting():
    """演示过拟合现象"""
    
    # 1. 生成真实的简单关系数据
    np.random.seed(42)
    n_samples = 20  # 小样本，类似您的数据
    X_true = np.linspace(0, 10, n_samples)
    
    # 真实关系：简单线性关系 + 少量噪声
    y_true = 2 * X_true + 3 + np.random.normal(0, 2, n_samples)
    
    X_true = X_true.reshape(-1, 1)
    
    print("🔍 过拟合现象演示")
    print(f"样本数量: {n_samples}")
    print("真实关系: y = 2x + 3 + 噪声")
    
    # 2. 比较不同复杂度的模型
    models = {
        '线性回归': (PolynomialFeatures(1), 'blue'),
        '2次多项式': (PolynomialFeatures(2), 'red'),
        '5次多项式': (PolynomialFeatures(5), 'green'),
        '10次多项式': (PolynomialFeatures(10), 'purple')
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, (poly, color)) in enumerate(models.items(), 1):
        plt.subplot(2, 2, i)
        
        # 转换特征
        X_poly = poly.fit_transform(X_true)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_poly, y_true)
        
        # 训练R²
        y_pred_train = model.predict(X_poly)
        train_r2 = r2_score(y_true, y_pred_train)
        
        # 交叉验证R²
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([('poly', poly), ('linear', LinearRegression())])
        cv_scores = cross_val_score(pipeline, X_true, y_true, cv=LeaveOneOut(), scoring='r2')
        cv_r2 = cv_scores.mean()
        
        # 过拟合程度
        overfitting = train_r2 - cv_r2
        
        # 绘制原始数据
        plt.scatter(X_true, y_true, alpha=0.7, s=50, color='black', label='原始数据')
        
        # 绘制拟合曲线
        X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
        X_plot_poly = poly.transform(X_plot)
        y_plot = model.predict(X_plot_poly)
        plt.plot(X_plot, y_plot, color=color, linewidth=2, label=f'{name}拟合')
        
        # 绘制真实关系
        y_true_line = 2 * X_plot.flatten() + 3
        plt.plot(X_plot, y_true_line, '--', color='orange', alpha=0.8, label='真实关系')
        
        plt.title(f'{name}\n训练R²={train_r2:.3f}, CV R²={cv_r2:.3f}\n过拟合程度={overfitting:.3f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 判断质量
        if overfitting > 0.3:
            quality = "过拟合严重"
            color_bg = 'red'
        elif cv_r2 > 0.3 and overfitting < 0.1:
            quality = "良好"
            color_bg = 'green'
        else:
            quality = "一般"
            color_bg = 'yellow'
        
        plt.text(0.02, 0.98, f'质量: {quality}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor=color_bg, alpha=0.3),
                verticalalignment='top')
        
        print(f"\n{name}:")
        print(f"  训练 R²: {train_r2:.3f}")
        print(f"  交叉验证 R²: {cv_r2:.3f}")
        print(f"  过拟合程度: {overfitting:.3f}")
        print(f"  质量评估: {quality}")
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/overfitting_demonstration.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 结论:")
    print(f"1. 高次多项式在训练数据上R²很高，但在新数据上表现很差")
    print(f"2. 这就是为什么高R²可能表示过拟合")
    print(f"3. 交叉验证R²更能反映模型的真实泛化能力")
    print(f"4. 过拟合程度 = 训练R² - 交叉验证R² > 0.3 时需要警惕")

def explain_your_data_issue():
    """解释您数据中的过拟合问题"""
    
    print(f"\n🎯 您的数据中的过拟合问题解释:")
    
    # 模拟您的数据情况
    scenarios = [
        {
            'name': 'cs_density_std -> charging_time_mean',
            'train_r2': 0.4837,
            'cv_r2': 0.0000,
            'sample_size': 81
        },
        {
            'name': 'cluster_count -> charging_station_coverage',
            'train_r2': 0.5522,
            'cv_r2': 0.0000,
            'sample_size': 81
        }
    ]
    
    for scenario in scenarios:
        overfitting = scenario['train_r2'] - scenario['cv_r2']
        print(f"\n案例: {scenario['name']}")
        print(f"  样本量: {scenario['sample_size']}")
        print(f"  训练 R²: {scenario['train_r2']:.4f}  <- 看起来很好！")
        print(f"  交叉验证 R²: {scenario['cv_r2']:.4f}  <- 实际泛化能力")
        print(f"  过拟合程度: {overfitting:.4f}  <- 超过0.3，严重过拟合")
        
        print(f"  📝 分析:")
        print(f"     - 模型在训练数据上'表现优秀'({scenario['train_r2']:.1%})")
        print(f"     - 但对新数据的预测能力为0")
        print(f"     - 这意味着模型学到的是噪声，而非真实规律")
        print(f"     - 在81个样本中，这种过拟合风险很高")

if __name__ == '__main__':
    demonstrate_overfitting()
    explain_your_data_issue()
    print(f"\n💡 这就是为什么您需要插值来增加样本量的原因！")
    print(f"📈 图表保存为: overfitting_demonstration.png")
