#!/usr/bin/env python3
"""
验证公平性指标公式的正确性
对比实际计算结果与理论公式
"""

import numpy as np
import pandas as pd

def verify_gini_formula():
    """验证基尼系数公式"""
    print("🔍 基尼系数公式验证")
    print("=" * 40)
    
    # 测试数据
    test_data = [100, 200, 150, 300, 50]  # 5个充电站的充电量
    n = len(test_data)
    mu = np.mean(test_data)
    
    print(f"测试数据: {test_data}")
    print(f"充电站数量 n = {n}")
    print(f"平均值 μ = {mu}")
    
    # 方法1: 您提供的公式
    formula1_numerator = 0
    for i in range(n):
        for j in range(n):
            formula1_numerator += abs(test_data[i] - test_data[j])
    
    gini_formula1 = formula1_numerator / (2 * n**2 * mu)
    print(f"\n方法1 (您的公式): Gini = {gini_formula1:.6f}")
    
    # 方法2: 标准排序公式
    sorted_data = np.sort(test_data)
    cumsum = np.cumsum(sorted_data)
    gini_formula2 = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    print(f"方法2 (标准公式): Gini = {gini_formula2:.6f}")
    
    # 方法3: 另一种标准公式
    total = np.sum(test_data)
    if total == 0:
        gini_formula3 = 0
    else:
        rank_sum = 0
        for i, val in enumerate(sorted_data):
            rank_sum += (i + 1) * val
        gini_formula3 = (2 * rank_sum / total - (n + 1)) / n
    print(f"方法3 (替代公式): Gini = {gini_formula3:.6f}")
    
    return gini_formula1, gini_formula2, gini_formula3

def verify_cv_formula():
    """验证变异系数公式"""
    print("\n🔍 变异系数公式验证")
    print("=" * 40)
    
    test_data = [100, 200, 150, 300, 50]
    sigma = np.std(test_data, ddof=0)  # 总体标准差
    mu = np.mean(test_data)
    
    cv = sigma / mu
    print(f"测试数据: {test_data}")
    print(f"标准差 σ = {sigma:.6f}")
    print(f"平均值 μ = {mu:.6f}")
    print(f"变异系数 CV = {cv:.6f}")
    
    return cv

def verify_hhi_formula():
    """验证HHI指数公式"""
    print("\n🔍 HHI指数公式验证")
    print("=" * 40)
    
    test_data = [100, 200, 150, 300, 50]
    total = np.sum(test_data)
    
    # 计算市场份额
    shares = [x / total for x in test_data]
    print(f"测试数据: {test_data}")
    print(f"总量: {total}")
    print(f"市场份额: {[f'{s:.4f}' for s in shares]}")
    
    # 计算HHI
    hhi = sum(s**2 for s in shares)
    print(f"HHI = Σ(sᵢ)² = {hhi:.6f}")
    
    return hhi

def main():
    """主函数"""
    print("📊 公平性指标公式验证与修正")
    print("=" * 50)
    
    # 验证各个公式
    gini1, gini2, gini3 = verify_gini_formula()
    cv = verify_cv_formula()
    hhi = verify_hhi_formula()
    
    print(f"\n📋 验证结果总结:")
    print(f"   • 基尼系数 (您的公式): {gini1:.6f}")
    print(f"   • 基尼系数 (标准公式): {gini2:.6f}")
    print(f"   • 变异系数: {cv:.6f}")
    print(f"   • HHI指数: {hhi:.6f}")
    
    # 检查一致性
    if abs(gini1 - gini2) < 0.0001:
        print(f"\n✅ 基尼系数公式一致性检验通过")
    else:
        print(f"\n⚠️ 基尼系数公式存在差异，需要修正")

if __name__ == '__main__':
    main()
