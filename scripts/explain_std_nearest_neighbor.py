#!/usr/bin/env python3
"""
解释std_nearest_neighbor特征的计算方式和含义
分析当前实现与理论定义的差异
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def demonstrate_std_nearest_neighbor():
    """演示std_nearest_neighbor的不同计算方式"""
    
    print("🔍 std_nearest_neighbor 特征分析")
    print("="*60)
    
    # 创建示例充电桩坐标
    cs_coords = np.array([
        [0, 0],     # 充电桩1
        [1, 0],     # 充电桩2  
        [0, 1],     # 充电桩3
        [2, 2],     # 充电桩4 (较远)
        [2.1, 2.1]  # 充电桩5 (紧邻充电桩4)
    ])
    
    print(f"📊 示例数据: {len(cs_coords)} 个充电桩")
    for i, coord in enumerate(cs_coords, 1):
        print(f"   充电桩{i}: ({coord[0]}, {coord[1]})")
    
    print(f"\n1️⃣ 当前脚本的实际计算方式:")
    print("-" * 40)
    
    # 当前脚本的计算方式
    pairwise_dists = pdist(cs_coords)
    current_avg = pairwise_dists.mean()
    current_std = pairwise_dists.std()
    current_min = pairwise_dists.min()
    
    print(f"所有两两距离: {pairwise_dists}")
    print(f"两两距离数量: {len(pairwise_dists)} 个")
    print(f"avg_nearest_neighbor (当前): {current_avg:.4f}")
    print(f"std_nearest_neighbor (当前): {current_std:.4f}")
    print(f"min_distance (当前): {current_min:.4f}")
    
    print(f"\n⚠️ 问题: 这不是真正的'最近邻'概念!")
    print(f"   - 计算了所有两两距离，不只是最近邻")
    print(f"   - 包含了远距离对，会被极值影响")
    
    print(f"\n2️⃣ 真正的最近邻距离计算:")
    print("-" * 40)
    
    # 正确的最近邻距离计算
    distance_matrix = squareform(pairwise_dists)
    np.fill_diagonal(distance_matrix, np.inf)  # 排除自己到自己的距离
    
    # 每个点的最近邻距离
    nearest_neighbor_dists = np.min(distance_matrix, axis=1)
    
    true_avg_nn = nearest_neighbor_dists.mean()
    true_std_nn = nearest_neighbor_dists.std()
    true_min_nn = nearest_neighbor_dists.min()
    
    print(f"距离矩阵形状: {distance_matrix.shape}")
    print(f"每个点的最近邻距离: {nearest_neighbor_dists}")
    print(f"真正的 avg_nearest_neighbor: {true_avg_nn:.4f}")
    print(f"真正的 std_nearest_neighbor: {true_std_nn:.4f}")
    print(f"真正的 min_nearest_neighbor: {true_min_nn:.4f}")
    
    print(f"\n3️⃣ 两种方法的差异:")
    print("-" * 40)
    print(f"平均值差异: {abs(current_avg - true_avg_nn):.4f}")
    print(f"标准差差异: {abs(current_std - true_std_nn):.4f}")
    print(f"最小值差异: {abs(current_min - true_min_nn):.4f}")
    
    print(f"\n4️⃣ 特征含义解释:")
    print("-" * 40)
    print("🔧 当前实现 (两两距离标准差):")
    print("   - 衡量所有充电桩对之间距离的离散程度")
    print("   - 反映整体布局的空间分散性")
    print("   - 包含了所有距离信息，更全面但可能被极值影响")
    
    print("\n🎯 理论定义 (最近邻距离标准差):")
    print("   - 衡量每个充电桩到其最近邻居距离的离散程度")
    print("   - 反映局部密度分布的均匀性")
    print("   - 更关注局部邻域关系，对极值不敏感")
    
    print(f"\n5️⃣ 在论文中的应用建议:")
    print("-" * 40)
    print("📝 当前变量名存在误导，建议:")
    print("   - 将 'std_nearest_neighbor' 重命名为 'std_pairwise_distance'")
    print("   - 或添加真正的最近邻标准差作为新特征")
    print("   - 在论文中明确说明计算方式")
    
    return {
        'current_method': {
            'avg': current_avg,
            'std': current_std,
            'min': current_min,
            'distances': pairwise_dists
        },
        'true_nearest_neighbor': {
            'avg': true_avg_nn,
            'std': true_std_nn,
            'min': true_min_nn,
            'distances': nearest_neighbor_dists
        }
    }

def analyze_real_data():
    """分析真实数据中的情况"""
    
    print(f"\n6️⃣ 真实数据分析:")
    print("="*60)
    
    # 加载真实数据
    try:
        import pandas as pd
        df = pd.read_csv("/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv")
        
        # 选择一个示例
        std_nn_values = df['std_nearest_neighbor'].values
        print(f"📊 真实数据中的 std_nearest_neighbor:")
        print(f"   样本数量: {len(std_nn_values)}")
        print(f"   平均值: {np.mean(std_nn_values):.4f}")
        print(f"   标准差: {np.std(std_nn_values):.4f}")
        print(f"   最小值: {np.min(std_nn_values):.4f}")
        print(f"   最大值: {np.max(std_nn_values):.4f}")
        
        # 分析与其他变量的关系
        correlation_with_avg = np.corrcoef(df['avg_nearest_neighbor'], df['std_nearest_neighbor'])[0,1]
        correlation_with_min = np.corrcoef(df['min_distance'], df['std_nearest_neighbor'])[0,1]
        
        print(f"\n📈 相关性分析:")
        print(f"   与 avg_nearest_neighbor 的相关性: {correlation_with_avg:.4f}")
        print(f"   与 min_distance 的相关性: {correlation_with_min:.4f}")
        
        if correlation_with_avg > 0.8:
            print(f"   ⚠️ 与平均距离高度相关，可能存在冗余")
        
        return df
        
    except Exception as e:
        print(f"❌ 无法加载真实数据: {e}")
        return None

def recommend_improvements():
    """推荐改进方案"""
    
    print(f"\n7️⃣ 改进建议:")
    print("="*60)
    
    recommendations = [
        "🔧 **立即修复**: 重命名变量避免混淆",
        "   - std_nearest_neighbor → std_pairwise_distance",
        "   - avg_nearest_neighbor → avg_pairwise_distance",
        "",
        "📊 **增加新特征**: 真正的最近邻统计",
        "   - true_avg_nearest_neighbor: 最近邻距离平均值",
        "   - true_std_nearest_neighbor: 最近邻距离标准差",
        "",
        "📝 **论文中说明**: 明确特征定义",
        "   - 在方法论部分清楚定义每个特征的计算方式",
        "   - 说明为什么选择两两距离而非最近邻距离",
        "",
        "🎯 **特征选择**: 根据研究目标选择",
        "   - 如果关注整体布局分散性 → 使用当前方法",
        "   - 如果关注局部密度均匀性 → 使用真正最近邻",
        "",
        "⚖️ **两种特征都保留**: 提供更丰富的信息",
        "   - pairwise_distance_std: 整体分散性",
        "   - nearest_neighbor_std: 局部均匀性"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == '__main__':
    # 演示计算差异
    results = demonstrate_std_nearest_neighbor()
    
    # 分析真实数据
    df = analyze_real_data()
    
    # 提供改进建议
    recommend_improvements()
    
    print(f"\n🎉 分析完成!")
    print(f"💡 关键发现: 当前的'std_nearest_neighbor'实际上是'std_pairwise_distance'")
    print(f"📋 建议在论文中明确说明特征的实际计算方式")
