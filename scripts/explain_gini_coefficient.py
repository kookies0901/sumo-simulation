#!/usr/bin/env python3
"""
演示 gini_coefficient (基尼系数) 的计算过程
衡量充电桩服务可达性的不平等程度
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def calculate_gini_coefficient(values):
    """计算基尼系数"""
    try:
        if len(values) == 0:
            return 0.0
        
        # 排序
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # 计算基尼系数
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
        
    except:
        return 0.0

def create_sample_road_network():
    """创建示例道路网络"""
    road_points = []
    
    # 创建网格状道路网络
    for y in range(500, 5500, 300):
        for x in range(500, 5500, 200):
            road_points.append((x, y))
    
    return np.array(road_points)

def calculate_gini_demo(cs_coords, road_coords, title="Gini Coefficient Analysis"):
    """演示基尼系数的计算过程"""
    
    # 设置字体
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 计算每个道路点到最近充电桩的距离
    distances_to_cs = []
    for road_point in road_coords:
        dists = [np.linalg.norm(road_point - cs_coord) for cs_coord in cs_coords]
        distances_to_cs.append(min(dists))
    
    distances_to_cs = np.array(distances_to_cs)
    
    # 1. 原始分布可视化
    scatter = ax1.scatter(road_coords[:, 0], road_coords[:, 1], 
                         c=distances_to_cs, s=40, alpha=0.8, 
                         cmap='RdYlGn_r', edgecolors='black', linewidth=0.3)
    ax1.scatter(cs_coords[:, 0], cs_coords[:, 1], c='blue', s=200, alpha=0.9, 
                edgecolors='white', linewidth=3, marker='*', label='Charging Stations')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Distance to Nearest CS (m)')
    
    ax1.set_title('Step 1: Service Accessibility Distribution')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 距离分布直方图
    bins = np.linspace(0, np.max(distances_to_cs) * 1.1, 15)
    counts, bin_edges = np.histogram(distances_to_cs, bins=bins)
    
    ax2.hist(distances_to_cs, bins=bins, alpha=0.7, color='lightblue', 
             edgecolor='black', linewidth=1, density=False)
    
    # 添加统计线
    mean_dist = np.mean(distances_to_cs)
    median_dist = np.median(distances_to_cs)
    ax2.axvline(mean_dist, color='red', linestyle='-', linewidth=2, 
                label=f'Mean: {mean_dist:.0f}m')
    ax2.axvline(median_dist, color='green', linestyle='-', linewidth=2, 
                label=f'Median: {median_dist:.0f}m')
    
    ax2.set_title('Step 2: Distance Distribution')
    ax2.set_xlabel('Distance to Nearest CS (m)')
    ax2.set_ylabel('Number of Road Points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 洛伦兹曲线 (Lorenz Curve)
    sorted_distances = np.sort(distances_to_cs)
    n = len(sorted_distances)
    
    # 计算累积比例
    cumulative_population = np.arange(1, n + 1) / n
    cumulative_distances = np.cumsum(sorted_distances) / np.sum(sorted_distances)
    
    # 绘制洛伦兹曲线
    ax3.plot([0] + cumulative_population.tolist(), [0] + cumulative_distances.tolist(), 
             'b-', linewidth=3, label='Lorenz Curve (Actual)')
    
    # 绘制完全平等线
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Equality Line')
    
    # 填充不平等面积
    lorenz_x = [0] + cumulative_population.tolist()
    lorenz_y = [0] + cumulative_distances.tolist()
    equality_x = [0, 1]
    equality_y = [0, 1]
    
    # 创建不平等区域的多边形
    inequality_x = lorenz_x + [1, 0]
    inequality_y = lorenz_y + [1, 0]
    
    ax3.fill_between(cumulative_population, cumulative_distances, cumulative_population, 
                     alpha=0.3, color='red', label='Inequality Area')
    
    # 计算基尼系数
    gini = calculate_gini_coefficient(distances_to_cs)
    
    ax3.set_title(f'Step 3: Lorenz Curve\nGini Coefficient = {gini:.3f}')
    ax3.set_xlabel('Cumulative Population Proportion')
    ax3.set_ylabel('Cumulative Distance Proportion')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # 添加基尼系数解释
    gini_text = f'Gini = Area between curves / Area under equality line\n'
    gini_text += f'Range: 0 (perfect equality) to 1 (maximum inequality)\n'
    gini_text += f'Current value: {gini:.3f}'
    ax3.text(0.05, 0.95, gini_text, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontsize=9)
    
    # 4. 基尼系数分解分析
    # 按距离区间分析不平等贡献
    distance_ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, np.inf)]
    range_labels = ['<500m', '500-1000m', '1000-1500m', '>1500m']
    range_colors = ['green', 'yellow', 'orange', 'red']
    
    range_counts = []
    range_proportions = []
    
    for (min_dist, max_dist) in distance_ranges:
        if max_dist == np.inf:
            mask = distances_to_cs >= min_dist
        else:
            mask = (distances_to_cs >= min_dist) & (distances_to_cs < max_dist)
        
        count = np.sum(mask)
        proportion = count / len(distances_to_cs)
        range_counts.append(count)
        range_proportions.append(proportion)
    
    # 创建饼图显示不平等分布
    ax4.pie(range_proportions, labels=range_labels, colors=range_colors, 
            autopct='%1.1f%%', startangle=90, explode=[0, 0.05, 0.1, 0.15])
    ax4.set_title('Step 4: Service Inequality Breakdown')
    
    # 添加基尼系数解释
    interpretation = ""
    if gini < 0.2:
        interpretation = "Low inequality - Very uniform service"
    elif gini < 0.4:
        interpretation = "Moderate inequality - Reasonably uniform service"
    elif gini < 0.6:
        interpretation = "High inequality - Uneven service distribution"
    else:
        interpretation = "Very high inequality - Severe service gaps"
    
    ax4.text(0, -1.3, f'Gini Interpretation: {interpretation}', 
             ha='center', va='center', fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # 添加详细统计信息
    stats_text = f'''Gini Coefficient Analysis:
Value: {gini:.3f}
Interpretation: {interpretation}

Distance Statistics:
Mean: {np.mean(distances_to_cs):.1f}m
Median: {np.median(distances_to_cs):.1f}m
Std Dev: {np.std(distances_to_cs):.1f}m
Min: {np.min(distances_to_cs):.1f}m
Max: {np.max(distances_to_cs):.1f}m

Service Distribution:
Excellent (<500m): {range_counts[0]} ({range_proportions[0]*100:.1f}%)
Good (500-1000m): {range_counts[1]} ({range_proportions[1]*100:.1f}%)
Fair (1000-1500m): {range_counts[2]} ({range_proportions[2]*100:.1f}%)
Poor (>1500m): {range_counts[3]} ({range_proportions[3]*100:.1f}%)'''
    
    # 在图外添加统计信息
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom')
    
    plt.suptitle(f'{title}\nGini Coefficient = {gini:.3f} ({interpretation})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    return gini, distances_to_cs

def demo_different_inequality_scenarios():
    """演示不同不平等程度的场景"""
    
    road_network = create_sample_road_network()
    
    # 场景1：低不平等（均匀分布）
    scenario1_cs = np.array([
        [1500, 1500], [3000, 1500], [4500, 1500],
        [1500, 3000], [3000, 3000], [4500, 3000],
        [1500, 4500], [3000, 4500], [4500, 4500]
    ])
    
    # 场景2：中等不平等（部分集中）
    scenario2_cs = np.array([
        [2000, 2000], [2500, 2000], [2000, 2500], [2500, 2500],
        [4000, 4000], [4500, 4000]
    ])
    
    # 场景3：高不平等（严重集中）
    scenario3_cs = np.array([
        [2800, 2800], [3000, 3000], [3200, 3200], [3000, 2800]
    ])
    
    scenarios = [
        (scenario1_cs, "Low Inequality (Uniform)"),
        (scenario2_cs, "Moderate Inequality (Mixed)"),
        (scenario3_cs, "High Inequality (Clustered)")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    results = []
    
    for i, (cs_coords, title) in enumerate(scenarios):
        # 上排：空间分布
        ax_space = axes[0, i]
        # 下排：洛伦兹曲线
        ax_lorenz = axes[1, i]
        
        # 计算距离和基尼系数
        distances = []
        for road_point in road_network:
            dists = [np.linalg.norm(road_point - cs_coord) for cs_coord in cs_coords]
            distances.append(min(dists))
        
        distances = np.array(distances)
        gini = calculate_gini_coefficient(distances)
        
        # 绘制空间分布
        scatter = ax_space.scatter(road_network[:, 0], road_network[:, 1], 
                                 c=distances, s=25, alpha=0.7, 
                                 cmap='RdYlGn_r', edgecolors='gray', linewidth=0.2)
        ax_space.scatter(cs_coords[:, 0], cs_coords[:, 1], c='blue', s=100, alpha=0.9, 
                        edgecolors='white', linewidth=2, marker='*')
        
        ax_space.set_title(f'{title}\nGini = {gini:.3f}')
        ax_space.set_xlabel('X (m)')
        ax_space.set_ylabel('Y (m)' if i == 0 else '')
        ax_space.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax_space, shrink=0.8)
        if i == 2:  # 只在最右边添加标签
            cbar.set_label('Distance (m)')
        
        # 绘制洛伦兹曲线
        sorted_distances = np.sort(distances)
        n = len(sorted_distances)
        cumulative_population = np.arange(1, n + 1) / n
        cumulative_distances = np.cumsum(sorted_distances) / np.sum(sorted_distances)
        
        ax_lorenz.plot([0] + cumulative_population.tolist(), 
                      [0] + cumulative_distances.tolist(), 
                      'b-', linewidth=2, label='Lorenz Curve')
        ax_lorenz.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Equality Line')
        
        # 填充不平等区域
        ax_lorenz.fill_between(cumulative_population, cumulative_distances, 
                              cumulative_population, alpha=0.3, color='red')
        
        ax_lorenz.set_title(f'Lorenz Curve (Gini = {gini:.3f})')
        ax_lorenz.set_xlabel('Cumulative Population')
        ax_lorenz.set_ylabel('Cumulative Distance' if i == 0 else '')
        ax_lorenz.grid(True, alpha=0.3)
        ax_lorenz.set_xlim(0, 1)
        ax_lorenz.set_ylim(0, 1)
        
        if i == 0:
            ax_lorenz.legend()
        
        results.append((title, gini, len(cs_coords)))
    
    plt.suptitle('Gini Coefficient Comparison for Different Service Inequality Levels', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/gini_scenarios.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 基尼系数比较:")
    for title, gini, cs_count in results:
        interpretation = ""
        if gini < 0.2:
            interpretation = "低不平等"
        elif gini < 0.4:
            interpretation = "中等不平等"
        elif gini < 0.6:
            interpretation = "高不平等"
        else:
            interpretation = "极高不平等"
        
        print(f"   {title}:")
        print(f"     基尼系数: {gini:.3f} ({interpretation})")
        print(f"     充电桩数: {cs_count}个")
        print()

if __name__ == "__main__":
    print("🎯 演示基尼系数 (gini_coefficient) 的计算过程")
    print("📚 衡量充电桩服务可达性的不平等程度\n")
    
    # 创建示例数据
    road_network = create_sample_road_network()
    
    # 示例充电桩布局（故意创造不平等）
    example_cs = np.array([
        [2000, 2000], [2200, 2200], [2400, 2000], [2000, 2400],  # 密集区域
        [4500, 4500]  # 孤立点
    ])
    
    print(f"📊 演示数据:")
    print(f"   道路采样点: {len(road_network)} 个")
    print(f"   充电桩数量: {len(example_cs)} 个")
    
    # 详细演示计算过程
    print(f"\n📊 详细计算过程演示...")
    plt.figure(figsize=(16, 12))
    
    gini, distances = calculate_gini_demo(
        example_cs, 
        road_network,
        title="Gini Coefficient Analysis for Service Accessibility"
    )
    
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/gini_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 计算结果:")
    print(f"   gini_coefficient = {gini:.3f}")
    print(f"   平均距离 = {np.mean(distances):.1f} 米")
    print(f"   距离标准差 = {np.std(distances):.1f} 米")
    print(f"   最大距离 = {np.max(distances):.1f} 米")
    
    # 演示不同场景
    print(f"\n🔄 不同不平等程度的场景...")
    demo_different_inequality_scenarios()
    
    print(f"\n📚 基尼系数的计算原理:")
    print(f"   1. 将所有道路点按距离从小到大排序")
    print(f"   2. 计算累积人口比例和累积距离比例")
    print(f"   3. 绘制洛伦兹曲线（实际分布）")
    print(f"   4. 计算洛伦兹曲线与完全平等线之间的面积")
    print(f"   5. 基尼系数 = 不平等面积 / 完全平等线下的总面积")
    
    print(f"\n🎯 基尼系数的含义:")
    print(f"   • 0.0 = 完全平等（所有道路点到充电桩距离相同）")
    print(f"   • 0.0-0.2 = 低不平等（服务分布很均匀）")
    print(f"   • 0.2-0.4 = 中等不平等（服务分布较均匀）")
    print(f"   • 0.4-0.6 = 高不平等（服务分布不均）")
    print(f"   • 0.6-1.0 = 极高不平等（服务严重不均）")
    print(f"   • 1.0 = 完全不平等（理论极值）")
    
    print(f"\n💡 在充电桩规划中的应用:")
    print(f"   - 评估服务公平性和均等化程度")
    print(f"   - 识别服务分配的不公平现象")
    print(f"   - 指导充电桩布局以改善服务均等性")
    print(f"   - 平衡效率与公平的权衡")
