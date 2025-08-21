#!/usr/bin/env python3
"""
演示 coverage_ratio (覆盖率) 的计算过程
衡量充电桩布局对道路网络的服务覆盖程度
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import xml.etree.ElementTree as ET

def create_sample_road_network():
    """创建一个简化的道路网络示例用于演示"""
    # 创建网格状道路网络
    road_points = []
    
    # 水平道路
    for y in [1000, 2000, 3000, 4000, 5000]:
        for x in range(500, 5500, 100):
            road_points.append((x, y))
    
    # 垂直道路
    for x in [1000, 2000, 3000, 4000, 5000]:
        for y in range(500, 5500, 100):
            road_points.append((x, y))
    
    return np.array(road_points)

def calculate_coverage_ratio_demo(cs_coords, road_coords, coverage_distance=500, title="Coverage Ratio Analysis"):
    """演示覆盖率的计算过程"""
    
    # 设置字体
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 原始道路网络和充电桩分布
    ax1.scatter(road_coords[:, 0], road_coords[:, 1], c='lightgray', s=20, alpha=0.6, 
                label='Road Points')
    ax1.scatter(cs_coords[:, 0], cs_coords[:, 1], c='red', s=150, alpha=0.8, 
                edgecolors='black', linewidth=2, marker='s', label='Charging Stations')
    
    # 添加充电桩编号
    for i, (x, y) in enumerate(cs_coords):
        ax1.annotate(f'CS{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='white', weight='bold')
    
    ax1.set_title('Step 1: Road Network & Charging Stations')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加参数说明
    param_text = f'Coverage Distance: {coverage_distance}m\nRoad Sample Points: {len(road_coords)}\nCharging Stations: {len(cs_coords)}'
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    # 2. 覆盖圆和距离计算
    ax2.scatter(road_coords[:, 0], road_coords[:, 1], c='lightgray', s=20, alpha=0.6)
    ax2.scatter(cs_coords[:, 0], cs_coords[:, 1], c='red', s=150, alpha=0.8, 
                edgecolors='black', linewidth=2, marker='s')
    
    # 绘制覆盖圆
    for i, (cs_x, cs_y) in enumerate(cs_coords):
        circle = patches.Circle((cs_x, cs_y), coverage_distance, 
                               linewidth=2, edgecolor='red', facecolor='red', alpha=0.1)
        ax2.add_patch(circle)
        ax2.annotate(f'CS{i+1}', (cs_x, cs_y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='white', weight='bold')
    
    ax2.set_title(f'Step 2: Coverage Circles ({coverage_distance}m radius)')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    
    # 3. 距离热图和覆盖分析
    # 计算每个道路点到最近充电桩的距离
    distances_to_cs = []
    closest_cs_indices = []
    
    for road_point in road_coords:
        dists = [np.linalg.norm(road_point - cs_coord) for cs_coord in cs_coords]
        min_dist = min(dists)
        closest_cs_idx = np.argmin(dists)
        distances_to_cs.append(min_dist)
        closest_cs_indices.append(closest_cs_idx)
    
    distances_to_cs = np.array(distances_to_cs)
    
    # 创建距离热图
    covered_mask = distances_to_cs <= coverage_distance
    uncovered_mask = ~covered_mask
    
    # 绘制覆盖状态
    if np.any(covered_mask):
        scatter_covered = ax3.scatter(road_coords[covered_mask, 0], road_coords[covered_mask, 1], 
                                     c=distances_to_cs[covered_mask], s=40, alpha=0.8, 
                                     cmap='RdYlGn_r', vmax=coverage_distance,
                                     edgecolors='black', linewidth=0.5,
                                     label=f'Covered ({np.sum(covered_mask)} points)')
    
    if np.any(uncovered_mask):
        ax3.scatter(road_coords[uncovered_mask, 0], road_coords[uncovered_mask, 1], 
                   c='darkred', s=40, alpha=0.8, marker='x', linewidth=2,
                   label=f'Not Covered ({np.sum(uncovered_mask)} points)')
    
    # 充电桩位置
    ax3.scatter(cs_coords[:, 0], cs_coords[:, 1], c='blue', s=150, alpha=0.9, 
                edgecolors='white', linewidth=2, marker='*', label='Charging Stations')
    
    ax3.set_title('Step 3: Coverage Analysis')
    ax3.set_xlabel('X Coordinate (m)')
    ax3.set_ylabel('Y Coordinate (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加颜色条
    if np.any(covered_mask):
        cbar = plt.colorbar(scatter_covered, ax=ax3, shrink=0.8)
        cbar.set_label('Distance to Nearest CS (m)')
    
    # 4. 统计分析和结果
    coverage_ratio = np.mean(covered_mask)
    max_gap_distance = np.max(distances_to_cs)
    avg_distance = np.mean(distances_to_cs)
    
    # 按距离分段统计
    distance_bins = [0, 200, 400, 500, 1000, np.inf]
    distance_labels = ['0-200m', '200-400m', '400-500m', '500-1000m', '>1000m']
    distance_counts = []
    
    for i in range(len(distance_bins)-1):
        mask = (distances_to_cs >= distance_bins[i]) & (distances_to_cs < distance_bins[i+1])
        distance_counts.append(np.sum(mask))
    
    # 绘制距离分布直方图
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    bars = ax4.bar(range(len(distance_counts)), distance_counts, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # 高亮覆盖阈值
    ax4.axvline(x=2.5, color='red', linestyle='--', linewidth=3, alpha=0.8, 
                label=f'Coverage Threshold ({coverage_distance}m)')
    
    ax4.set_title('Step 4: Distance Distribution')
    ax4.set_xlabel('Distance Range')
    ax4.set_ylabel('Number of Road Points')
    ax4.set_xticks(range(len(distance_labels)))
    ax4.set_xticklabels(distance_labels, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for i, (bar, count) in enumerate(zip(bars, distance_counts)):
        if count > 0:
            height = bar.get_height()
            percentage = count / len(distances_to_cs) * 100
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=8, weight='bold')
    
    # 添加统计信息
    stats_text = f'''Coverage Statistics:
Total Road Points: {len(road_coords)}
Covered Points: {np.sum(covered_mask)} ({coverage_ratio:.3f})
Coverage Ratio: {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)

Distance Statistics:
Average Distance: {avg_distance:.1f}m
Maximum Distance: {max_gap_distance:.1f}m
Median Distance: {np.median(distances_to_cs):.1f}m

Coverage Quality:
Excellent (<200m): {distance_counts[0]} ({distance_counts[0]/len(distances_to_cs)*100:.1f}%)
Good (200-400m): {distance_counts[1]} ({distance_counts[1]/len(distances_to_cs)*100:.1f}%)
Acceptable (400-500m): {distance_counts[2]} ({distance_counts[2]/len(distances_to_cs)*100:.1f}%)'''
    
    # 在图外添加统计信息
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom')
    
    plt.suptitle(f'{title}\nCoverage Ratio = {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # 为底部统计信息留空间
    
    return coverage_ratio, max_gap_distance, distances_to_cs

def demo_different_coverage_scenarios():
    """演示不同覆盖场景"""
    
    # 创建标准道路网络
    road_network = create_sample_road_network()
    
    # 场景1：良好覆盖
    scenario1_cs = np.array([
        [1500, 1500], [2500, 2500], [3500, 3500], [4500, 4500],
        [1500, 3500], [3500, 1500]
    ])
    
    # 场景2：覆盖不足
    scenario2_cs = np.array([
        [1000, 1000], [5000, 5000]
    ])
    
    # 场景3：过度集中
    scenario3_cs = np.array([
        [2800, 2800], [3000, 3000], [3200, 3200], [3000, 2800], [3000, 3200]
    ])
    
    scenarios = [
        (scenario1_cs, "Good Coverage"),
        (scenario2_cs, "Poor Coverage"), 
        (scenario3_cs, "Clustered Coverage")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    results = []
    
    for i, (cs_coords, title) in enumerate(scenarios):
        ax = axes[i]
        
        # 计算覆盖率
        distances = []
        for road_point in road_network:
            dists = [np.linalg.norm(road_point - cs_coord) for cs_coord in cs_coords]
            distances.append(min(dists))
        
        distances = np.array(distances)
        coverage_ratio = np.mean(distances <= 500)
        
        # 绘制结果
        covered_mask = distances <= 500
        uncovered_mask = ~covered_mask
        
        # 道路点
        if np.any(covered_mask):
            ax.scatter(road_network[covered_mask, 0], road_network[covered_mask, 1], 
                      c='green', s=20, alpha=0.6, label='Covered')
        if np.any(uncovered_mask):
            ax.scatter(road_network[uncovered_mask, 0], road_network[uncovered_mask, 1], 
                      c='red', s=20, alpha=0.6, label='Not Covered')
        
        # 充电桩和覆盖圆
        ax.scatter(cs_coords[:, 0], cs_coords[:, 1], c='blue', s=100, alpha=0.8, 
                  edgecolors='black', linewidth=1, marker='s')
        
        for cs_x, cs_y in cs_coords:
            circle = patches.Circle((cs_x, cs_y), 500, 
                                   linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.1)
            ax.add_patch(circle)
        
        ax.set_title(f'{title}\nCoverage = {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        results.append((title, coverage_ratio, len(cs_coords)))
    
    plt.suptitle('Coverage Ratio Comparison for Different Charging Station Layouts\n(Coverage Distance = 500m)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/coverage_scenarios.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 覆盖率比较:")
    for title, coverage, cs_count in results:
        print(f"   {title}: {coverage:.3f} ({coverage*100:.1f}%) - {cs_count} 个充电桩")

if __name__ == "__main__":
    print("🎯 演示覆盖率 (coverage_ratio) 的计算过程")
    print("📚 衡量充电桩布局对道路网络的服务覆盖程度\n")
    
    # 创建示例数据
    road_network = create_sample_road_network()
    
    # 示例充电桩布局
    example_cs = np.array([
        [2000, 2000], [4000, 3000], [1500, 4000], [3500, 1500]
    ])
    
    print(f"📊 演示数据:")
    print(f"   道路采样点: {len(road_network)} 个")
    print(f"   充电桩数量: {len(example_cs)} 个")
    print(f"   覆盖距离阈值: 500米")
    
    # 详细演示计算过程
    print(f"\n📊 详细计算过程演示...")
    plt.figure(figsize=(16, 12))
    
    coverage_ratio, max_gap, distances = calculate_coverage_ratio_demo(
        example_cs, 
        road_network,
        coverage_distance=500,
        title="Coverage Ratio Calculation for Charging Station Layout"
    )
    
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/coverage_ratio_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 计算结果:")
    print(f"   coverage_ratio = {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)")
    print(f"   max_gap_distance = {max_gap:.1f}米")
    print(f"   平均距离 = {np.mean(distances):.1f}米")
    
    # 演示不同场景
    print(f"\n🔄 不同场景下的覆盖率...")
    demo_different_coverage_scenarios()
    
    print(f"\n📚 coverage_ratio 的计算步骤:")
    print(f"   1. 从道路网络中采样代表性点位（默认10%采样率）")
    print(f"   2. 计算每个道路点到最近充电桩的距离")
    print(f"   3. 统计距离≤500米的道路点数量")
    print(f"   4. coverage_ratio = 覆盖的道路点数 / 总道路点数")
    
    print(f"\n🎯 coverage_ratio 的意义:")
    print(f"   • 1.0 = 完美覆盖，所有道路都在500米内有充电桩")
    print(f"   • 0.8+ = 良好覆盖，大部分道路有充电服务")
    print(f"   • 0.5-0.8 = 一般覆盖，有明显的服务盲区")
    print(f"   • <0.5 = 覆盖不足，存在大量服务空白")
    
    print(f"\n💡 在充电桩规划中的应用:")
    print(f"   - 评估充电服务的可达性")
    print(f"   - 识别服务盲区和薄弱环节")
    print(f"   - 优化充电桩布局以提高覆盖率")
    print(f"   - 平衡覆盖率与建设成本")
