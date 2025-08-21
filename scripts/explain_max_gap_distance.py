#!/usr/bin/env python3
"""
演示 max_gap_distance (最大间隙距离) 的计算过程
识别充电桩布局中的最大服务空白区域
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

def create_sample_road_network():
    """创建一个简化的道路网络示例用于演示"""
    # 创建更密集的网格状道路网络
    road_points = []
    
    # 水平道路
    for y in range(500, 5500, 200):
        for x in range(500, 5500, 100):
            road_points.append((x, y))
    
    # 垂直道路
    for x in range(500, 5500, 200):
        for y in range(500, 5500, 100):
            road_points.append((x, y))
    
    return np.array(road_points)

def calculate_max_gap_distance_demo(cs_coords, road_coords, title="Max Gap Distance Analysis"):
    """演示最大间隙距离的计算过程"""
    
    # 设置字体
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 原始布局
    ax1.scatter(road_coords[:, 0], road_coords[:, 1], c='lightblue', s=15, alpha=0.6, 
                label=f'Road Points ({len(road_coords)})')
    ax1.scatter(cs_coords[:, 0], cs_coords[:, 1], c='red', s=200, alpha=0.9, 
                edgecolors='black', linewidth=2, marker='s', label=f'Charging Stations ({len(cs_coords)})')
    
    # 添加充电桩编号
    for i, (x, y) in enumerate(cs_coords):
        ax1.annotate(f'CS{i+1}', (x, y), xytext=(0, -15), textcoords='offset points',
                    fontsize=9, color='white', weight='bold', ha='center')
    
    ax1.set_title('Step 1: Road Network & Charging Station Layout')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 距离计算和可视化
    # 计算每个道路点到最近充电桩的距离
    distances_to_cs = []
    closest_cs_indices = []
    
    print(f"🔍 计算 {len(road_coords)} 个道路点到最近充电桩的距离...")
    
    for i, road_point in enumerate(road_coords):
        dists = [np.linalg.norm(road_point - cs_coord) for cs_coord in cs_coords]
        min_dist = min(dists)
        closest_cs_idx = np.argmin(dists)
        distances_to_cs.append(min_dist)
        closest_cs_indices.append(closest_cs_idx)
    
    distances_to_cs = np.array(distances_to_cs)
    
    # 创建距离热图
    scatter = ax2.scatter(road_coords[:, 0], road_coords[:, 1], 
                         c=distances_to_cs, s=30, alpha=0.8, 
                         cmap='RdYlBu_r', edgecolors='black', linewidth=0.3)
    ax2.scatter(cs_coords[:, 0], cs_coords[:, 1], c='blue', s=200, alpha=0.9, 
                edgecolors='white', linewidth=3, marker='*', label='Charging Stations')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Distance to Nearest CS (m)')
    
    ax2.set_title('Step 2: Distance Heatmap to Nearest Charging Station')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 识别最大间隙点
    max_gap_distance = np.max(distances_to_cs)
    max_gap_indices = np.where(distances_to_cs == max_gap_distance)[0]
    
    print(f"📏 最大间隙距离: {max_gap_distance:.1f} 米")
    print(f"🎯 最大间隙点数量: {len(max_gap_indices)}")
    
    # 绘制最大间隙分析
    ax3.scatter(road_coords[:, 0], road_coords[:, 1], 
               c=distances_to_cs, s=25, alpha=0.7, 
               cmap='RdYlBu_r', edgecolors='gray', linewidth=0.2)
    
    # 高亮最大间隙点
    max_gap_points = road_coords[max_gap_indices]
    ax3.scatter(max_gap_points[:, 0], max_gap_points[:, 1], 
               c='darkred', s=150, alpha=1.0, 
               edgecolors='yellow', linewidth=3, marker='X', 
               label=f'Max Gap Points ({max_gap_distance:.0f}m)')
    
    # 充电桩
    ax3.scatter(cs_coords[:, 0], cs_coords[:, 1], c='blue', s=200, alpha=0.9, 
                edgecolors='white', linewidth=3, marker='*', label='Charging Stations')
    
    # 绘制到最近充电桩的连线（仅对最大间隙点）
    for gap_idx in max_gap_indices[:5]:  # 只显示前5个以免过于拥挤
        gap_point = road_coords[gap_idx]
        closest_cs_idx = closest_cs_indices[gap_idx]
        closest_cs = cs_coords[closest_cs_idx]
        
        ax3.plot([gap_point[0], closest_cs[0]], [gap_point[1], closest_cs[1]], 
                'r--', linewidth=2, alpha=0.8)
        
        # 添加距离标注
        mid_x = (gap_point[0] + closest_cs[0]) / 2
        mid_y = (gap_point[1] + closest_cs[1]) / 2
        ax3.annotate(f'{max_gap_distance:.0f}m', (mid_x, mid_y), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='red', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax3.set_title(f'Step 3: Max Gap Distance Identification')
    ax3.set_xlabel('X Coordinate (m)')
    ax3.set_ylabel('Y Coordinate (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 距离分布统计
    # 创建距离分布直方图
    bins = np.linspace(0, max_gap_distance * 1.1, 20)
    counts, bin_edges = np.histogram(distances_to_cs, bins=bins)
    
    ax4.hist(distances_to_cs, bins=bins, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=1, label='Distance Distribution')
    
    # 标记统计量
    mean_dist = np.mean(distances_to_cs)
    median_dist = np.median(distances_to_cs)
    p90_dist = np.percentile(distances_to_cs, 90)
    
    ax4.axvline(mean_dist, color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {mean_dist:.0f}m')
    ax4.axvline(median_dist, color='orange', linestyle='-', linewidth=2, 
                label=f'Median: {median_dist:.0f}m')
    ax4.axvline(p90_dist, color='purple', linestyle='-', linewidth=2, 
                label=f'90th %ile: {p90_dist:.0f}m')
    ax4.axvline(max_gap_distance, color='red', linestyle='-', linewidth=3, 
                label=f'Max Gap: {max_gap_distance:.0f}m')
    
    # 添加服务质量区间
    ax4.axvspan(0, 500, alpha=0.2, color='green', label='Good Service (<500m)')
    ax4.axvspan(500, 1000, alpha=0.2, color='yellow', label='Fair Service (500-1000m)')
    ax4.axvspan(1000, max_gap_distance * 1.1, alpha=0.2, color='red', label='Poor Service (>1000m)')
    
    ax4.set_title('Step 4: Distance Distribution Analysis')
    ax4.set_xlabel('Distance to Nearest CS (m)')
    ax4.set_ylabel('Number of Road Points')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 添加统计文本
    stats_text = f'''Max Gap Statistics:
Max Gap Distance: {max_gap_distance:.1f}m
Max Gap Points: {len(max_gap_indices)}
Worst Service Area: {len(max_gap_indices)/len(road_coords)*100:.1f}%

Distance Summary:
Mean: {mean_dist:.1f}m
Median: {median_dist:.1f}m
90th Percentile: {p90_dist:.1f}m
Standard Deviation: {np.std(distances_to_cs):.1f}m

Service Quality:
Good (<500m): {np.sum(distances_to_cs < 500)/len(distances_to_cs)*100:.1f}%
Fair (500-1000m): {np.sum((distances_to_cs >= 500) & (distances_to_cs < 1000))/len(distances_to_cs)*100:.1f}%
Poor (>1000m): {np.sum(distances_to_cs >= 1000)/len(distances_to_cs)*100:.1f}%'''
    
    # 在图外添加统计信息
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom')
    
    plt.suptitle(f'{title}\nMax Gap Distance = {max_gap_distance:.1f} meters', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # 为底部统计信息留空间
    
    return max_gap_distance, max_gap_points, distances_to_cs

def demo_different_gap_scenarios():
    """演示不同布局对最大间隙距离的影响"""
    
    # 创建标准道路网络
    road_network = create_sample_road_network()
    
    # 场景1：均匀分布（较小间隙）
    scenario1_cs = np.array([
        [1500, 1500], [2500, 1500], [3500, 1500], [4500, 1500],
        [1500, 2500], [2500, 2500], [3500, 2500], [4500, 2500],
        [1500, 3500], [2500, 3500], [3500, 3500], [4500, 3500],
        [1500, 4500], [2500, 4500], [3500, 4500], [4500, 4500]
    ])
    
    # 场景2：集中分布（较大间隙）
    scenario2_cs = np.array([
        [2000, 2000], [2200, 2200], [2400, 2000], [2000, 2400]
    ])
    
    # 场景3：边缘分布（中心有大间隙）
    scenario3_cs = np.array([
        [1000, 1000], [1000, 4000], [4000, 1000], [4000, 4000],
        [2500, 500], [2500, 4500], [500, 2500], [4500, 2500]
    ])
    
    scenarios = [
        (scenario1_cs, "Uniform Distribution"),
        (scenario2_cs, "Clustered Distribution"), 
        (scenario3_cs, "Edge Distribution")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    results = []
    
    for i, (cs_coords, title) in enumerate(scenarios):
        ax = axes[i]
        
        # 计算最大间隙距离
        distances = []
        for road_point in road_network:
            dists = [np.linalg.norm(road_point - cs_coord) for cs_coord in cs_coords]
            distances.append(min(dists))
        
        distances = np.array(distances)
        max_gap = np.max(distances)
        max_gap_indices = np.where(distances == max_gap)[0]
        
        # 绘制结果
        scatter = ax.scatter(road_network[:, 0], road_network[:, 1], 
                           c=distances, s=20, alpha=0.7, 
                           cmap='RdYlBu_r', edgecolors='gray', linewidth=0.2)
        
        # 高亮最大间隙点
        if len(max_gap_indices) > 0:
            max_gap_points = road_network[max_gap_indices]
            ax.scatter(max_gap_points[:, 0], max_gap_points[:, 1], 
                      c='darkred', s=80, alpha=1.0, 
                      edgecolors='yellow', linewidth=2, marker='X')
        
        # 充电桩
        ax.scatter(cs_coords[:, 0], cs_coords[:, 1], c='blue', s=100, alpha=0.9, 
                  edgecolors='white', linewidth=2, marker='*')
        
        ax.set_title(f'{title}\nMax Gap = {max_gap:.0f}m')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Distance (m)')
        
        results.append((title, max_gap, len(cs_coords), len(max_gap_indices)))
    
    plt.suptitle('Max Gap Distance Comparison for Different Charging Station Layouts', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/max_gap_scenarios.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 最大间隙距离比较:")
    for title, max_gap, cs_count, gap_points in results:
        print(f"   {title}:")
        print(f"     最大间隙: {max_gap:.0f}米")
        print(f"     充电桩数: {cs_count}个")
        print(f"     最差服务点: {gap_points}个")
        print()

if __name__ == "__main__":
    print("🎯 演示最大间隙距离 (max_gap_distance) 的计算过程")
    print("📚 识别充电桩布局中的最大服务空白区域\n")
    
    # 创建示例数据
    road_network = create_sample_road_network()
    
    # 示例充电桩布局（故意留出服务空白）
    example_cs = np.array([
        [1500, 1500], [4000, 4000], [1500, 4000]
    ])
    
    print(f"📊 演示数据:")
    print(f"   道路采样点: {len(road_network)} 个")
    print(f"   充电桩数量: {len(example_cs)} 个")
    
    # 详细演示计算过程
    print(f"\n📊 详细计算过程演示...")
    plt.figure(figsize=(16, 12))
    
    max_gap, gap_points, all_distances = calculate_max_gap_distance_demo(
        example_cs, 
        road_network,
        title="Max Gap Distance Analysis for Charging Station Layout"
    )
    
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/max_gap_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 计算结果:")
    print(f"   max_gap_distance = {max_gap:.1f} 米")
    print(f"   最差服务点数量 = {len(gap_points)} 个")
    print(f"   平均距离 = {np.mean(all_distances):.1f} 米")
    print(f"   距离标准差 = {np.std(all_distances):.1f} 米")
    
    # 演示不同场景
    print(f"\n🔄 不同布局方案的最大间隙距离...")
    demo_different_gap_scenarios()
    
    print(f"\n📚 max_gap_distance 的计算原理:")
    print(f"   1. 计算每个道路点到最近充电桩的距离")
    print(f"   2. 找出所有距离中的最大值")
    print(f"   3. max_gap_distance = max(所有最近距离)")
    
    print(f"\n🎯 max_gap_distance 的意义:")
    print(f"   • 小值 (<500m) = 服务覆盖良好，无明显盲区")
    print(f"   • 中等值 (500-1500m) = 存在服务薄弱区域")
    print(f"   • 大值 (>1500m) = 有严重的服务空白区域")
    print(f"   • 反映布局的'最坏情况'服务质量")
    
    print(f"\n💡 在充电桩规划中的应用:")
    print(f"   - 识别服务盲区的位置和严重程度")
    print(f"   - 评估布局方案的最坏情况服务质量")
    print(f"   - 指导新充电桩的优先建设位置")
    print(f"   - 平衡整体覆盖率与局部服务质量")
