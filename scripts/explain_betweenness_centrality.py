#!/usr/bin/env python3
"""
演示 avg_betweenness_centrality (平均介数中心性) 的计算过程
衡量充电桩在道路网络中的战略重要性和交通枢纽地位
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

def create_sample_road_network():
    """创建一个示例道路网络用于演示"""
    G = nx.Graph()
    
    # 创建网格状路网结构
    rows, cols = 6, 6
    node_positions = {}
    
    # 添加节点
    for i in range(rows):
        for j in range(cols):
            node_id = f"n_{i}_{j}"
            x = j * 1000  # 1000米间距
            y = i * 1000
            G.add_node(node_id, pos=(x, y))
            node_positions[node_id] = (x, y)
    
    # 添加水平边
    for i in range(rows):
        for j in range(cols - 1):
            node1 = f"n_{i}_{j}"
            node2 = f"n_{i}_{j+1}"
            length = 1000  # 边长1000米
            G.add_edge(node1, node2, weight=length)
    
    # 添加垂直边
    for i in range(rows - 1):
        for j in range(cols):
            node1 = f"n_{i}_{j}"
            node2 = f"n_{i+1}_{j}"
            length = 1000
            G.add_edge(node1, node2, weight=length)
    
    # 添加一些对角线连接以增加复杂性
    diagonal_connections = [
        ("n_1_1", "n_2_2"), ("n_2_2", "n_3_3"), ("n_3_3", "n_4_4"),
        ("n_1_4", "n_2_3"), ("n_2_3", "n_3_2"), ("n_3_2", "n_4_1"),
        ("n_0_2", "n_1_3"), ("n_4_2", "n_5_3")
    ]
    
    for node1, node2 in diagonal_connections:
        if node1 in G.nodes() and node2 in G.nodes():
            pos1 = node_positions[node1]
            pos2 = node_positions[node2]
            length = np.linalg.norm(np.array(pos1) - np.array(pos2))
            G.add_edge(node1, node2, weight=length)
    
    return G, node_positions

def find_closest_network_node(G, target_coord, node_positions):
    """找到距离目标坐标最近的网络节点"""
    min_distance = float('inf')
    closest_node = None
    
    target_pos = np.array(target_coord)
    
    for node_id, pos in node_positions.items():
        node_pos = np.array(pos)
        distance = np.linalg.norm(target_pos - node_pos)
        
        if distance < min_distance:
            min_distance = distance
            closest_node = node_id
    
    return closest_node, min_distance

def calculate_betweenness_centrality_demo(G, node_positions, cs_coords, title="Betweenness Centrality Analysis"):
    """演示介数中心性的计算过程"""
    
    # 设置字体
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 原始路网结构
    pos_array = np.array(list(node_positions.values()))
    
    # 绘制节点
    ax1.scatter(pos_array[:, 0], pos_array[:, 1], c='lightblue', s=80, alpha=0.8, 
                edgecolors='black', linewidth=1, label='Network Nodes')
    
    # 绘制边
    for edge in G.edges():
        node1, node2 = edge
        pos1 = node_positions[node1]
        pos2 = node_positions[node2]
        ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', linewidth=1, alpha=0.6)
    
    # 绘制充电桩
    ax1.scatter(cs_coords[:, 0], cs_coords[:, 1], c='red', s=200, alpha=0.9, 
                edgecolors='black', linewidth=2, marker='s', label='Charging Stations')
    
    # 添加充电桩编号
    for i, (x, y) in enumerate(cs_coords):
        ax1.annotate(f'CS{i+1}', (x, y), xytext=(0, -20), textcoords='offset points',
                    fontsize=9, color='white', weight='bold', ha='center')
    
    ax1.set_title('Step 1: Road Network Structure')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 计算介数中心性
    print("🔍 计算网络中所有节点的介数中心性...")
    
    # 对于演示，使用较小的采样以加快计算
    if len(G.nodes()) > 20:
        betweenness = nx.betweenness_centrality(G, k=min(20, len(G.nodes())), normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    
    print(f"✅ 中心性计算完成，共 {len(betweenness)} 个节点")
    
    # 创建中心性热图
    centrality_values = [betweenness.get(node, 0.0) for node in G.nodes()]
    max_centrality = max(centrality_values) if centrality_values else 1.0
    
    # 绘制节点（按中心性着色）
    scatter = ax2.scatter(pos_array[:, 0], pos_array[:, 1], 
                         c=centrality_values, s=120, alpha=0.8, 
                         cmap='YlOrRd', vmin=0, vmax=max_centrality,
                         edgecolors='black', linewidth=1)
    
    # 绘制边
    for edge in G.edges():
        node1, node2 = edge
        pos1 = node_positions[node1]
        pos2 = node_positions[node2]
        ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', linewidth=0.5, alpha=0.4)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Betweenness Centrality')
    
    # 标注高中心性节点
    sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    top_nodes = sorted_nodes[:5]  # 显示前5个高中心性节点
    
    for node, centrality_val in top_nodes:
        if centrality_val > 0:
            pos = node_positions[node]
            ax2.annotate(f'{centrality_val:.3f}', pos, xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='black', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.set_title('Step 2: Network Betweenness Centrality')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 充电桩的中心性分析
    # 找到每个充电桩最近的路网节点
    cs_nodes = []
    cs_distances = []
    cs_centralities = []
    
    for i, cs_coord in enumerate(cs_coords):
        closest_node, distance = find_closest_network_node(G, cs_coord, node_positions)
        centrality_val = betweenness.get(closest_node, 0.0)
        
        cs_nodes.append(closest_node)
        cs_distances.append(distance)
        cs_centralities.append(centrality_val)
        
        print(f"CS{i+1}: 最近节点 {closest_node}, 距离 {distance:.0f}m, 中心性 {centrality_val:.3f}")
    
    # 绘制充电桩与对应网络节点的关系
    ax3.scatter(pos_array[:, 0], pos_array[:, 1], 
               c=centrality_values, s=80, alpha=0.6, 
               cmap='YlOrRd', vmin=0, vmax=max_centrality,
               edgecolors='gray', linewidth=0.5)
    
    # 绘制边（淡化）
    for edge in G.edges():
        node1, node2 = edge
        pos1 = node_positions[node1]
        pos2 = node_positions[node2]
        ax3.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', linewidth=0.3, alpha=0.3)
    
    # 高亮充电桩对应的网络节点
    for i, (cs_coord, cs_node, centrality_val) in enumerate(zip(cs_coords, cs_nodes, cs_centralities)):
        # 充电桩位置
        ax3.scatter(cs_coord[0], cs_coord[1], c='red', s=200, alpha=0.9, 
                   edgecolors='black', linewidth=2, marker='s')
        
        # 对应的网络节点
        node_pos = node_positions[cs_node]
        ax3.scatter(node_pos[0], node_pos[1], c='blue', s=150, alpha=0.9, 
                   edgecolors='white', linewidth=2, marker='*')
        
        # 连接线
        ax3.plot([cs_coord[0], node_pos[0]], [cs_coord[1], node_pos[1]], 
                'b--', linewidth=2, alpha=0.8)
        
        # 标注
        ax3.annotate(f'CS{i+1}\n{centrality_val:.3f}', cs_coord, 
                    xytext=(0, -25), textcoords='offset points',
                    fontsize=9, color='white', weight='bold', ha='center')
    
    ax3.set_title('Step 3: CS Network Node Mapping')
    ax3.set_xlabel('X Coordinate (m)')
    ax3.set_ylabel('Y Coordinate (m)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 统计分析
    avg_centrality = np.mean(cs_centralities) if cs_centralities else 0.0
    
    # 创建中心性分布图
    x_positions = range(len(cs_coords))
    bars = ax4.bar(x_positions, cs_centralities, alpha=0.7, 
                   color=['red' if c == max(cs_centralities) else 'lightblue' for c in cs_centralities],
                   edgecolor='black', linewidth=1)
    
    # 添加平均线
    ax4.axhline(avg_centrality, color='green', linestyle='--', linewidth=2, 
                label=f'Average: {avg_centrality:.3f}')
    
    # 标注数值
    for i, (bar, centrality_val) in enumerate(zip(bars, cs_centralities)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{centrality_val:.3f}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax4.set_title('Step 4: CS Centrality Distribution')
    ax4.set_xlabel('Charging Station')
    ax4.set_ylabel('Betweenness Centrality')
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels([f'CS{i+1}' for i in range(len(cs_coords))])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加详细统计信息
    stats_text = f'''Centrality Analysis:
Average Centrality: {avg_centrality:.3f}
Max Centrality: {max(cs_centralities):.3f}
Min Centrality: {min(cs_centralities):.3f}
Std Deviation: {np.std(cs_centralities):.3f}

Network Statistics:
Total Nodes: {len(G.nodes())}
Total Edges: {len(G.edges())}
Network Density: {nx.density(G):.3f}

CS Mapping Quality:
Avg Distance to Node: {np.mean(cs_distances):.1f}m
Max Distance to Node: {max(cs_distances):.1f}m'''
    
    # 在图外添加统计信息
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom')
    
    plt.suptitle(f'{title}\nAverage Betweenness Centrality = {avg_centrality:.3f}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    return avg_centrality, cs_centralities, betweenness

def demo_centrality_importance():
    """演示不同位置充电桩的中心性重要性"""
    
    G, node_positions = create_sample_road_network()
    
    # 场景1：中心位置（高中心性）
    scenario1_cs = np.array([
        [2500, 2500], [2500, 1500], [1500, 2500]  # 网络中心区域
    ])
    
    # 场景2：边缘位置（低中心性）
    scenario2_cs = np.array([
        [0, 0], [5000, 0], [0, 5000]  # 网络边缘
    ])
    
    # 场景3：混合位置（中等中心性）
    scenario3_cs = np.array([
        [2500, 2500], [0, 0], [5000, 5000]  # 中心+边缘混合
    ])
    
    scenarios = [
        (scenario1_cs, "Central Locations"),
        (scenario2_cs, "Edge Locations"),
        (scenario3_cs, "Mixed Locations")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 计算所有节点的中心性（用于背景）
    betweenness = nx.betweenness_centrality(G, normalized=True)
    pos_array = np.array(list(node_positions.values()))
    centrality_values = [betweenness.get(node, 0.0) for node in G.nodes()]
    
    results = []
    
    for i, (cs_coords, title) in enumerate(scenarios):
        ax = axes[i]
        
        # 绘制网络背景
        scatter = ax.scatter(pos_array[:, 0], pos_array[:, 1], 
                           c=centrality_values, s=60, alpha=0.6, 
                           cmap='YlOrRd', edgecolors='gray', linewidth=0.5)
        
        # 绘制边
        for edge in G.edges():
            node1, node2 = edge
            pos1 = node_positions[node1]
            pos2 = node_positions[node2]
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', linewidth=0.3, alpha=0.4)
        
        # 计算充电桩中心性
        cs_centralities = []
        for cs_coord in cs_coords:
            closest_node, _ = find_closest_network_node(G, cs_coord, node_positions)
            centrality_val = betweenness.get(closest_node, 0.0)
            cs_centralities.append(centrality_val)
        
        avg_centrality = np.mean(cs_centralities)
        
        # 绘制充电桩
        ax.scatter(cs_coords[:, 0], cs_coords[:, 1], c='red', s=150, alpha=0.9, 
                  edgecolors='black', linewidth=2, marker='s')
        
        # 添加中心性标注
        for j, (cs_coord, centrality_val) in enumerate(zip(cs_coords, cs_centralities)):
            ax.annotate(f'{centrality_val:.3f}', cs_coord, 
                       xytext=(0, -20), textcoords='offset points',
                       fontsize=8, color='white', weight='bold', ha='center')
        
        ax.set_title(f'{title}\nAvg Centrality = {avg_centrality:.3f}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)' if i == 0 else '')
        ax.grid(True, alpha=0.3)
        
        if i == 2:  # 只在最右边添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Centrality')
        
        results.append((title, avg_centrality, cs_centralities))
    
    plt.suptitle('Betweenness Centrality Comparison for Different CS Locations', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/centrality_scenarios.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 介数中心性比较:")
    for title, avg_cent, centralities in results:
        print(f"   {title}:")
        print(f"     平均中心性: {avg_cent:.3f}")
        print(f"     各桩中心性: {[f'{c:.3f}' for c in centralities]}")
        print(f"     标准差: {np.std(centralities):.3f}")
        print()

if __name__ == "__main__":
    print("🎯 演示平均介数中心性 (avg_betweenness_centrality) 的计算过程")
    print("📚 衡量充电桩在道路网络中的战略重要性和交通枢纽地位\n")
    
    # 创建示例道路网络
    G, node_positions = create_sample_road_network()
    
    # 示例充电桩布局
    example_cs = np.array([
        [1500, 1500], [3500, 2500], [2500, 4000]
    ])
    
    print(f"📊 演示数据:")
    print(f"   路网节点: {len(G.nodes())} 个")
    print(f"   路网边: {len(G.edges())} 条")
    print(f"   充电桩数量: {len(example_cs)} 个")
    
    # 详细演示计算过程
    print(f"\n📊 详细计算过程演示...")
    plt.figure(figsize=(16, 12))
    
    avg_centrality, cs_centralities, all_centrality = calculate_betweenness_centrality_demo(
        G, node_positions, example_cs,
        title="Betweenness Centrality Analysis for Charging Station Layout"
    )
    
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/centrality_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 计算结果:")
    print(f"   avg_betweenness_centrality = {avg_centrality:.3f}")
    print(f"   各充电桩中心性: {[f'{c:.3f}' for c in cs_centralities]}")
    print(f"   中心性标准差: {np.std(cs_centralities):.3f}")
    
    # 演示不同场景
    print(f"\n🔄 不同位置的中心性重要性...")
    demo_centrality_importance()
    
    print(f"\n📚 介数中心性的计算原理:")
    print(f"   1. 构建道路网络图（节点=路口，边=道路段）")
    print(f"   2. 计算所有节点对之间的最短路径")
    print(f"   3. 对每个节点，统计有多少最短路径经过它")
    print(f"   4. 介数中心性 = 经过该节点的最短路径数 / 总最短路径数")
    print(f"   5. 为每个充电桩找到最近的路网节点")
    print(f"   6. 平均介数中心性 = 所有充电桩对应节点中心性的平均值")
    
    print(f"\n🎯 介数中心性的含义:")
    print(f"   • 高值 (>0.1) = 充电桩位于交通要道，是重要的交通枢纽")
    print(f"   • 中等值 (0.05-0.1) = 充电桩位于重要交通节点")
    print(f"   • 低值 (<0.05) = 充电桩位于边缘或次要道路")
    print(f"   • 反映充电桩在交通网络中的战略重要性")
    
    print(f"\n💡 在充电桩规划中的应用:")
    print(f"   - 评估充电桩位置的交通战略价值")
    print(f"   - 识别关键交通节点以优先布设充电桩")
    print(f"   - 平衡交通便利性与覆盖广度")
    print(f"   - 分析充电桩对交通流量的潜在影响")
    print(f"   - 优化充电桩布局以最大化交通网络效益")
