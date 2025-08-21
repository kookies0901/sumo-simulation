#!/usr/bin/env python3
"""
演示 cluster_count (聚类数量) 的计算过程
使用DBSCAN算法识别充电桩的空间聚类
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
import seaborn as sns

def calculate_cluster_count_demo(coords, eps=500, min_samples=2, title="Cluster Count Analysis"):
    """演示聚类数量的计算过程"""
    
    # 设置字体
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 原始充电桩分布
    ax1.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, alpha=0.8, 
                edgecolors='black', linewidth=1, marker='o')
    
    # 添加点的编号
    for i, (x, y) in enumerate(coords):
        ax1.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='white', weight='bold')
    
    ax1.set_title('Step 1: Original Charging Station Locations')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.grid(True, alpha=0.3)
    
    # 添加参数说明
    param_text = f'DBSCAN Parameters:\neps = {eps} meters\nmin_samples = {min_samples}'
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    # 2. 距离矩阵可视化
    n_points = len(coords)
    distance_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    # 创建距离矩阵热图
    im = ax2.imshow(distance_matrix, cmap='RdYlBu_r', alpha=0.8)
    ax2.set_title('Step 2: Distance Matrix (meters)')
    ax2.set_xlabel('Station Index')
    ax2.set_ylabel('Station Index')
    
    # 添加距离值
    for i in range(n_points):
        for j in range(n_points):
            if i != j and distance_matrix[i, j] <= eps:
                ax2.text(j, i, f'{distance_matrix[i, j]:.0f}', 
                        ha='center', va='center', fontsize=8, 
                        color='white', weight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Distance (m)')
    
    # 高亮eps阈值
    ax2.axhline(y=-0.5, color='red', linewidth=3, alpha=0.7)
    ax2.text(n_points/2, -1, f'eps = {eps}m threshold', ha='center', 
             color='red', weight='bold', fontsize=10)
    
    # 3. DBSCAN聚类结果
    if len(coords) > 1:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coords)
        
        # 计算聚类数量
        unique_labels = set(cluster_labels)
        cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # 如果没有形成聚类，则每个点都是独立的聚类
        if cluster_count == 0:
            cluster_count = len(coords)
            cluster_labels = np.arange(len(coords))  # 每个点一个聚类
        
        # 创建颜色映射
        n_clusters = len(unique_labels)
        if -1 in unique_labels:  # 有噪声点
            colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
            colors = ['red'] + colors[1:].tolist()  # 红色为噪声点
        else:
            colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
        
        # 绘制聚类结果
        for i, label in enumerate(unique_labels):
            if label == -1:
                # 噪声点
                mask = cluster_labels == label
                ax3.scatter(coords[mask, 0], coords[mask, 1], 
                           c='red', s=100, alpha=0.8, marker='x', 
                           linewidth=3, label='Noise')
            else:
                # 聚类点
                mask = cluster_labels == label
                ax3.scatter(coords[mask, 0], coords[mask, 1], 
                           c=[colors[label % len(colors)]], s=100, alpha=0.8, 
                           edgecolors='black', linewidth=1,
                           label=f'Cluster {label + 1}')
                
                # 绘制聚类中心
                cluster_coords = coords[mask]
                center = cluster_coords.mean(axis=0)
                ax3.scatter(center[0], center[1], c='black', s=200, 
                           marker='*', edgecolors='white', linewidth=2)
        
        # 添加点的编号和聚类标签
        for i, (x, y) in enumerate(coords):
            label_text = f'{i+1}'
            if cluster_labels[i] == -1:
                label_text += '\n(Noise)'
            else:
                label_text += f'\n(C{cluster_labels[i] + 1})'
            
            ax3.annotate(label_text, (x, y), xytext=(8, 8), textcoords='offset points',
                        fontsize=7, ha='left', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax3.set_title(f'Step 3: DBSCAN Clustering Result\nCluster Count = {cluster_count}')
        ax3.set_xlabel('X Coordinate (m)')
        ax3.set_ylabel('Y Coordinate (m)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. 聚类统计分析
        cluster_stats = {}
        if cluster_count > 0:
            for label in unique_labels:
                if label != -1:  # 排除噪声点
                    mask = cluster_labels == label
                    cluster_coords = coords[mask]
                    cluster_size = len(cluster_coords)
                    
                    if cluster_size > 1:
                        # 计算聚类内距离
                        intra_distances = []
                        for i in range(cluster_size):
                            for j in range(i+1, cluster_size):
                                dist = np.linalg.norm(cluster_coords[i] - cluster_coords[j])
                                intra_distances.append(dist)
                        
                        cluster_stats[f'Cluster {label + 1}'] = {
                            'size': cluster_size,
                            'avg_intra_dist': np.mean(intra_distances),
                            'max_intra_dist': np.max(intra_distances),
                            'center': cluster_coords.mean(axis=0)
                        }
                    else:
                        cluster_stats[f'Cluster {label + 1}'] = {
                            'size': cluster_size,
                            'avg_intra_dist': 0,
                            'max_intra_dist': 0,
                            'center': cluster_coords[0]
                        }
        
        # 噪声点统计
        noise_count = np.sum(cluster_labels == -1)
        
        # 创建统计表格
        stats_text = f"Clustering Statistics:\n"
        stats_text += f"Total Stations: {len(coords)}\n"
        stats_text += f"Clusters Found: {cluster_count}\n"
        stats_text += f"Noise Points: {noise_count}\n"
        stats_text += f"Clustered Points: {len(coords) - noise_count}\n\n"
        
        if cluster_stats:
            stats_text += "Cluster Details:\n"
            for cluster_name, stats in cluster_stats.items():
                stats_text += f"{cluster_name}:\n"
                stats_text += f"  Size: {stats['size']} stations\n"
                stats_text += f"  Avg Intra-distance: {stats['avg_intra_dist']:.1f}m\n"
                stats_text += f"  Max Intra-distance: {stats['max_intra_dist']:.1f}m\n\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                 verticalalignment='top', fontsize=9, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax4.set_title('Step 4: Clustering Statistics')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
    else:
        cluster_count = 1 if len(coords) == 1 else 0
        ax3.text(0.5, 0.5, f'Insufficient data for clustering\nCluster Count = {cluster_count}', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax4.text(0.5, 0.5, 'No clustering performed', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    
    plt.suptitle(f'{title}\nFinal Cluster Count = {cluster_count}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return cluster_count, cluster_labels if len(coords) > 1 else np.array([0])

def demo_different_cluster_scenarios():
    """演示不同聚类场景"""
    
    # 场景1：明显的聚类
    scenario1 = np.vstack([
        np.random.normal([1000, 1000], 100, (5, 2)),  # 聚类1
        np.random.normal([3000, 3000], 150, (4, 2)),  # 聚类2
        np.array([[5000, 1000]])  # 孤立点
    ])
    
    # 场景2：紧密分布（一个大聚类）
    scenario2 = np.random.normal([2000, 2000], 200, (10, 2))
    
    # 场景3：分散分布（多个小聚类或噪声点）
    scenario3 = np.array([
        [1000, 1000], [1200, 1100],  # 可能的聚类
        [3000, 3000], [3000, 4000],  # 边界情况
        [5000, 1000], [7000, 2000], [8000, 5000]  # 分散点
    ])
    
    scenarios = [
        (scenario1, "Scenario 1: Clear Clusters"),
        (scenario2, "Scenario 2: Dense Distribution"),
        (scenario3, "Scenario 3: Sparse Distribution")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (coords, title) in enumerate(scenarios):
        ax = axes[i]
        
        # 计算聚类
        eps = 500
        min_samples = 2
        
        if len(coords) > 1:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(coords)
            
            unique_labels = set(labels)
            cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if cluster_count == 0:
                cluster_count = len(coords)
                labels = np.arange(len(coords))
            
            # 绘制结果
            colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 3)))
            
            for label in unique_labels:
                if label == -1:
                    mask = labels == label
                    ax.scatter(coords[mask, 0], coords[mask, 1], 
                             c='red', s=80, alpha=0.8, marker='x', linewidth=2)
                else:
                    mask = labels == label
                    ax.scatter(coords[mask, 0], coords[mask, 1], 
                             c=[colors[label % len(colors)]], s=80, alpha=0.8, 
                             edgecolors='black', linewidth=1)
        else:
            cluster_count = 1 if len(coords) == 1 else 0
            ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=80, alpha=0.8)
        
        ax.set_title(f'{title}\nCluster Count = {cluster_count}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        
        # 添加编号
        for j, (x, y) in enumerate(coords):
            ax.annotate(f'{j+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white', weight='bold')
    
    plt.suptitle('Cluster Count for Different Distribution Patterns\n(eps=500m, min_samples=2)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/cluster_count_scenarios.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🎯 演示聚类数量 (cluster_count) 的计算过程")
    print("📚 使用DBSCAN算法识别充电桩的空间聚类模式\n")
    
    # 创建示例数据：包含2个明显聚类 + 1个噪声点
    np.random.seed(42)
    cluster1 = np.random.normal([2000, 3000], 150, (4, 2))
    cluster2 = np.random.normal([5000, 2000], 200, (3, 2))
    noise_point = np.array([[7000, 5000]])
    
    example_coords = np.vstack([cluster1, cluster2, noise_point])
    
    # 详细演示计算过程
    print("📊 详细计算过程演示...")
    plt.figure(figsize=(16, 12))
    
    cluster_count, labels = calculate_cluster_count_demo(
        example_coords, 
        eps=500, 
        min_samples=2,
        title="DBSCAN Clustering Analysis for Charging Stations"
    )
    
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/cluster_count_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 计算结果:")
    print(f"   cluster_count = {cluster_count}")
    print(f"   聚类标签 = {labels}")
    
    # 演示不同场景
    print(f"\n🔄 不同场景下的聚类结果...")
    demo_different_cluster_scenarios()
    
    print(f"\n📚 DBSCAN算法参数说明:")
    print(f"   eps = 500米    # 邻域半径，两点距离小于500米才可能在同一聚类")
    print(f"   min_samples = 2 # 形成聚类的最小点数，至少2个点才能形成聚类")
    
    print(f"\n🎯 cluster_count的意义:")
    print(f"   • 高值 = 充电桩形成多个分散的聚类（空间分布碎片化）")
    print(f"   • 低值 = 充电桩形成少数几个大聚类（空间分布集中）")
    print(f"   • 1 = 所有充电桩形成一个紧密聚类")
    print(f"   • 等于桩数 = 所有充电桩都是孤立的（分布过于分散）")
    
    print(f"\n💡 在布局分析中的应用:")
    print(f"   - 评估布局的聚集程度")
    print(f"   - 识别服务热点区域")
    print(f"   - 分析空间分布的合理性")
    print(f"   - 与等待时间、服务效率等性能指标建立关系")
