#!/usr/bin/env python3
"""
演示 cs_density_std (充电桩密度标准差) 的计算过程
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

def calculate_density_std_demo(coords, title="Charging Station Density Analysis"):
    """演示密度标准差的计算过程"""
    
    # 设置中文字体
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 原始充电桩分布
    ax1.scatter(coords[:, 0], coords[:, 1], c='red', s=100, alpha=0.8, 
                edgecolors='black', linewidth=1, marker='s')
    ax1.set_title('Step 1: Original Charging Station Locations')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.grid(True, alpha=0.3)
    
    # 计算边界
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 添加边界框
    boundary = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
    ax1.add_patch(boundary)
    ax1.text(x_min, y_max + (y_max - y_min) * 0.05, 
             f'Boundary: {x_max - x_min:.0f}m × {y_max - y_min:.0f}m', 
             fontsize=10, color='blue')
    
    # 2. 网格划分
    grid_size = 10
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    
    # 绘制网格
    ax2.scatter(coords[:, 0], coords[:, 1], c='red', s=80, alpha=0.8, 
                edgecolors='black', linewidth=1, marker='s')
    
    # 绘制网格线
    for i in range(len(x_bins)):
        ax2.axvline(x_bins[i], color='gray', linewidth=1, alpha=0.7)
    for i in range(len(y_bins)):
        ax2.axhline(y_bins[i], color='gray', linewidth=1, alpha=0.7)
    
    ax2.set_title(f'Step 2: Grid Division ({grid_size}×{grid_size} cells)')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    
    # 计算每个网格的桩数
    grid_counts = np.zeros((grid_size, grid_size))
    cs_count = len(coords)
    
    for i in range(cs_count):
        x_idx = np.digitize(x_coords[i], x_bins) - 1
        y_idx = np.digitize(y_coords[i], y_bins) - 1
        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            grid_counts[x_idx, y_idx] += 1
    
    # 在网格中央显示桩数
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_counts[i, j] > 0:
                x_center = (x_bins[i] + x_bins[i+1]) / 2
                y_center = (y_bins[j] + y_bins[j+1]) / 2
                ax2.text(x_center, y_center, f'{int(grid_counts[i, j])}',
                        ha='center', va='center', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 3. 密度热图
    # 计算每个网格的面积（平方公里）
    grid_area_km2 = ((x_max - x_min) / grid_size) * ((y_max - y_min) / grid_size) / 1000000
    
    # 计算密度（桩数/平方公里）
    density_grid = grid_counts / grid_area_km2
    
    # 创建热图
    im = ax3.imshow(density_grid.T, origin='lower', 
                    extent=[x_min, x_max, y_min, y_max],
                    cmap='YlOrRd', alpha=0.8)
    
    # 添加充电桩位置
    ax3.scatter(coords[:, 0], coords[:, 1], c='blue', s=60, alpha=0.9, 
                edgecolors='white', linewidth=1, marker='o')
    
    # 添加网格线
    for i in range(len(x_bins)):
        ax3.axvline(x_bins[i], color='gray', linewidth=0.5, alpha=0.5)
    for i in range(len(y_bins)):
        ax3.axhline(y_bins[i], color='gray', linewidth=0.5, alpha=0.5)
    
    ax3.set_title('Step 3: Density Heatmap (stations/km²)')
    ax3.set_xlabel('X Coordinate (m)')
    ax3.set_ylabel('Y Coordinate (m)')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Density (stations/km²)')
    
    # 在网格中央显示密度值
    for i in range(grid_size):
        for j in range(grid_size):
            if density_grid[i, j] > 0:
                x_center = (x_bins[i] + x_bins[i+1]) / 2
                y_center = (y_bins[j] + y_bins[j+1]) / 2
                ax3.text(x_center, y_center, f'{density_grid[i, j]:.1f}',
                        ha='center', va='center', fontsize=6, 
                        color='black', weight='bold')
    
    # 4. 密度分布统计
    # 过滤掉没有桩的网格
    densities = density_grid.flatten()
    densities = densities[densities > 0]
    
    cs_density_std = np.std(densities) if len(densities) > 0 else 0.0
    density_mean = np.mean(densities) if len(densities) > 0 else 0.0
    
    # 绘制密度分布直方图
    ax4.hist(densities, bins=min(10, len(densities)), alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=1)
    ax4.axvline(density_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {density_mean:.2f}')
    ax4.axvline(density_mean - cs_density_std, color='orange', linestyle=':', linewidth=2, 
                label=f'Mean - Std: {density_mean - cs_density_std:.2f}')
    ax4.axvline(density_mean + cs_density_std, color='orange', linestyle=':', linewidth=2, 
                label=f'Mean + Std: {density_mean + cs_density_std:.2f}')
    
    ax4.set_title('Step 4: Density Distribution & Standard Deviation')
    ax4.set_xlabel('Density (stations/km²)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'''Statistics:
Grid Size: {grid_size}×{grid_size}
Grid Area: {grid_area_km2*1000000:.0f} m² ({grid_area_km2:.6f} km²)
Non-empty Grids: {len(densities)}/{grid_size*grid_size}
Density Mean: {density_mean:.3f} stations/km²
Density Std: {cs_density_std:.3f} stations/km²
CV (Std/Mean): {cs_density_std/density_mean:.3f}''' if density_mean > 0 else f'''Statistics:
Grid Size: {grid_size}×{grid_size}
Grid Area: {grid_area_km2*1000000:.0f} m² ({grid_area_km2:.6f} km²)
Non-empty Grids: {len(densities)}/{grid_size*grid_size}
Density Std: {cs_density_std:.3f} stations/km²'''
    
    ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             fontsize=8, family='monospace')
    
    plt.suptitle(f'{title}\nDensity Standard Deviation = {cs_density_std:.3f}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return cs_density_std, density_mean, densities, grid_area_km2

def demo_different_distributions():
    """演示不同分布模式的密度标准差差异"""
    
    # 1. 均匀分布
    np.random.seed(42)
    uniform_coords = np.random.uniform(0, 1000, (20, 2))
    
    # 2. 聚集分布
    cluster_coords = np.vstack([
        np.random.normal([300, 300], 100, (15, 2)),
        np.random.normal([700, 700], 80, (5, 2))
    ])
    
    # 3. 线性分布
    linear_coords = np.column_stack([
        np.linspace(100, 900, 20),
        np.random.normal(500, 50, 20)
    ])
    
    # 计算和显示
    plt.figure(figsize=(18, 6))
    
    # 均匀分布
    plt.subplot(1, 3, 1)
    std1, mean1, _, _ = calculate_density_std_simple(uniform_coords)
    plt.scatter(uniform_coords[:, 0], uniform_coords[:, 1], c='blue', s=80, alpha=0.7)
    plt.title(f'Uniform Distribution\nDensity Std = {std1:.3f}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, alpha=0.3)
    
    # 聚集分布
    plt.subplot(1, 3, 2)
    std2, mean2, _, _ = calculate_density_std_simple(cluster_coords)
    plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], c='red', s=80, alpha=0.7)
    plt.title(f'Clustered Distribution\nDensity Std = {std2:.3f}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, alpha=0.3)
    
    # 线性分布
    plt.subplot(1, 3, 3)
    std3, mean3, _, _ = calculate_density_std_simple(linear_coords)
    plt.scatter(linear_coords[:, 0], linear_coords[:, 1], c='green', s=80, alpha=0.7)
    plt.title(f'Linear Distribution\nDensity Std = {std3:.3f}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Density Standard Deviation for Different Distribution Patterns', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/density_std_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 密度标准差比较:")
    print(f"   均匀分布: {std1:.3f}")
    print(f"   聚集分布: {std2:.3f}")
    print(f"   线性分布: {std3:.3f}")
    print(f"\n💡 解释:")
    print(f"   - 聚集分布的密度标准差最高，表示分布最不均匀")
    print(f"   - 均匀分布的密度标准差最低，表示分布最均匀")
    print(f"   - 线性分布介于两者之间")

def calculate_density_std_simple(coords):
    """简化版本的密度标准差计算（用于比较演示）"""
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # 计算边界
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 划分网格
    grid_size = 10
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    
    # 计算每个网格的桩数
    grid_counts = np.zeros((grid_size, grid_size))
    cs_count = len(coords)
    
    for i in range(cs_count):
        x_idx = np.digitize(x_coords[i], x_bins) - 1
        y_idx = np.digitize(y_coords[i], y_bins) - 1
        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            grid_counts[x_idx, y_idx] += 1
    
    # 计算密度
    grid_area_km2 = ((x_max - x_min) / grid_size) * ((y_max - y_min) / grid_size) / 1000000
    densities = grid_counts.flatten() / grid_area_km2
    densities = densities[densities > 0]
    
    if len(densities) > 0:
        cs_density_std = np.std(densities)
        density_mean = np.mean(densities)
    else:
        cs_density_std = 0.0
        density_mean = 0.0
    
    return cs_density_std, density_mean, densities, grid_area_km2

if __name__ == "__main__":
    print("🎯 演示充电桩密度标准差 (cs_density_std) 的计算过程")
    
    # 创建示例数据：一个不均匀分布的充电桩布局
    np.random.seed(123)
    # 创建两个集群 + 一些散点
    cluster1 = np.random.normal([2000, 3000], 200, (8, 2))
    cluster2 = np.random.normal([4000, 5000], 150, (6, 2))
    scattered = np.random.uniform([1000, 2000], [5000, 6000], (4, 2))
    
    example_coords = np.vstack([cluster1, cluster2, scattered])
    
    # 详细演示计算过程
    print("\n📊 详细计算过程演示...")
    plt.figure(figsize=(15, 12))
    std_result, mean_result, densities, grid_area = calculate_density_std_demo(
        example_coords, 
        "Charging Station Density Standard Deviation Calculation"
    )
    
    plt.savefig('/home/ubuntu/project/MSC/Msc_Project/density_std_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 最终结果:")
    print(f"   cs_density_std = {std_result:.3f} stations/km²")
    print(f"   平均密度 = {mean_result:.3f} stations/km²")
    print(f"   变异系数 = {std_result/mean_result:.3f}" if mean_result > 0 else "   变异系数 = N/A")
    
    # 比较不同分布模式
    print(f"\n🔄 比较不同分布模式...")
    demo_different_distributions()
    
    print(f"\n📚 总结:")
    print(f"   cs_density_std 衡量充电桩空间分布的不均匀程度")
    print(f"   • 高值 = 分布不均匀（有聚集区域和稀疏区域）")
    print(f"   • 低值 = 分布均匀（密度在各区域相对一致）")
    print(f"   • 计算方法：10×10网格 → 计算各网格密度 → 标准差")
