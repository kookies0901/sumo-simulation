#!/usr/bin/env python3
"""
基于简化回归模型的插值数据集生成器
复制generate_graphs_simple.py的回归逻辑，在回归曲线附近插值
确保插值后数据的回归趋势保持一致
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import argparse
import warnings
warnings.filterwarnings('ignore')

def get_feature_performance_columns():
    """定义特征变量和性能指标列（复制自generate_graphs_simple.py）"""
    
    # 布局特征变量
    feature_columns = [
        'avg_dist_to_center',
        'std_nearest_neighbor', 
        'min_distance',
        'max_pairwise_distance',
        'cs_density_std',
        'cluster_count',
        'coverage_ratio',
        'max_gap_distance',
        'gini_coefficient',
        'avg_betweenness_centrality'
    ]
    
    # 性能指标变量
    performance_columns = [
        'duration_mean', 'duration_median', 'duration_p90',
        'charging_time_mean', 'charging_time_median', 'charging_time_p90',
        'waiting_time_mean', 'waiting_time_median', 'waiting_time_p90',
        'energy_gini', 'energy_cv', 'energy_hhi', 'energy_p90_p50_ratio',
        'vehicle_gini', 'vehicle_cv', 'vehicle_hhi',
        'charging_station_coverage', 'reroute_count',
        'ev_charging_participation_rate', 'ev_charging_failures'
    ]
    
    return feature_columns, performance_columns

def fit_simple_models(x, y):
    """训练Linear和Polynomial回归模型（复制自generate_graphs_simple.py）"""
    try:
        # 移除NaN值
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 5:
            return None, None, 0.0, "insufficient_data", {}, None
        
        # 重塑数据为sklearn格式
        X = x_clean.reshape(-1, 1)
        
        # 定义两个回归模型
        models = {
            'Linear': LinearRegression(),
            'Polynomial': LinearRegression()
        }
        
        model_results = {}
        best_model = None
        best_r2 = -np.inf
        best_model_name = ""
        best_poly_features = None
        
        for name, model in models.items():
            try:
                if name == 'Polynomial':
                    # 多项式回归
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(X)
                    model.fit(X_poly, y_clean)
                    y_pred = model.predict(X_poly)
                    
                    # 生成预测曲线
                    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
                    X_fit_poly = poly_features.transform(x_fit)
                    y_fit = model.predict(X_fit_poly)
                    current_poly_features = poly_features
                else:
                    # 线性回归
                    model.fit(X, y_clean)
                    y_pred = model.predict(X)
                    
                    # 生成预测曲线
                    x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
                    y_fit = model.predict(x_fit)
                    current_poly_features = None
                
                # 计算指标
                r2 = r2_score(y_clean, y_pred)
                mse = mean_squared_error(y_clean, y_pred)
                
                # 计算皮尔逊相关系数
                correlation, p_value = stats.pearsonr(x_clean, y_clean)
                
                model_results[name] = {
                    'r2': r2,
                    'mse': mse,
                    'correlation': correlation,
                    'p_value': p_value,
                    'x_fit': x_fit.flatten(),
                    'y_fit': y_fit,
                    'model': model,
                    'poly_features': current_poly_features
                }
                
                # 更新最佳模型
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name
                    best_model_name = name
                    best_poly_features = current_poly_features
                    
            except Exception as e:
                print(f"   ⚠️ 模型 {name} 训练失败: {e}")
                continue
        
        if best_model is None:
            return None, None, 0.0, "no_valid_model", {}, None
        
        # 返回最佳模型的结果
        best_result = model_results[best_model]
        return (best_result['x_fit'], best_result['y_fit'], 
                best_result['r2'], best_model_name, model_results, best_poly_features)
        
    except Exception as e:
        print(f"❌ 回归模型拟合失败: {e}")
        return None, None, 0.0, "fitting_error", {}, None

def predict_y_from_model(x_val, model_name, model, poly_features=None):
    """使用训练好的模型预测Y值"""
    try:
        x_array = np.array([[x_val]])
        
        if model_name == 'Polynomial' and poly_features is not None:
            x_poly = poly_features.transform(x_array)
            return model.predict(x_poly)[0]
        else:
            return model.predict(x_array)[0]
    except Exception as e:
        print(f"预测失败: {e}")
        return np.nan

def identify_sparse_regions(x_data, max_regions=None):
    """识别X轴稀疏区域"""
    x_clean = x_data[~np.isnan(x_data)]
    x_sorted = np.sort(x_clean)
    
    # 计算相邻点之间的间距
    gaps = np.diff(x_sorted)
    
    # 创建间隙信息列表
    gap_info = []
    for i, gap in enumerate(gaps):
        start_x = x_sorted[i]
        end_x = x_sorted[i + 1]
        gap_info.append((start_x, end_x, gap))
    
    # 按间隙大小降序排序
    gap_info.sort(key=lambda x: x[2], reverse=True)
    
    if max_regions:
        gap_info = gap_info[:max_regions]
    
    return gap_info

def add_noise(y_value, noise_factor=0.03):
    """添加噪声"""
    if np.isnan(y_value):
        return y_value
    
    noise = np.random.normal(0, abs(y_value) * noise_factor)
    return y_value + noise

def generate_regression_based_interpolations(df, target_count=6, noise_factor=0.03):
    """基于回归模型生成插值点"""
    
    feature_columns, performance_columns = get_feature_performance_columns()
    
    # 获取可用的特征和性能列
    available_features = [col for col in feature_columns if col in df.columns]
    available_performance = [col for col in performance_columns if col in df.columns]
    
    print(f"可用特征: {len(available_features)}")
    print(f"可用性能指标: {len(available_performance)}")
    
    # 关键特征对（与原脚本保持一致）
    important_pairs = [
        ('avg_dist_to_center', 'duration_mean'),
        ('avg_dist_to_center', 'charging_time_mean'),
        ('avg_dist_to_center', 'waiting_time_mean'),
        ('gini_coefficient', 'energy_gini'),
        ('cluster_count', 'charging_station_coverage'),
        ('std_nearest_neighbor', 'duration_p90'),
        ('max_pairwise_distance', 'energy_cv'),
        ('coverage_ratio', 'vehicle_gini')
    ]
    
    all_candidates = []
    
    print(f"\n🔍 分析回归关系并生成插值候选点...")
    
    for x_col, y_col in important_pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
            
        print(f"\n📊 处理: {x_col} vs {y_col}")
        
        x_data = df[x_col].values
        y_data = df[y_col].values
        
        # 使用与generate_graphs_simple.py相同的回归逻辑
        x_fit, y_fit, r2, model_name, model_results, poly_features = fit_simple_models(x_data, y_data)
        
        if x_fit is None:
            print(f"   ❌ 拟合失败")
            continue
            
        print(f"   📈 最佳模型: {model_name}, R² = {r2:.3f}")
        
        # 获取最佳模型
        best_model = model_results[model_name]['model']
        
        # 识别稀疏区域
        sparse_regions = identify_sparse_regions(x_data)
        
        print(f"   🎯 发现 {len(sparse_regions)} 个稀疏区域")
        
        # 在稀疏区域生成插值点
        for start_x, end_x, gap in sparse_regions:
            # 在大间隙中生成多个点
            gap_values = [g for _, _, g in sparse_regions]
            num_points_in_gap = 1
            
            if len(gap_values) > 1 and gap > np.percentile(gap_values, 75):
                median_gap = np.median(gap_values)
                if median_gap > 0:
                    num_points_in_gap = min(3, max(1, int(gap / median_gap)))
            
            # 在区间内生成点
            if num_points_in_gap == 1:
                x_values = [(start_x + end_x) / 2]
            else:
                x_values = np.linspace(start_x + gap*0.2, end_x - gap*0.2, num_points_in_gap)
            
            for i, x_val in enumerate(x_values):
                # 使用回归模型预测Y值
                y_pred = predict_y_from_model(x_val, model_name, best_model, poly_features)
                
                if not np.isnan(y_pred):
                    # 添加噪声
                    y_noisy = add_noise(y_pred, noise_factor=noise_factor)
                    
                    # 创建唯一标识符
                    point_id = f"{x_col}_{y_col}_{i}" if num_points_in_gap > 1 else f"{x_col}_{y_col}"
                    
                    # 存储候选点
                    all_candidates.append((gap, x_col, y_col, x_val, y_noisy, r2, point_id))
    
    # 按间隙大小和R²排序选择最佳点
    all_candidates.sort(key=lambda x: (x[0], x[5]), reverse=True)
    
    # 选择目标数量的点
    selected_points = []
    used_point_ids = set()
    
    print(f"\n✅ 选择前 {target_count} 个最佳插值点:")
    
    for gap, x_col, y_col, x_val, y_noisy, r2, point_id in all_candidates:
        if len(selected_points) >= target_count:
            break
            
        if point_id not in used_point_ids:
            selected_points.append((x_col, y_col, x_val, y_noisy))
            used_point_ids.add(point_id)
            print(f"   ✅ {x_col} = {x_val:.1f} → {y_col} = {y_noisy:.1f} (间隙: {gap:.1f}, R²: {r2:.3f})")
    
    return selected_points

def create_regression_based_dataset(input_file, output_file, target_count=6, noise_factor=0.03):
    """创建基于回归的插值数据集"""
    
    print("📊 读取原始数据...")
    df_original = pd.read_csv(input_file)
    print(f"   原始数据点: {len(df_original)}")
    
    # 生成基于回归的插值点
    print(f"\n🎯 生成 {target_count} 个基于回归曲线的插值点 (噪声: {noise_factor*100:.1f}%)...")
    interpolated_points = generate_regression_based_interpolations(df_original, target_count=target_count, noise_factor=noise_factor)
    
    if not interpolated_points:
        print("❌ 未能生成任何插值点")
        return df_original
    
    # 创建插值数据行
    interpolated_rows = []
    
    print(f"\n📝 创建插值数据行...")
    for row_counter, (x_col, y_col, x_val, y_val) in enumerate(interpolated_points, 1):
        # 从原数据的数值列中位数开始
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        new_row = df_original[numeric_cols].median().copy()
        
        # 添加非数值列
        for col in df_original.columns:
            if col not in numeric_cols:
                new_row[col] = f'cs_group_000_{row_counter:03d}' if col == 'layout_id' else df_original[col].mode().iloc[0]
        
        # 设置插值点的X和Y值
        new_row[x_col] = x_val
        new_row[y_col] = y_val
        
        # 为其他相关变量生成合理值
        for col in df_original.columns:
            if col != x_col and col != y_col and col != 'layout_id' and col in numeric_cols:
                # 基于与主X变量的相关性调整
                correlation = df_original[col].corr(df_original[x_col])
                if abs(correlation) > 0.1 and not np.isnan(correlation):
                    original_mean = df_original[col].mean()
                    x_normalized = (x_val - df_original[x_col].mean()) / df_original[x_col].std()
                    adjustment = correlation * x_normalized * df_original[col].std()
                    new_row[col] = original_mean + adjustment + np.random.normal(0, df_original[col].std() * 0.05)
        
        interpolated_rows.append(new_row)
        print(f"   📄 创建行 {row_counter}: {x_col}={x_val:.1f}, {y_col}={y_val:.1f}")
    
    # 合并数据
    df_interpolated = pd.DataFrame(interpolated_rows)
    df_combined = pd.concat([df_original, df_interpolated], ignore_index=True)
    
    # 保存结果
    df_combined.to_csv(output_file, index=False)
    
    print(f"\n✅ 基于回归曲线的插值数据集已创建!")
    print(f"   原始数据点: {len(df_original)}")
    print(f"   插值数据点: {len(df_interpolated)}")
    print(f"   总计数据点: {len(df_combined)}")
    print(f"   保存至: {output_file}")
    print(f"\n💡 现在可以运行 'python scripts/generate_graphs_simple.py' 验证回归趋势一致性")
    
    return df_combined

def main():
    parser = argparse.ArgumentParser(description='基于回归曲线生成插值数据集 - 保持趋势一致性')
    parser.add_argument('--count', '-c', type=int, default=6, 
                        help='插值点数量 (默认: 6)')
    parser.add_argument('--noise', '-n', type=float, default=0.03, 
                        help='噪声因子，百分比形式 (默认: 0.03，即3%%)')
    parser.add_argument('--input', '-i', type=str, 
                        default='/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/merged_dataset.csv',
                        help='输入文件路径')
    parser.add_argument('--output', '-o', type=str,
                        default='/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/dataset_interpolated.csv',
                        help='输出文件路径')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    print(f"🔧 参数设置:")
    print(f"   插值点数量: {args.count}")
    print(f"   噪声大小: {args.noise*100:.1f}%")
    print(f"   输入文件: {args.input}")
    print(f"   输出文件: {args.output}")
    print(f"   随机种子: {args.seed}")
    
    # 生成基于回归的插值数据集
    create_regression_based_dataset(args.input, args.output, args.count, args.noise)

if __name__ == '__main__':
    main()


