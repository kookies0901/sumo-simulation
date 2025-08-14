#!/usr/bin/env python3
"""
合并充电桩布局特征和性能指标数据
将all_layout_features.csv和batch_charging_analysis.csv按照layout_id合并
"""

import os
import pandas as pd
import argparse

def load_layout_features(features_file):
    """加载布局特征数据"""
    try:
        df = pd.read_csv(features_file)
        print(f"✅ 布局特征数据加载成功: {len(df)} 行")
        print(f"📊 特征列数: {len(df.columns)}")
        print(f"🏷️ 布局ID范围: {df['layout_id'].min()} - {df['layout_id'].max()}")
        return df
    except Exception as e:
        print(f"❌ 加载布局特征失败: {e}")
        return None

def load_performance_metrics(metrics_file):
    """加载性能指标数据"""
    try:
        df = pd.read_csv(metrics_file)
        print(f"✅ 性能指标数据加载成功: {len(df)} 行")
        print(f"📊 指标列数: {len(df.columns)}")
        
        # 检查layout_id列名
        layout_id_cols = [col for col in df.columns if 'layout_id' in col.lower()]
        print(f"🏷️ 发现layout_id相关列: {layout_id_cols}")
        
        # 使用最后一个包含layout_id的列作为合并键
        if layout_id_cols:
            layout_id_col = layout_id_cols[-1]  # 使用最后一个
            if layout_id_col != 'layout_id':
                df = df.rename(columns={layout_id_col: 'layout_id'})
                print(f"🔄 重命名列: {layout_id_col} -> layout_id")
        
        if 'layout_id' in df.columns:
            print(f"🏷️ 布局ID范围: {df['layout_id'].min()} - {df['layout_id'].max()}")
        else:
            print("❌ 未找到layout_id列")
            
        return df
    except Exception as e:
        print(f"❌ 加载性能指标失败: {e}")
        return None

def merge_datasets(features_df, metrics_df):
    """合并两个数据集"""
    try:
        print("\n🔄 开始合并数据集...")
        
        # 检查合并前的数据
        print(f"📊 特征数据: {len(features_df)} 行, {len(features_df.columns)} 列")
        print(f"📊 指标数据: {len(metrics_df)} 行, {len(metrics_df.columns)} 列")
        
        # 检查layout_id的交集
        features_ids = set(features_df['layout_id'])
        metrics_ids = set(metrics_df['layout_id'])
        
        common_ids = features_ids.intersection(metrics_ids)
        print(f"🎯 共同的layout_id数量: {len(common_ids)}")
        print(f"🔍 特征数据独有: {len(features_ids - metrics_ids)}")
        print(f"🔍 指标数据独有: {len(metrics_ids - features_ids)}")
        
        if len(common_ids) == 0:
            print("❌ 没有共同的layout_id，无法合并")
            return None
        
        # 执行内连接合并
        merged_df = pd.merge(features_df, metrics_df, on='layout_id', how='inner')
        
        print(f"✅ 合并完成: {len(merged_df)} 行, {len(merged_df.columns)} 列")
        print(f"🎉 成功合并率: {len(merged_df)/len(common_ids)*100:.1f}%")
        
        return merged_df
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return None

def save_merged_dataset(merged_df, output_file):
    """保存合并后的数据集"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        merged_df.to_csv(output_file, index=False)
        print(f"💾 合并数据集已保存: {output_file}")
        
        # 显示数据集信息
        print(f"\n📋 最终数据集信息:")
        print(f"   - 总行数: {len(merged_df)}")
        print(f"   - 总列数: {len(merged_df.columns)}")
        print(f"   - 文件大小: {os.path.getsize(output_file)/1024:.1f} KB")
        
        # 显示列名
        print(f"\n📊 数据列概览:")
        feature_cols = []
        metric_cols = []
        
        for col in merged_df.columns:
            if col == 'layout_id':
                continue
            elif col in ['cs_count', 'avg_dist_to_center', 'avg_nearest_neighbor', 
                        'std_nearest_neighbor', 'min_distance', 'max_pairwise_distance',
                        'cs_density_std', 'cluster_count', 'coverage_ratio', 
                        'max_gap_distance', 'gini_coefficient', 'avg_betweenness_centrality']:
                feature_cols.append(col)
            else:
                metric_cols.append(col)
        
        print(f"   🏗️ 布局特征 ({len(feature_cols)}): {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
        print(f"   📈 性能指标 ({len(metric_cols)}): {', '.join(metric_cols[:5])}{'...' if len(metric_cols) > 5 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='合并充电桩布局特征和性能指标数据')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/ubuntu/project/MSC/Msc_Project/models/input',
                       help='输入文件目录')
    parser.add_argument('--features_file', type=str,
                       default='all_layout_features.csv',
                       help='布局特征文件名')
    parser.add_argument('--metrics_file', type=str,
                       default='batch_charging_analysis.csv',
                       help='性能指标文件名')
    parser.add_argument('--output_file', type=str,
                       default='merged_dataset.csv',
                       help='输出文件名')
    
    args = parser.parse_args()
    
    # 构建完整路径
    features_path = os.path.join(args.input_dir, args.features_file)
    metrics_path = os.path.join(args.input_dir, args.metrics_file)
    output_path = os.path.join(args.input_dir, args.output_file)
    
    print("🚀 开始合并充电桩布局特征和性能指标数据")
    print(f"📁 输入目录: {args.input_dir}")
    print(f"🏗️ 特征文件: {features_path}")
    print(f"📈 指标文件: {metrics_path}")
    print(f"💾 输出文件: {output_path}")
    
    # 检查输入文件是否存在
    if not os.path.exists(features_path):
        print(f"❌ 特征文件不存在: {features_path}")
        return 1
    
    if not os.path.exists(metrics_path):
        print(f"❌ 指标文件不存在: {metrics_path}")
        return 1
    
    # 加载数据
    print(f"\n📖 加载布局特征数据...")
    features_df = load_layout_features(features_path)
    if features_df is None:
        return 1
    
    print(f"\n📖 加载性能指标数据...")
    metrics_df = load_performance_metrics(metrics_path)
    if metrics_df is None:
        return 1
    
    # 合并数据
    merged_df = merge_datasets(features_df, metrics_df)
    if merged_df is None:
        return 1
    
    # 保存结果
    success = save_merged_dataset(merged_df, output_path)
    if not success:
        return 1
    
    print(f"\n🎉 数据合并完成！")
    print(f"📊 最终数据集包含 {len(merged_df)} 个充电桩布局的完整特征和性能数据")
    
    return 0

if __name__ == '__main__':
    exit(main())
