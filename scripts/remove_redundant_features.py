#!/usr/bin/env python3
"""
删除merged_dataset.csv中的重复特征列
移除: cs_count, energy_zero_usage_rate, vehicle_zero_usage_rate
保留: charging_station_coverage (这三个指标实际上是重复的)
"""

import pandas as pd
import os
import shutil
from datetime import datetime

def remove_redundant_features(input_file, output_file=None, backup=True):
    """
    删除重复特征列
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径，如果为None则覆盖原文件
        backup: 是否创建备份文件
    """
    
    # 要删除的重复列
    columns_to_remove = [
        'cs_count',                    # 所有值都相同(215)，无变化
        'energy_zero_usage_rate',      # 与charging_station_coverage完全互补(r=-1.0)
        'vehicle_zero_usage_rate'      # 与energy_zero_usage_rate完全相同(r=1.0)
    ]
    
    print("🗑️ 移除重复特征脚本")
    print("="*50)
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    print(f"📊 输入文件: {input_file}")
    
    # 读取数据
    try:
        df = pd.read_csv(input_file)
        print(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # 检查要删除的列是否存在
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ 以下列不存在: {missing_columns}")
    
    if not existing_columns:
        print("ℹ️ 没有找到需要删除的列")
        return True
    
    print(f"🎯 将删除的列: {existing_columns}")
    
    # 创建备份
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = input_file.replace('.csv', f'_backup_{timestamp}.csv')
        try:
            shutil.copy2(input_file, backup_file)
            print(f"💾 备份文件已创建: {backup_file}")
        except Exception as e:
            print(f"⚠️ 备份创建失败: {e}")
    
    # 删除重复列
    df_cleaned = df.drop(columns=existing_columns)
    print(f"✅ 删除完成: {df.shape[1]} -> {df_cleaned.shape[1]} 列 (减少 {len(existing_columns)} 列)")
    
    # 显示剩余的特征列
    layout_features = [col for col in df_cleaned.columns if col not in ['layout_id', 'scenario_id']]
    performance_metrics = [col for col in df_cleaned.columns if col.startswith(('duration_', 'charging_time_', 'waiting_time_', 'energy_', 'vehicle_', 'charging_station_', 'reroute_', 'ev_'))]
    
    print(f"\n📋 清理后的数据结构:")
    print(f"   - 布局特征: {len([col for col in layout_features if col not in performance_metrics])} 个")
    print(f"   - 性能指标: {len(performance_metrics)} 个")
    print(f"   - 总特征数: {len(layout_features)} 个")
    
    # 确定输出文件
    if output_file is None:
        output_file = input_file
    
    # 保存清理后的数据
    try:
        df_cleaned.to_csv(output_file, index=False)
        print(f"💾 清理后的数据已保存: {output_file}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False
    
    # 显示删除的列的统计信息
    print(f"\n📊 已删除列的统计信息:")
    for col in existing_columns:
        if col in df.columns:
            print(f"   {col}:")
            print(f"     - 均值: {df[col].mean():.6f}")
            print(f"     - 范围: [{df[col].min():.6f}, {df[col].max():.6f}]")
            print(f"     - 唯一值数量: {df[col].nunique()}")
    
    print(f"\n🎉 特征清理完成！")
    print(f"📈 现在可以使用清理后的数据进行回归分析")
    
    return True

def main():
    # 设置文件路径
    input_file = "/home/ubuntu/project/MSC/Msc_Project/models/input/merged_dataset.csv"
    
    # 执行清理
    success = remove_redundant_features(
        input_file=input_file,
        output_file=None,  # 覆盖原文件
        backup=True        # 创建备份
    )
    
    if success:
        print(f"\n✅ 重复特征删除成功！")
        print(f"🔄 现在可以重新运行回归分析脚本")
        print(f"📁 清理后的文件: {input_file}")
    else:
        print(f"\n❌ 重复特征删除失败")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
