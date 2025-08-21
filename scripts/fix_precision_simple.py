#!/usr/bin/env python3
"""
简单修复现有CSV文件中的精度问题
"""

import pandas as pd
import numpy as np

def fix_csv_precision(input_file, output_file, decimal_places=4):
    """修复CSV文件中的浮点精度问题"""
    
    print(f"📊 读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"原始数据形状: {df.shape}")
    
    # 识别数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    print(f"数值列数量: {len(numeric_columns)}")
    
    # 修复精度
    for col in numeric_columns:
        if col != 'reroute_count' and col != 'ev_charging_failures':  # 保持整数列不变
            df[col] = df[col].round(decimal_places)
    
    # 保存修复后的文件
    df.to_csv(output_file, index=False)
    
    print(f"✅ 修复完成，保存至: {output_file}")
    
    # 显示修复前后的对比
    print(f"\n🔍 修复示例 (取前3行的duration_p90列):")
    original_df = pd.read_csv(input_file)
    print("修复前:")
    for i in range(min(3, len(original_df))):
        if 'duration_p90' in original_df.columns:
            print(f"  行{i+1}: {original_df.loc[i, 'duration_p90']}")
    
    print("修复后:")
    for i in range(min(3, len(df))):
        if 'duration_p90' in df.columns:
            print(f"  行{i+1}: {df.loc[i, 'duration_p90']}")

def main():
    # 检查是否有批量分析结果文件
    input_file = '/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/batch_charging_analysis.csv'
    output_file = '/home/ubuntu/project/MSC/Msc_Project/models/input_1-100/batch_charging_analysis_fixed.csv'
    
    try:
        fix_csv_precision(input_file, output_file)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {input_file}")
        print("💡 请先运行批量分析生成数据文件")

if __name__ == '__main__':
    main()


