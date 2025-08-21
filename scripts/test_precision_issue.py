#!/usr/bin/env python3
"""
演示浮点精度问题和解决方案
"""

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

def demonstrate_precision_issue():
    """演示浮点精度问题"""
    print("🔍 浮点精度问题演示")
    print("="*50)
    
    # 模拟从XML读取的数据（字符串格式）
    xml_values = [
        "1652.3333333333333",
        "2341.6666666666665", 
        "3456.1000000000004",
        "4567.9000000000015",
        "5678.2333333333334"
    ]
    
    print("1. 原始XML数据:")
    for v in xml_values:
        print(f"   {v}")
    
    # 转换为float（原始方式）
    float_values = [float(v) for v in xml_values]
    print(f"\n2. 转换为float后:")
    for v in float_values:
        print(f"   {v}")
    
    # 计算P90（原始方式）
    p90_original = np.percentile(float_values, 90)
    print(f"\n3. 原始P90计算结果:")
    print(f"   {p90_original}")
    print(f"   {p90_original:.10f} (显示10位小数)")
    
    print("\n" + "="*50)
    print("✅ 修复后的方法:")
    
    # 修复方法1：使用round()
    rounded_values = [round(float(v), 4) for v in xml_values]
    p90_rounded = round(np.percentile(rounded_values, 90), 4)
    
    print(f"方法1 - round()修复:")
    print(f"   处理后的值: {rounded_values}")
    print(f"   P90结果: {p90_rounded}")
    
    # 修复方法2：使用Decimal
    def safe_float_convert(value_str, precision=4):
        decimal_val = Decimal(value_str)
        rounded_decimal = decimal_val.quantize(
            Decimal('0.' + '0' * precision), 
            rounding=ROUND_HALF_UP
        )
        return float(rounded_decimal)
    
    decimal_values = [safe_float_convert(v) for v in xml_values]
    p90_decimal = round(np.percentile(decimal_values, 90), 4)
    
    print(f"\n方法2 - Decimal修复:")
    print(f"   处理后的值: {decimal_values}")
    print(f"   P90结果: {p90_decimal}")

def show_interpolation_artifacts():
    """展示插值导致的精度问题"""
    print("\n🎯 插值精度问题演示")
    print("="*50)
    
    # 模拟SUMO数据中的典型情况
    values = np.array([1500.0, 1652.333333, 2100.666667, 2341.666667, 2800.1])
    
    print(f"原始数据: {values}")
    print(f"数据长度: {len(values)}")
    print(f"90%位置: {len(values) * 0.9} (需要插值)")
    
    # numpy插值计算过程
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = 0.9 * (n - 1)  # 90%分位数的索引位置
    
    print(f"\n插值计算过程:")
    print(f"   排序后: {sorted_vals}")
    print(f"   90%索引位置: {index}")
    
    if index != int(index):
        # 需要插值
        lower_index = int(index)
        upper_index = lower_index + 1
        fraction = index - lower_index
        
        lower_val = sorted_vals[lower_index]
        upper_val = sorted_vals[upper_index]
        
        interpolated = lower_val + fraction * (upper_val - lower_val)
        
        print(f"   下界值[{lower_index}]: {lower_val}")
        print(f"   上界值[{upper_index}]: {upper_val}")  
        print(f"   插值比例: {fraction}")
        print(f"   插值公式: {lower_val} + {fraction} * ({upper_val} - {lower_val})")
        print(f"   插值结果: {interpolated}")
        print(f"   精确显示: {interpolated:.15f}")
        
        # 使用numpy验证
        numpy_result = np.percentile(values, 90)
        print(f"   numpy结果: {numpy_result:.15f}")
        
        # 修复版本
        fixed_result = round(numpy_result, 4)
        print(f"   修复后: {fixed_result}")

if __name__ == '__main__':
    demonstrate_precision_issue()
    show_interpolation_artifacts()


