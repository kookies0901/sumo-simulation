#!/usr/bin/env python3
"""
动态性能指标定义和计算方式详解
20个关键性能指标的完整说明文档
"""

import pandas as pd
import os

def create_metrics_definition_guide():
    """创建详细的指标定义指南"""
    
    # 定义所有20个指标
    metrics_definitions = [
        {
            "指标名称": "duration_mean",
            "中文名称": "平均行驶时间",
            "分类": "车辆行程时间指标",
            "定义": "所有车辆从出发到到达目的地的平均行驶时间",
            "计算公式": "mean(车辆i的duration) for i in 所有车辆",
            "数据来源": "tripinfo_output.xml中的duration属性",
            "单位": "秒(s)",
            "解释": "反映整体交通流效率，值越小表示交通越顺畅",
            "影响因素": "充电站布局、交通拥堵、路网结构"
        },
        {
            "指标名称": "duration_median",
            "中文名称": "行驶时间中位数",
            "分类": "车辆行程时间指标",
            "定义": "所有车辆行驶时间的中位数（50%分位数）",
            "计算公式": "median(车辆i的duration) for i in 所有车辆",
            "数据来源": "tripinfo_output.xml中的duration属性",
            "单位": "秒(s)",
            "解释": "更稳健的中心趋势指标，不受极值影响",
            "影响因素": "网络拥堵分布、充电站可达性"
        },
        {
            "指标名称": "duration_p90",
            "中文名称": "行驶时间90分位数",
            "分类": "车辆行程时间指标", 
            "定义": "90%的车辆行驶时间低于此值",
            "计算公式": "percentile(车辆i的duration, 90) for i in 所有车辆",
            "数据来源": "tripinfo_output.xml中的duration属性",
            "单位": "秒(s)",
            "解释": "反映最不利情况下的行程时间，衡量系统可靠性",
            "影响因素": "极端拥堵、充电等待、路网瓶颈"
        },
        {
            "指标名称": "charging_time_mean",
            "中文名称": "平均充电时间",
            "分类": "充电时间指标",
            "定义": "所有充电事件的平均持续时间",
            "计算公式": "mean(充电桩j的chargingSteps) for j in 所有有充电活动的充电桩",
            "数据来源": "chargingevents.xml中的chargingSteps属性",
            "单位": "秒(s)",
            "解释": "反映平均充电服务时间，值越小表示充电效率越高",
            "影响因素": "充电桩功率、电池容量、充电策略"
        },
        {
            "指标名称": "charging_time_median", 
            "中文名称": "充电时间中位数",
            "分类": "充电时间指标",
            "定义": "所有充电事件时间的中位数",
            "计算公式": "median(充电桩j的chargingSteps) for j in 所有有充电活动的充电桩",
            "数据来源": "chargingevents.xml中的chargingSteps属性", 
            "单位": "秒(s)",
            "解释": "更稳健的充电时间中心趋势",
            "影响因素": "充电需求分布、充电桩配置"
        },
        {
            "指标名称": "charging_time_p90",
            "中文名称": "充电时间90分位数", 
            "分类": "充电时间指标",
            "定义": "90%的充电事件时间低于此值",
            "计算公式": "percentile(充电桩j的chargingSteps, 90) for j in 所有有充电活动的充电桩",
            "数据来源": "chargingevents.xml中的chargingSteps属性",
            "单位": "秒(s)", 
            "解释": "反映极端情况下的充电时间，衡量充电系统压力",
            "影响因素": "高峰期排队、充电桩故障、特殊充电需求"
        },
        {
            "指标名称": "waiting_time_mean",
            "中文名称": "平均等待时间",
            "分类": "车辆等待时间指标",
            "定义": "所有车辆的平均等待时间（包括交通信号、拥堵等待）",
            "计算公式": "mean(车辆i的waitingTime) for i in 所有车辆",
            "数据来源": "tripinfo_output.xml中的waitingTime属性",
            "单位": "秒(s)",
            "解释": "反映交通拥堵程度，值越大表示拥堵越严重",
            "影响因素": "交通流量、信号控制、路网容量、充电站排队"
        },
        {
            "指标名称": "waiting_time_median",
            "中文名称": "等待时间中位数",
            "分类": "车辆等待时间指标", 
            "定义": "所有车辆等待时间的中位数",
            "计算公式": "median(车辆i的waitingTime) for i in 所有车辆",
            "数据来源": "tripinfo_output.xml中的waitingTime属性",
            "单位": "秒(s)",
            "解释": "更稳健的等待时间指标，减少极值影响",
            "影响因素": "常规拥堵水平、交通组织"
        },
        {
            "指标名称": "waiting_time_p90",
            "中文名称": "等待时间90分位数",
            "分类": "车辆等待时间指标",
            "定义": "90%的车辆等待时间低于此值", 
            "计算公式": "percentile(车辆i的waitingTime, 90) for i in 所有车辆",
            "数据来源": "tripinfo_output.xml中的waitingTime属性",
            "单位": "秒(s)",
            "解释": "反映最坏情况下的等待时间，衡量系统鲁棒性",
            "影响因素": "严重拥堵、交通事故、充电站排队"
        },
        {
            "指标名称": "energy_gini",
            "中文名称": "充电量基尼系数",
            "分类": "能源分配公平性指标",
            "定义": "衡量充电桩间充电量分布的不均匀程度",
            "计算公式": "Gini(充电桩j的totalEnergyCharged) = (2Σ(i×x_i))/(n×Σx_i) - (n+1)/n",
            "数据来源": "chargingevents.xml中的totalEnergyCharged属性",
            "单位": "无量纲 [0,1]",
            "解释": "0表示完全均匀，1表示完全不均匀。值越大表示充电量分布越不平衡",
            "影响因素": "充电站布局、EV出行模式、充电站容量"
        },
        {
            "指标名称": "energy_cv",
            "中文名称": "充电量变异系数",
            "分类": "能源分配公平性指标", 
            "定义": "充电桩充电量的标准差与均值的比值",
            "计算公式": "CV = std(totalEnergyCharged) / mean(totalEnergyCharged)",
            "数据来源": "chargingevents.xml中的totalEnergyCharged属性",
            "单位": "无量纲",
            "解释": "衡量充电量的相对变异程度，值越大表示分布越分散",
            "影响因素": "充电需求分布、充电站设计、空间可达性"
        },
        {
            "指标名称": "energy_hhi", 
            "中文名称": "充电量HHI指数",
            "分类": "能源分配公平性指标",
            "定义": "赫芬达尔-赫希曼指数，衡量充电量集中程度",
            "计算公式": "HHI = Σ(s_i^2), 其中s_i是充电桩i的充电量份额",
            "数据来源": "chargingevents.xml中的totalEnergyCharged属性",
            "单位": "无量纲 [0,1]",
            "解释": "值越大表示充电量越集中在少数充电桩上，反映市场集中度",
            "影响因素": "充电站布局优化程度、EV路径选择"
        },
        {
            "指标名称": "energy_p90_p50_ratio",
            "中文名称": "充电量90/50分位数比值",
            "分类": "能源分配公平性指标",
            "定义": "90分位数充电量与50分位数充电量的比值",
            "计算公式": "P90(totalEnergyCharged) / P50(totalEnergyCharged)",
            "数据来源": "chargingevents.xml中的totalEnergyCharged属性", 
            "单位": "无量纲",
            "解释": "衡量高负荷充电桩与中等负荷充电桩的差异倍数",
            "影响因素": "充电站利用率分化、位置优势差异"
        },
        {
            "指标名称": "vehicle_gini",
            "中文名称": "充电车辆数基尼系数", 
            "分类": "车辆分配公平性指标",
            "定义": "衡量充电桩间服务车辆数分布的不均匀程度",
            "计算公式": "Gini(充电桩j服务的车辆数)",
            "数据来源": "chargingevents.xml中每个充电桩的vehicle元素数量",
            "单位": "无量纲 [0,1]",
            "解释": "反映充电桩服务负荷的公平性，值越大表示负荷分布越不均",
            "影响因素": "充电站容量配置、地理分布、交通流分布"
        },
        {
            "指标名称": "vehicle_cv",
            "中文名称": "充电车辆数变异系数",
            "分类": "车辆分配公平性指标",
            "定义": "充电桩服务车辆数的标准差与均值比值",
            "计算公式": "CV = std(车辆数) / mean(车辆数)",
            "数据来源": "chargingevents.xml中每个充电桩的vehicle元素数量",
            "单位": "无量纲", 
            "解释": "衡量充电桩服务量的相对离散程度",
            "影响因素": "充电站规模差异、位置吸引力"
        },
        {
            "指标名称": "vehicle_hhi",
            "中文名称": "充电车辆数HHI指数",
            "分类": "车辆分配公平性指标",
            "定义": "充电桩服务车辆数的集中度指数",
            "计算公式": "HHI = Σ(s_i^2), 其中s_i是充电桩i的服务车辆份额",
            "数据来源": "chargingevents.xml中每个充电桩的vehicle元素数量",
            "单位": "无量纲 [0,1]",
            "解释": "值越大表示服务越集中在少数充电桩，反映负荷集中程度",
            "影响因素": "充电站布局合理性、网络效应"
        },
        {
            "指标名称": "charging_station_coverage",
            "中文名称": "充电桩使用覆盖率",
            "分类": "基础设施利用率指标", 
            "定义": "有实际充电活动的充电桩占总充电桩数量的比例",
            "计算公式": "使用过的充电桩数量 / 总充电桩数量",
            "数据来源": "chargingevents.xml中totalEnergyCharged > 0的充电桩",
            "单位": "无量纲 [0,1]",
            "解释": "反映充电基础设施的利用程度，值越高表示利用率越好",
            "影响因素": "充电站布局合理性、充电需求分布、可达性"
        },
        {
            "指标名称": "reroute_count",
            "中文名称": "重新路由车辆数",
            "分类": "交通系统稳定性指标",
            "定义": "在行程中发生重新路由的车辆总数",
            "计算公式": "count(车辆i的rerouteNo > 0) for i in 所有车辆",
            "数据来源": "tripinfo_output.xml中的rerouteNo属性",
            "单位": "车辆数",
            "解释": "反映交通系统的不稳定性，值越大表示交通扰动越多",
            "影响因素": "交通拥堵、充电站排队、路网瓶颈、动态交通管理"
        },
        {
            "指标名称": "ev_charging_participation_rate", 
            "中文名称": "EV充电参与率",
            "分类": "充电服务覆盖指标",
            "定义": "实际参与充电的EV数量占总EV数量的比例",
            "计算公式": "实际充电的EV数量 / 总EV数量(1800)",
            "数据来源": "chargingevents.xml中ID以'EV_'开头的vehicle元素",
            "单位": "无量纲 [0,1]",
            "解释": "反映充电服务的覆盖程度，值越高表示更多EV能获得充电服务",
            "影响因素": "充电站密度、充电桩容量、EV行程特征、充电策略"
        },
        {
            "指标名称": "ev_charging_failures",
            "中文名称": "EV充电失败数",
            "分类": "充电服务质量指标",
            "定义": "未能成功完成充电的EV数量",
            "计算公式": "count(stationfinder元素) in tripinfo_output.xml",
            "数据来源": "tripinfo_output.xml中的stationfinder子元素",
            "单位": "车辆数", 
            "解释": "反映充电系统的服务失败情况，值越大表示充电服务质量越差",
            "影响因素": "充电站容量不足、位置不合理、充电桩故障、排队等待超时"
        }
    ]
    
    # 创建DataFrame
    df = pd.DataFrame(metrics_definitions)
    
    # 保存为CSV文件
    output_file = "analysis/charts/correlation/metrics_definition_guide.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 指标定义指南已保存到: {output_file}")
    
    # 创建简化的英文版本
    english_definitions = []
    for metric in metrics_definitions:
        english_definitions.append({
            "Metric": metric["指标名称"],
            "Category": metric["分类"],
            "Definition": metric["定义"],
            "Unit": metric["单位"],
            "Data_Source": metric["数据来源"]
        })
    
    df_english = pd.DataFrame(english_definitions)
    english_output_file = "analysis/charts/correlation/metrics_definition_english.csv"
    df_english.to_csv(english_output_file, index=False)
    print(f"✅ 英文版指标定义已保存到: {english_output_file}")
    
    return df, df_english

def create_metrics_summary_report():
    """创建指标摘要报告"""
    
    report_content = """
# 动态性能指标定义和计算方式详解

## 概述
本研究采用20个动态性能指标来全面评估充电站布局对电动车交通系统的影响。这些指标分为6大类：

## 指标分类体系

### 1. 车辆行程时间指标 (Vehicle Trip Time Metrics)
- **duration_mean**: 平均行驶时间 - 反映整体交通效率
- **duration_median**: 行驶时间中位数 - 稳健的中心趋势指标  
- **duration_p90**: 行驶时间90分位数 - 极端情况下的系统可靠性

### 2. 充电时间指标 (Charging Time Metrics)  
- **charging_time_mean**: 平均充电时间 - 充电效率指标
- **charging_time_median**: 充电时间中位数 - 典型充电服务时间
- **charging_time_p90**: 充电时间90分位数 - 充电系统压力指标

### 3. 车辆等待时间指标 (Vehicle Waiting Time Metrics)
- **waiting_time_mean**: 平均等待时间 - 交通拥堵程度
- **waiting_time_median**: 等待时间中位数 - 常规拥堵水平  
- **waiting_time_p90**: 等待时间90分位数 - 极端拥堵情况

### 4. 能源分配公平性指标 (Energy Distribution Equity Metrics)
- **energy_gini**: 充电量基尼系数 - 充电桩间负荷分布均匀程度
- **energy_cv**: 充电量变异系数 - 充电量相对变异程度
- **energy_hhi**: 充电量HHI指数 - 充电量集中度
- **energy_p90_p50_ratio**: 充电量90/50分位数比值 - 高低负荷差异

### 5. 车辆分配公平性指标 (Vehicle Distribution Equity Metrics)  
- **vehicle_gini**: 充电车辆数基尼系数 - 服务负荷分布公平性
- **vehicle_cv**: 充电车辆数变异系数 - 服务量离散程度
- **vehicle_hhi**: 充电车辆数HHI指数 - 服务集中度

### 6. 系统性能指标 (System Performance Metrics)
- **charging_station_coverage**: 充电桩使用覆盖率 - 基础设施利用率
- **reroute_count**: 重新路由车辆数 - 交通系统稳定性
- **ev_charging_participation_rate**: EV充电参与率 - 充电服务覆盖度
- **ev_charging_failures**: EV充电失败数 - 充电服务质量

## 数据来源说明

### SUMO仿真输出文件:
1. **tripinfo_output.xml**: 包含每辆车的行程信息
   - duration: 车辆总行程时间
   - waitingTime: 车辆总等待时间
   - rerouteNo: 重新路由次数
   - stationfinder: 充电失败标记

2. **chargingevents.xml**: 包含充电站使用信息  
   - chargingSteps: 充电持续时间
   - totalEnergyCharged: 总充电量
   - vehicle: 充电车辆列表

## 计算方法特点

### 统计指标选择原则:
- **Mean (均值)**: 反映总体平均水平
- **Median (中位数)**: 减少极值影响，更稳健
- **P90 (90分位数)**: 捕捉极端情况，评估系统鲁棒性

### 公平性指标含义:
- **Gini系数**: [0,1]，0=完全均匀，1=完全不均匀
- **变异系数CV**: 标准差/均值，衡量相对离散程度  
- **HHI指数**: [0,1]，衡量集中度，值越大越集中
- **P90/P50比值**: 高负荷与中等负荷的倍数关系

## 指标间相关性特征

根据相关性分析结果，主要发现:

1. **infrastructure-performance nexus**: 充电站覆盖率与多个性能指标强相关
2. **Concentration effects**: HHI指数类指标高度相关，反映资源分配模式
3. **Time performance clustering**: 各类时间指标内部高度相关
4. **Equity-efficiency trade-off**: 公平性与效率指标存在权衡关系

## 应用价值

这20个指标形成了一个全面的评估框架，能够:
- 量化充电站布局对交通系统的多维影响
- 识别不同布局策略的性能权衡  
- 为充电基础设施规划提供科学依据
- 支持多目标优化决策

"""
    
    # 保存报告
    report_file = "analysis/charts/correlation/metrics_definition_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 指标摘要报告已保存到: {report_file}")
    return report_file

def main():
    """主函数"""
    print("📊 创建动态性能指标定义指南")
    print("=" * 50)
    
    # 创建指标定义表格
    df_chinese, df_english = create_metrics_definition_guide()
    
    # 创建摘要报告
    report_file = create_metrics_summary_report()
    
    print(f"\n🎉 指标定义文档创建完成!")
    print(f"📁 生成的文件:")
    print(f"   • metrics_definition_guide.csv - 详细中文定义表格")
    print(f"   • metrics_definition_english.csv - 英文定义表格") 
    print(f"   • metrics_definition_report.md - 指标摘要报告")
    print(f"\n📊 指标统计:")
    print(f"   • 总计: 20个动态性能指标")
    print(f"   • 分类: 6大类指标体系")
    print(f"   • 数据源: 2个SUMO输出文件")
    
    # 显示分类统计
    category_counts = df_chinese['分类'].value_counts()
    print(f"\n📈 各类指标数量:")
    for category, count in category_counts.items():
        print(f"   • {category}: {count}个")

if __name__ == '__main__':
    main()
