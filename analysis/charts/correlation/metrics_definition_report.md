
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

