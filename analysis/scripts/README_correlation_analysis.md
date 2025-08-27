# 动态性能指标相关性分析工具

## 功能概述

这是一套专业的相关性分析工具，用于分析81个布局方案的20个动态性能指标之间的关系，生成皮尔逊（Pearson）和斯皮尔曼（Spearman）相关系数矩阵及专业热力图。

## 主要功能

### ✨ 核心特性
- **双重相关性分析**: 支持Pearson和Spearman相关性计算
- **专业可视化**: 生成高质量的相关性热力图
- **层次聚类**: 带聚类的热力图揭示指标群组关系
- **显著性检验**: 计算相关性的p值并标记显著性水平
- **智能分析**: 自动识别最强相关性并分类分析

### 📊 输出结果
1. **相关性矩阵** - CSV格式的完整相关系数矩阵
2. **热力图** - 标准版和聚类版热力图
3. **显著性检验** - p值矩阵和显著性标记
4. **描述性统计** - 各指标的基本统计信息
5. **分析报告** - 自动生成的结果总结

## 数据说明

### 输入数据
- **文件**: `models/input_1-100/merged_dataset.csv`
- **样本数**: 81个布局方案
- **指标数**: 20个动态性能指标

### 20个动态性能指标
```
行驶时间类 (3个):
1. duration_mean         - 平均行驶时间
2. duration_median       - 中位数行驶时间  
3. duration_p90          - 90分位数行驶时间

充电时间类 (3个):
4. charging_time_mean    - 平均充电时间
5. charging_time_median  - 中位数充电时间
6. charging_time_p90     - 90分位数充电时间

等待时间类 (3个):
7. waiting_time_mean     - 平均等待时间
8. waiting_time_median   - 中位数等待时间
9. waiting_time_p90      - 90分位数等待时间

能耗分布类 (4个):
10. energy_gini          - 能耗基尼系数
11. energy_cv            - 能耗变异系数
12. energy_hhi           - 能耗HHI指数
13. energy_p90_p50_ratio - 能耗P90/P50比值

车辆分布类 (3个):
14. vehicle_gini         - 车辆基尼系数
15. vehicle_cv           - 车辆变异系数
16. vehicle_hhi          - 车辆HHI指数

系统指标类 (4个):
17. charging_station_coverage    - 充电桩覆盖率
18. reroute_count               - 重新路由数量
19. ev_charging_participation_rate - EV充电参与率
20. ev_charging_failures        - EV充电失败数
```

## 使用方法

### 基本用法
```bash
# 运行完整的相关性分析（包含Pearson和Spearman）
python analysis/scripts/performance_correlation_analysis.py

# 只计算Pearson相关性
python analysis/scripts/performance_correlation_analysis.py --method pearson

# 只计算Spearman相关性
python analysis/scripts/performance_correlation_analysis.py --method spearman

# 自定义高相关性阈值
python analysis/scripts/performance_correlation_analysis.py --threshold 0.8

# 生成相关性总结
python analysis/scripts/correlation_summary.py
```

### 完整参数
```bash
python analysis/scripts/performance_correlation_analysis.py \
  --input models/input_1-100/merged_dataset.csv \
  --output_dir analysis/charts/correlation \
  --method both \
  --threshold 0.6 \
  --figsize 16,14
```

## 主要发现

### 🔥 最强相关性 (|r| > 0.9)

1. **vehicle_hhi ↔ charging_station_coverage** (r = -0.941)
   - 车辆分布集中度与充电桩覆盖率强负相关
   - 覆盖率高时，车辆分布更均匀

2. **charging_time_p90 ↔ charging_station_coverage** (r = -0.931)
   - 充电时间与充电桩覆盖率强负相关
   - 覆盖率高时，极端充电时间减少

3. **charging_time_p90 ↔ vehicle_hhi** (r = 0.926)
   - 充电时间与车辆分布集中度强正相关
   - 车辆分布不均时，充电时间显著增加

### 📈 相关性强度分布
- **极强相关** (|r| ≥ 0.8): 23对 (12.1%)
- **强相关** (0.6 ≤ |r| < 0.8): 64对 (33.7%)
- **中等相关** (0.3 ≤ |r| < 0.6): 65对 (34.2%)
- **弱相关** (|r| < 0.3): 38对 (20.0%)

### 🎯 指标群组特征

#### 充电系统效率群
- `charging_time_*` 指标内部高度相关
- 与 `vehicle_hhi` 和 `energy_hhi` 正相关
- 与 `charging_station_coverage` 负相关

#### 时间性能群
- `duration_*` 和 `waiting_time_*` 相关
- 体现系统整体性能协同效应

#### 分布不均度群
- `*_gini`, `*_cv`, `*_hhi` 指标相关
- 反映资源分配的不均衡性

## 输出文件结构

```
analysis/charts/correlation/
├── correlation_matrix_pearson.csv           # Pearson相关矩阵
├── correlation_matrix_spearman.csv          # Spearman相关矩阵
├── significance_matrix_pearson.csv          # Pearson显著性矩阵
├── significance_matrix_spearman.csv         # Spearman显著性矩阵
├── correlation_heatmap_pearson.png          # Pearson热力图
├── correlation_heatmap_spearman.png         # Spearman热力图
├── correlation_heatmap_clustered_pearson.png # Pearson聚类热力图
├── correlation_heatmap_clustered_spearman.png # Spearman聚类热力图
├── descriptive_statistics_pearson.csv       # 描述性统计
├── correlation_analysis_report_pearson.txt  # Pearson分析报告
├── correlation_analysis_report_spearman.txt # Spearman分析报告
└── summary/
    ├── top_correlations.png                 # 最强相关性条形图
    ├── categorized_correlation.png          # 分类相关性热力图
    └── correlation_summary_report.txt       # 总结报告
```

## 技术特性

### 统计方法
- **Pearson相关性**: 线性关系，适用于正态分布数据
- **Spearman相关性**: 单调关系，适用于非正态分布或有序数据
- **显著性检验**: p < 0.05 (*), p < 0.01 (**), p < 0.001 (***)

### 可视化特性
- **色彩映射**: 红-白-蓝渐变，红色负相关，蓝色正相关
- **层次聚类**: 基于相关性距离的聚类分析
- **高分辨率**: 300 DPI输出，适合学术发表
- **中英双语**: 标题和标签支持中英文

### 数据处理
- **缺失值处理**: 自动检测并使用中位数填充
- **数据验证**: 输入数据格式和完整性检查
- **性能优化**: 矩阵运算优化，支持大规模数据

## 解读指南

### 相关性强度解释
- **|r| ≥ 0.8**: 极强相关，几乎线性关系
- **0.6 ≤ |r| < 0.8**: 强相关，明显关联
- **0.3 ≤ |r| < 0.6**: 中等相关，有一定关联
- **|r| < 0.3**: 弱相关，关联较弱

### 业务含义
1. **负相关性**: 充电桩覆盖率与多数效率指标负相关，表明良好的基础设施布局能显著改善系统性能
2. **正相关群**: 各类时间指标间的正相关反映了系统性能的协同效应
3. **分布指标**: HHI、基尼系数等分布指标的相关性揭示了资源配置的不均衡特征

## 注意事项

### 统计假设
- 相关性不等于因果关系
- Pearson相关性假设线性关系
- 异常值可能影响相关性计算

### 数据要求
- 样本数量充足（当前81个样本）
- 数据质量良好，缺失值较少
- 指标单位和量级适当

### 解释限制
- 相关性分析仅显示关联强度
- 需结合业务知识进行深入解释
- 建议配合其他分析方法使用

## 扩展功能

### 自定义分析
可通过修改脚本实现：
- 不同指标子集的相关性分析
- 分组相关性分析（如按布局类型）
- 时间序列相关性分析
- 偏相关性分析

### 集成其他工具
- 可与回归分析工具结合
- 支持导出到其他统计软件
- 与机器学习管道集成

## 许可证

本工具为学术研究项目的一部分，遵循MIT许可证。
