# 🔋 Intelligent EV Charging Layout Optimization via SUMO Simulation and ML

This project aims to **optimize electric vehicle (EV) charging station layouts** using **realistic traffic simulations** and **machine learning (ML)** models. It is built upon **SUMO (Simulation of Urban Mobility)** and targets Glasgow’s real road network.

---

## 🎯 Project Goal

> Build a digital pipeline that simulates EV usage in real traffic, evaluates charging station layouts, and trains an ML model to **predict the effectiveness of future layouts** — enabling faster iteration without rerunning SUMO each time.

---

## 🗺️ Overall Workflow

```text
┌────────────────────┐
│ scenario_matrix.csv│◄────── experiment configs (EV count, cs count, layout ID, etc.)
└────────┬───────────┘
         │
         ▼
  load_scenario.py   ←── parses parameters
         │
         ▼
 clean_net.py        ←── cleans raw net.xml (removes drone, rail, etc.)
         │
         ▼
generate_cs_candidates.py
         │
         ▼
generator_charging_site.py
         │
         ▼
generator_trip.py  ←──── generates routes
         │
         ▼
generate_sumocfg.py
         │
         ▼
run_simulation.py  ←──── runs SUMO via TraCI, collects outputs
         │
         ▼
extract_layout_features.py
         │
         ▼
build_training_dataset.py  ←── combines simulation + layout for ML training

---

## 📁 File Structure
.
├── config/
│
├── data/
│   └── map/glasgow_clean.net.xml     # Cleaned map for Glasgow
|   |__ _dataset_1                      # One dataset of the experiment, like fixd EV number = 2000
|       |
│       └── scenario_matrix.csv       # Defines dataset1 configurations
|       |__ layout_registry.json      # Define  charging station layout of datatset1
│
├── sumo/
│   └── dataset_1
|       |
|       |__S001/                        # Example scenario output
│          ├── routes/                  # .rou.xml files
│          └── cs/                      # charging_stations.xml and layout_features_sample.csv
|       |__ output/                  # traci_data.csv
│
├── scripts/
│   ├── clean_net.py
│   ├── generate_cs_candidates.py
│   ├── generator_charging_site.py
│   ├── generator_trip.py
│   ├── generate_sumocfg.py
│   ├── run_simulation.py
│   ├── extract_layout_features.py
│   ├── build_training_dataset.py
│   └── load_scenario.py
```

---

## 📊 Output Dataset Format (for ML)

After simulation + feature extraction, the ML-ready training dataset will include:

| scenario_id | cs_layout_id | num_cs | layout_features (e.g. coords, spread) | avg_wait_time | avg_charge_time |
| ------------ | -------------- | ------- | -------------------------------------- | --------------- | ----------------- |
| S001         | cs_group_001 | 13      | [x1,y1,x2,y2,...]                     | 142.5 s         | 370.2 s           |

---

## 🤖 Potential Use Cases

* Predict best-performing charging layouts **without rerunning SUMO**
* Use ML model to explore large layout design space
* Fine-tune layout strategies for large-scale deployment cities

---

## 🚧 Current Progress

✅ Cleaned Glasgow net.xml
✅ Parsed scenario configs
✅ Generated charging station layouts
✅ Generated trips, sumocfg, and ran SUMO simulations
✅ Extracted SOC + waiting time
✅ Created layout feature vectors
🔄 Next: batch training ML model + inference on new layouts

---

## 🛠️ Requirements

* Python ≥ 3.10
* SUMO (with TraCI)
* pandas, numpy, scikit-learn
* Optional: torch, xgboost for modeling

---

## ✍️ Author

**Tingting Yang**, MSc Computing Science (University of Glasgow)
Special focus: intelligent systems, mobility simulation, AI companionship research

For contributions, issues or guidance, feel free to contact or open a pull request.

---
目标的文件系统结构（统一规范）

# 📁 scripts/ 说明文档

本目录下为 MSC 项目的主要数据生成与仿真脚本。整体结构如下：

pipeline
├── clean_net.py                 # 清洗原始 net.xml，去除非法车道类型
│
├── generate_cs_candidates.py    # 从清洗后的 net.xml 中筛选候选充电桩选址点
│
├── generator_charging_site.py   # 根据 scenario_matrix.csv 生成 charging_stations.xml
│
├── generator_trip.py            # 生成 SUMO trip 文件（.trip.xml）并转换为 .rou.xml
│
├── generate_sumocfg.py          # 生成对应的 sumo config (.sumocfg) 文件
│
├── run_simulation.py            # 使用 TraCI 接口运行 SUMO 仿真，记录输出指标
│
├── extract_layout_features.py   # 从 charging_stations.xml 中提取布局特征向量
│
├── build_training_dataset.py    # 汇总多个仿真输出和布局特征，构建模型训练数据集
│
├── load_scenario.py             # 加载 scenario_matrix.csv 的参数配置


## 💡 建议执行顺序
1. `clean_net.py` → `generate_cs_candidates.py` → 生成候选桩点  
2. `generator_charging_site.py` → 根据布局方案生成 charging_stations.xml  
3. `generator_trip.py` → 生成路由  
4. `generate_sumocfg.py` → 生成配置文件  
5. `run_simulation.py` → 执行仿真并采集指标  
6. `extract_layout_features.py` + `build_training_dataset.py` → 构建模型训练数据

-------------------------------------------------------------------------
# cursor analysis summary
Ran tool
Read file: README.md

---
### 项目简介

MSC_PROJECT旨在通过**SUMO交通仿真**与**机器学习**，优化电动汽车（EV）充电站布局。项目以格拉斯哥真实路网为基础，构建数字化仿真与数据分析管道，最终目标是训练ML模型，**预测不同充电站布局的效果**，从而无需每次都运行SUMO仿真即可快速评估新布局。

---

### 工作流程

1. **配置实验参数**（如EV数量、充电站数量、布局ID等）——`scenario_matrix.csv`
2. **加载参数**——`load_scenario.py`
3. **清洗路网**——`clean_net.py`
4. **生成充电站候选点**——`generate_cs_candidates.py`
5. **生成充电站布局**——`generator_charging_site.py`
6. **生成出行路线**——`generator_trip.py`
7. **生成SUMO配置**——`generate_sumocfg.py`
8. **运行仿真并采集数据**——`run_simulation.py`
9. **提取布局特征**——`extract_layout_features.py`
10. **构建ML训练数据集**——`build_training_dataset.py`

---

### 目录结构

- `config/`：配置文件
- `data/`：地图、实验数据、布局定义等
- `sumo/`：仿真输出（如路由、充电站、仿真结果等）
- `scripts/`：所有核心脚本（见下方详细说明）

---

### scripts/ 目录说明

- `clean_net.py`：清洗原始net.xml，去除不需要的车道类型
- `generate_cs_candidates.py`：筛选充电桩候选点
- `generator_charging_site.py`：根据实验参数生成充电站布局
- `generator_trip.py`：生成车辆出行路线
- `generate_sumocfg.py`：生成SUMO仿真配置文件
- `run_simulation.py`：运行SUMO仿真并采集输出
- `extract_layout_features.py`：提取布局特征向量
- `build_training_dataset.py`：汇总仿真与布局特征，生成ML训练数据
- `load_scenario.py`：加载实验参数配置

**建议执行顺序**：  
1. clean_net.py → generate_cs_candidates.py  
2. generator_charging_site.py  
3. generator_trip.py  
4. generate_sumocfg.py  
5. run_simulation.py  
6. extract_layout_features.py + build_training_dataset.py

---

### 输出数据格式（供ML训练）

| scenario_id | cs_layout_id | num_cs | layout_features | avg_wait_time | avg_charge_time |
| ----------- | ----------- | ------ | -------------- | ------------- | -------------- |
| S001        | cs_group_001| 13     | [x1,y1,x2,...] | 142.5 s       | 370.2 s        |

---

### 依赖要求

- Python ≥ 3.10
- SUMO（含TraCI接口）
- pandas, numpy, scikit-learn
- 可选：torch, xgboost
