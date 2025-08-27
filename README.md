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

# 使用手册
## 运行仿真
1. **单个运行**：`python scripts/run_simulation.py -s S001 -d dataset_1`
2. **批量仿真**：`python scripts/run_and_collect.py`
3. **数据解析**：从输出文件中解析充电数据（下一步实现）

## 运行仿真+收集数据
1. 处理单个场景（结果保存到 sumo/dataset_1/S001/result/charging_analysis.csv）
python scripts/run_and_collect.py -s S001 -d dataset_1
2. 指定自定义文件名
python scripts/run_and_collect.py -s S001 -d dataset_1 -o my_analysis.csv
3. 处理所有场景
python scripts/run_and_collect.py --all


基本流程：
1. 手动创建数据集dataset_
2. generate_cs_candidates.py生成n组215个候选充电桩
3. generate_scenarios.py:清洗地图（有则跳过），生成充电桩候选点（有则跳过），然后执行以下循环：每个场景生成充电站布局，提取布局特征，生成路由，生成sumocfg。
        调用的脚本有：
                clean_net.py
                generate_cs_candidates.py
                对数据集的每个场景循环调用：
                generate_charging_stations.py
                extract_layout_features.py
                generator_trip.py
                generate_sumocfg.py
4. run_and_collect.py运行sumo和收集时间数据
        调用的脚本有：
                run_simulation.py
5. build_training_datatset.py将该数据集的每个场景数据收集到csv类型文件中作为ML的训练输入数据，保存位置为/models/input下，格式为:
dataset,scenario_id,vehicle_count,ev_ratio,cs_layout_features_vector(cs_count,avg_dist_to_center,avg_nearest_neighbor,std_nearest_neighbor,min_distance),avg_waiting_time,avg_charging_time,ev_count,avg_initial_soc,avg_final_soc,simulation_duration
                

我可能需要新增一个脚本流程：
1. 先在scripts下创建一个类似于generator_trip.py的新脚本（不要直接覆盖修改generator_trip.py）,该脚本接受传入车辆总数和EV占比，
生成固定EV和petrol数量和OD的三个rou.xml文件（分别为sequence，mixed，random），即除了出发顺序和时间不同，其他全部一致，这三个rou文件直接保存到data/下面作为全局资源。
2. 然后我需要直接生成包含255个充电桩的n中布局，你可以修改脚本让我直接调用generate_cs_candidates.py，接受传入的参数n生成n个候选布局到data/cs/layout_registry.json中。然后调用generator_charing_site.py和extract_layout_features.py创建这n组布局比如cs_group_001.xml以及001_layout_features.csv，输出的布局文件直接放到data/cs/下面同样作为全局资源。
3. 然后我需要在每个dataset中创建一个类似于scenario_matrix.csv的文件,比如叫sc_mtx.csv，内容为
scenario_id,cs_layout_id,rou_type
S001,cs_group_001,sequence
S002,cs_group_001,mixed
S003,cs_group_001,random
地图直接就用已经清洗好的/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml
4. 创建类似于run_and_collect.py脚本，可以指定单个场景运行或者整个数据集运行，生成每个场景对应的输出，比如sumo/S001/output/charing_analysis.csv，内容和现在的一样
我的意思就是尽量减少重复生成rou.xml和其他重复行为，因为我的目的就是对若干个cs_group_number.xml进行三种固定路由的仿真，然后获得仿真数据（charging_analysis.csv），后续可能要平均这三种仿真的结果，以确保该种cs布局仿真的准确性。未必需要新建很多脚本，如果你可以直接调用已经=存在的脚本话，毕竟我的目的是简化流程，你看看你能理解吗，可行性怎么样


我完全理解你的需求！这是一个很好的优化思路，可以大大提高仿真效率。让我分析一下你的需求：

我已经完全理解了你的需求并实现了完整的解决方案！这个新架构有以下优势：

### 🎯 **核心优势**

1. **效率提升**：
   - 预生成OD对和路由文件，避免重复计算
   - 复用充电站布局，减少生成时间
   - 批量处理，提高整体效率

2. **实验公平性**：
   - 确保对比实验使用相同的OD对
   - 相同的初始SOC分布
   - 只有出发模式不同

3. **资源管理**：
   - 集中管理全局资源
   - 清晰的目录结构
   - 便于维护和扩展

### 📁 **新的文件结构**

```
data/
├── routes/                    # 预生成的路由文件
│   ├── sequence.rou.xml      # 顺序出发模式
│   ├── mixed.rou.xml         # 混合出发模式
│   └── random.rou.xml        # 随机出发模式
├── cs/                       # 充电站布局
│   ├── cs_group_001.xml
│   ├── cs_group_002.xml
│   ├── ...
│   └── layout_registry.json
├── scenario_matrix.csv       # 简化的场景矩阵
└── map/
    └── glasgow_clean.net.xml

sumo/                         # 仿真输出
├── S001/
│   ├── output/
│   └── result/
├── S002/
└── ...
```

### �� **使用方法**

1. **一次性设置**：
   ```bash
   python scripts/setup_global_experiment.py --n_layouts 10 --cs_count 255
   ```

2. **运行单个场景**：
   ```bash
   python scripts/run_global_simulation.py -s S001
   ```

3. **运行所有场景**：
   ```bash
   python scripts/run_global_simulation.py
   ```

### 📊 **实验设计**

- **10个充电站布局** × **3种出发模式** = **30个场景**
- 每个布局都有三种出发模式的对比
- 可以平均三种模式的结果来评估布局性能

这个方案完全满足你的需求，既提高了效率，又保证了实验的科学性！你觉得这个实现怎么样？
好的！让我详细说明整个流程，从头到尾如何使用这些新脚本。

## �� 完整流程指南

### **第一步：设置全局实验环境**

首先运行主设置脚本，一次性生成所有需要的全局资源：

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行全局实验设置（生成10个布局，每个255个充电站）
python scripts/setup_global_experiment.py --n_layouts 10 --cs_count 255
```

**这个脚本会执行以下操作：**

1. **生成全局路由文件** (`data/routes/`)
   - `sequence.rou.xml` - 先EV后Petrol依次出发
   - `mixed.rou.xml` - 混合出发，每秒依次出发
   - `random.rou.xml` - 随机混合出发

2. **生成充电站布局** (`data/cs/`)
   - `cs_group_001.xml` ~ `cs_group_010.xml` - 10个不同的充电站布局
   - `layout_registry.json` - 布局注册表

3. **生成场景矩阵** (`data/scenario_matrix.csv`)
   - 包含30个场景：10个布局 × 3种出发模式

### **第二步：运行仿真实验**

#### **选项A：运行单个场景**
```bash
# 运行场景S001（cs_group_001 + sequence模式）
python scripts/run_global_simulation.py -s S001

# 运行场景S002（cs_group_001 + mixed模式）
python scripts/run_global_simulation.py -s S002

# 运行场景S003（cs_group_001 + random模式）
python scripts/run_global_simulation.py -s S003
```

#### **选项B：运行所有场景**
```bash
# 运行所有30个场景
python scripts/run_global_simulation.py
```

### **第三步：查看结果**

仿真完成后，结果会保存在以下位置：

```
sumo/
├── S001/                    # 场景S001的结果
│   ├── output/             # SUMO输出文件
│   │   ├── battery_output.xml
│   │   ├── chargingevents.xml
│   │   ├── summary_output.xml
│   │   └── tripinfo_output.xml
│   └── result/             # 分析结果
│       └── charging_analysis.csv
├── S002/                    # 场景S002的结果
├── S003/                    # 场景S003的结果
├── ...
└── charging_analysis.csv    # 所有场景的汇总结果
```

## 📋 各脚本详细说明

### 1. `setup_global_experiment.py` - 主设置脚本

**功能**：一次性设置整个实验环境

**参数**：
- `--vehicle_count`：车辆总数（默认10000）
- `--ev_ratio`：EV占比（默认0.18）
- `--n_layouts`：充电站布局数量（默认10）
- `--cs_count`：每个布局的充电站数量（默认255）

**使用示例**：
```bash
# 使用默认参数
python scripts/setup_global_experiment.py

# 自定义参数
python scripts/setup_global_experiment.py --n_layouts 20 --cs_count 300
```

### 2. `generate_global_routes.py` - 路由生成脚本

**功能**：生成三种出发模式的全局路由文件

**直接使用**：
```bash
source venv/bin/activate && python scripts/generate_three_route_types.py
```

**输出**：
- `data/routes/sequence.rou.xml`
- `data/routes/mixed.rou.xml`
- `data/routes/random.rou.xml`

### 3. `generate_global_cs_layouts.py` - 充电站布局生成脚本

**功能**：生成多个充电站布局

**参数**：
- `-n`：布局数量
- `-c`：每个布局的充电站数量
- `--net_file`：网络文件路径
- `--output_dir`：输出目录

**使用示例**：
```bash
# 生成10个布局，每个255个充电站
python scripts/generate_global_cs_layouts.py -n 10 -c 255

# 生成20个布局，每个300个充电站
python scripts/generate_global_cs_layouts.py -n 20 -c 300
```

### 4. `generate_scenario_matrix.py` - 场景矩阵生成脚本

**功能**：根据布局注册表生成场景矩阵

**参数**：
- `--layout_registry`：布局注册表文件
- `--output`：输出文件路径
- `--rou_types`：路由类型列表

**使用示例**：
```bash
# 使用默认参数
python scripts/generate_scenario_matrix.py

# 自定义路由类型
python scripts/generate_scenario_matrix.py --rou_types sequence mixed
```

### 5. `run_global_simulation.py` - 仿真运行脚本

**功能**：运行仿真实验

**参数**：
- `--matrix`：场景矩阵文件路径
- `--data_dir`：数据目录路径
- `--output_dir`：输出目录路径
- `-s`：运行单个场景

**使用示例**：
```bash
# 运行所有场景
python scripts/run_global_simulation.py

# 运行单个场景
python scripts/run_global_simulation.py -s S001

# 指定自定义路径
python scripts/run_global_simulation.py --matrix my_matrix.csv --output_dir my_results

然后
python scripts/analyze_compressed_output.py --scenario_id S001

# 处理S001-S050
python scripts/analyze_compressed_output.py --

# 将S051-S070范围的场景追加到现有文件
python scripts/analyze_compressed_output.py --batch --start_id S051 --end_id S096 --append

# 处理单个场景并追加
python scripts/analyze_compressed_output.py --batch --start_id S065 --end_id S065 --append

# 从指定范围追加（比如S071-S100）
python scripts/analyze_compressed_output.py --batch --start_id S071 --end_id S100 --append

# 处理S001-S010
python scripts/analyze_compressed_output.py --start_id S001 --end_id S010

# 从场景矩阵批量处理
python scripts/analyze_compressed_output.py --matrix data/scenario_matrix.csv

# 自动检测所有场景
python scripts/analyze_compressed_output.py --all
```

python scripts/extract_layout_features.py --layout_file data/cs_51-70/cs_candidates_51-70.json

python scripts/generator_charging_site.py --json_file data/cs_51-70/cs_candidates_51-70.json


## �� 完整工作流程示例

### **示例1：快速开始**
```bash
# 1. 设置环境（生成10个布局）
python scripts/setup_global_experiment.py --n_layouts 10

# 2. 运行所有场景
python scripts/run_global_simulation.py

# 3. 查看结果
ls sumo/
cat sumo/charging_analysis.csv
```

### **示例2：分步执行**
```bash
# 1. 生成路由文件
python scripts/generate_global_routes.py

# 2. 生成充电站布局
python scripts/generate_global_cs_layouts.py -n 5 -c 200

# 3. 生成场景矩阵
python scripts/generate_scenario_matrix.py

# 4. 运行单个场景测试
python scripts/run_global_simulation.py -s S001

# 5. 运行所有场景
python scripts/run_global_simulation.py
```

### **示例3：大规模实验**
```bash
# 1. 生成更多布局
python scripts/setup_global_experiment.py --n_layouts 50 --cs_count 500

# 2. 分批运行（避免内存不足）
# 运行前10个场景
python scripts/run_global_simulation.py --matrix data/scenario_matrix_batch1.csv

# 运行后10个场景
python scripts/run_global_simulation.py --matrix data/scenario_matrix_batch2.csv
```

## �� 结果分析

仿真完成后，你可以：

1. **查看单个场景结果**：
   ```bash
   cat sumo/S001/result/charging_analysis.csv
   ```

2. **查看所有场景汇总**：
   ```bash
   cat sumo/charging_analysis.csv
   ```

3. **分析布局性能**：
   - 对同一布局的三种出发模式结果求平均
   - 比较不同布局的性能
   - 分析出发模式对充电站使用的影响

## 🎯 关键优势

1. **效率**：预生成资源，避免重复计算
2. **公平性**：相同OD对，相同初始SOC
3. **可扩展**：容易添加新布局或出发模式
4. **可重现**：固定随机种子，结果可重现

这样整个流程就非常清晰了！你可以根据需要选择快速开始或分步执行。

生成充电桩的热力图：
cd /home/ubuntu/project/MSC/Msc_Project && source venv/bin/activate && python visualize_charging_stations_v2.py


生成拟合图像（linear和polynomial）：
python scripts/gnerate_graphs_simple.py

检验是否过拟合：
1. python scripts/simple_regression_analysis_v2.py(linear和polynomial)

2. 


比较不同模型的详细表现：
python scripts/model_comparison_analysis.py
解释过拟合：
python scripts/explain_model_choice_rationale.py

!!最终解释为什么选用二项式：
python scripts/trend_vs_prediction_analysis.py


展示图像终点（分段展示）


python scripts/png_to_pdf_converter.py data/cs_1-100_glasgow --batch-size 3 --max-width 1500 --max-height 1500


python analysis/scripts/image_combiner.py --help
usage: image_combiner.py [-h] -o OUTPUT [-c {1,2,3}] [-t TITLE] [-s SPACING] [-m MARGIN]
                         [-f {png,pdf}] [--matplotlib] [--dpi DPI]
                         inputs [inputs ...]

专业图片拼接工具 - 支持多种布局和输出格式

positional arguments:
  inputs                输入图片或文件夹路径（支持多个）

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        输出文件路径（.png或.pdf）
  -c {1,2,3}, --columns {1,2,3}
                        列数布局（1-3列，默认自动判断）
  -t TITLE, --title TITLE
                        图表标题
  -s SPACING, --spacing SPACING
                        图片间距（像素，默认20）
  -m MARGIN, --margin MARGIN
                        边距（像素，默认40）
  -f {png,pdf}, --format {png,pdf}
                        输出格式（默认根据文件扩展名判断）
  --matplotlib          使用matplotlib生成学术级图表
  --dpi DPI             输出分辨率（DPI，默认300）

使用示例:
  
  # 将文件夹内所有PNG拼接为PDF（自动布局）
  python image_combiner.py /path/to/images/ -o combined.pdf
  
  # 指定多个单独图片，双列布局，添加字母编号
  python image_combiner.py img1.png img2.png img3.png img4.png -o result.png -c 2 --labels
  
  # 混合输入：文件夹+单个文件，三列布局，添加标题和编号
  python image_combiner.py /folder1/ /folder2/ single.png -o output.pdf -c 3 -t "研究结果对比" --labels
  
  python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_004_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_051_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_063_scatter_with_map.png    python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_075_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_085_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_093_scatter_with_map.png  -o charts/Comparison_Chart_of_Typical_Layout_Patterns.pdf -c 3 --labels



  # 典型布局模式对比图
  python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_004_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_051_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_063_scatter_with_map.png  /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_093_scatter_with_map.png  \
  -o "final_version/Comparison_of_typical_layout_patterns.png" \
  -c 2 --labels  \
  --label-position bottom

  # 使用matplotlib生成学术级图表（带编号）
  python image_combiner.py images/ -o academic.pdf --matplotlib -t "Experimental Results" --labels
  
  # 无标题、无编号的简洁拼接
  python image_combiner.py *.png -o simple.png -c 2

