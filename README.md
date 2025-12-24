# Intelligent EV Charging Layout Optimization via SUMO Simulation and ML

This project aims to **optimize electric vehicle (EV) charging station layouts** using **realistic traffic simulations** and **machine learning (ML)** models. It is built upon **SUMO (Simulation of Urban Mobility)** and targets Glasgow’s real road network.

---

## Project Goal

> Build a digital pipeline that simulates EV usage in real traffic, evaluates charging station layouts, and trains an ML model to **predict the effectiveness of future layouts** — enabling faster iteration without rerunning SUMO each time.

---

## Overall Workflow

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

## File Structure
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

## Output Dataset Format (for ML)

After simulation + feature extraction, the ML-ready training dataset will include:

| scenario_id | cs_layout_id | num_cs | layout_features (e.g. coords, spread) | avg_wait_time | avg_charge_time |
| ------------ | -------------- | ------- | -------------------------------------- | --------------- | ----------------- |
| S001         | cs_group_001 | 13      | [x1,y1,x2,y2,...]                     | 142.5 s         | 370.2 s           |

---

## Potential Use Cases

* Predict best-performing charging layouts **without rerunning SUMO**
* Use ML model to explore large layout design space
* Fine-tune layout strategies for large-scale deployment cities

---


## Requirements

* Python ≥ 3.10
* SUMO (with TraCI)
* pandas, numpy, scikit-learn
* Optional: torch, xgboost for modeling

---

## Author

**Tingting Yang**, MSc Computing Science (University of Glasgow)
Special focus: intelligent systems, mobility simulation, AI companionship research

For contributions, issues or guidance, feel free to contact or open a pull request.

---


