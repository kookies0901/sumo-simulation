  # Fig 5
  python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/models/plots_simple/cluster_count_waiting_time_mean.png  /home/ubuntu/project/MSC/Msc_Project/models/plots_simple/cluster_count_waiting_time_p90.png -o "final_version/Impact_of_multi-centredness_experience.png" -c 2 --labels   --label-position bottom  --h-spacing 60 --v-spacing 200  --label-font-size 120

  # Fig 6
   python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/models/plots_simple/std_pairwise_distance_vehicle_gini.png  /home/ubuntu/project/MSC/Msc_Project/models/plots_simple/std_pairwise_distance_charging_station_coverage.png  /home/ubuntu/project/MSC/Msc_Project/models/plots_simple/cluster_count_vehicle_gini.png /home/ubuntu/project/MSC/Msc_Project/models/plots_simple/cluster_count_charging_station_coverage.png -o "final_version/Impact_of_fairness_and_asset_utilization.png" -c 2 --labels   --label-position bottom  --h-spacing 60 --v-spacing 200  --label-font-size 120


   # Appendix D: Maps of Typical Layout Scenarios
   python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_010_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_020_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_030_scatter_with_map.png  \
  -o "final_version/Random_Uniform_Layout_3.png" \
  -c 3 --labels   --label-position bottom  --h-spacing 60 --v-spacing 200  --label-font-size 120

     python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_053_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_055_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_057_scatter_with_map.png  \
  -o "final_version/Centre_Clustered_Layout_3.png" \
  -c 3 --labels   --label-position bottom  --h-spacing 60 --v-spacing 200  --label-font-size 120

     python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_065_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_067_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_069_scatter_with_map.png  \
  -o "final_version/Peripheral_Dispersed_Layout_3.png" \
  -c 3 --labels   --label-position bottom  --h-spacing 60 --v-spacing 200  --label-font-size 120


     python scripts/image_combiner.py /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_091_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_094_scatter_with_map.png /home/ubuntu/project/MSC/Msc_Project/data/cs_1-100_glasgow/cs_group_096_scatter_with_map.png  \
  -o "final_version/Dense_Layout_3.png" \
  -c 3 --labels   --label-position bottom  --h-spacing 60 --v-spacing 200  --label-font-size 120