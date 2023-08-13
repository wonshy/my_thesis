#!/bin/bash
#rm -rf data_splits/openlane/*
#rm -rf .cache




# # python -m torch.distributed.launch  --nproc_per_node  1  main.py 

# #
# echo "========================================1-camera evaluate==========================================================="
# array=("curve_case"  "intersection_case"  "night_case"  "extreme_weather_case"  "merge_split_case"  "up_down_case")
# for element in ${array[@]}
# do
#    python -m torch.distributed.launch  --nproc_per_node  1  main.py  --test_case $element --evaluate
# done
# python -m torch.distributed.launch  --nproc_per_node  1  main.py  --evaluate --evaluate_fps  --save_lines

# ##evaluate flops fps
python -m torch.distributed.launch  --nproc_per_node  1  main.py  --evaluate --evaluate_flops




