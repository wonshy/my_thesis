#!/bin/bash
#rm -rf data_splits/openlane/*
#python -m torch.distributed.launch  --nproc_per_node  3  main.py --camera_nums=5 --evaluate











# python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 


#echo "========================================1-camera evaluate==========================================================="
# array=("curve_case"  "intersection_case"  "night_case"  "extreme_weather_case"  "merge_split_case"  "up_down_case")
# for element in ${array[@]}
# do
#     python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --test_case $element   --evaluate_fps
# done
# python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --evaluate --evaluate_fps

# evaluate flops fps
# python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --evaluate --evaluate_flops




python -m torch.distributed.launch  --nproc_per_node  4  main.py --camera_nums=5 


#echo "========================================5-camera evaluate==========================================================="

# array=("curve_case"  "intersection_case"  "night_case"  "extreme_weather_case"  "merge_split_case"  "up_down_case")
# for element in ${array[@]}
# do
#     python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --test_case $element --evaluate_fps
# done

#python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --evaluate --evaluate_fps
# python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --evaluate --evaluate_flops
