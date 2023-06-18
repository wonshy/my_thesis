#!/bin/bash
#rm -rf data_splits/openlane/*
#python -m torch.distributed.launch  --nproc_per_node  3  main.py --camera_nums=5 --evaluate













array=("curve_case"  "intersection_case"  "night_case"  "extreme_weather_case"  "merge_split_case"  "up_down_case")
for element in ${array[@]}
do
    time python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --test_case $element 
done
echo "==================================================================================================="
time python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --evaluate









# array=("curve_case"  "intersection_case"  "night_case"  "extreme_weather_case"  "merge_split_case"  "up_down_case")
# for element in ${array[@]}
# do
#     time python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --test_case $element 
# done

# echo "==================================================================================================="
# time python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --evaluate