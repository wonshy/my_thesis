#!/bin/bash
#rm -rf data_splits/openlane/*



# python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 

#
#echo "========================================1-camera evaluate==========================================================="
#array=("curve_case"  "intersection_case"  "night_case"  "extreme_weather_case"  "merge_split_case"  "up_down_case")
#for element in ${array[@]}
#do
#    python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --test_case $element --evaluate
#done
#python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --evaluate --evaluate_fps
#
##evaluate flops fps
#python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --evaluate --evaluate_flops
#


#rm -rf .cache

python -m torch.distributed.launch  --nproc_per_node  4  main.py 


#echo "========================================5-camera evaluate==========================================================="
#
#array=("curve_case"  "intersection_case"  "night_case"  "extreme_weather_case"  "merge_split_case"  "up_down_case")
#for element in ${array[@]}
#do
#    python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --test_case $element --evaluate
#done
#




#python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --evaluate --evaluate_fps  --save_lines







# python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --test_case "curve_case" --evaluate --save_lines


# python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=5 --evaluate --evaluate_flops



#echo "========================================check multi-cam ==========================================================="
#
#for i in $(seq 1 1 4)
#do
#        python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=$i --evaluate 
#done




