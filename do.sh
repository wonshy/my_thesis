#!/bin/bash
#rm -rf data_splits/openlane/*
#python -m torch.distributed.launch  --nproc_per_node  3  main.py --camera_nums=5 
#python -m torch.distributed.launch  --nproc_per_node  3  main.py --camera_nums=5 --evaluate

python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1 --evaluate
#python -m torch.distributed.launch  --nproc_per_node  1  main.py --camera_nums=1
