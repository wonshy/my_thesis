#!/bin/bash
rm -rf data_splits/openlane/*
python -m torch.distributed.launch  --nproc_per_node 4   main.py 
