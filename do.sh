#!/bin/bash
rm -rf data_splits/openlane/*
python -m torch.distributed.launch main.py --local_rank=0
