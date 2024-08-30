#!/bin/bash

for i in $(seq 200  257); do
  GPUS=1  GPUS_PER_NODE=1 CPUS_PER_TASK=2 bash ./tools/slurm_test.sh Test test configs/vig/vig_t_multihead2.py work_dirs/vig_t_ft/epoch_${i}.pth --metrics accuracy --ne $((258 - i))
  wait
done