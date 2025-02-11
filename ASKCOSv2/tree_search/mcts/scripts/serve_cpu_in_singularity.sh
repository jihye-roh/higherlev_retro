#!/bin/bash

singularity instance start mcts_cpu.sif mcts
nohup \
singularity exec instance://mcts \
  python mcts_server.py \
&>/dev/null &
