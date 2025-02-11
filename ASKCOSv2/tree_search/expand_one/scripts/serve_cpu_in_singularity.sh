#!/bin/bash

singularity instance start expand_one_cpu.sif expand_one
nohup \
singularity exec instance://expand_one \
  python expand_one_server.py \
&>/dev/null &
