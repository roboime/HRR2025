#!/bin/bash

# Build ROS2 com suporte a CUDA
colcon build \
  --cmake-args \
    -DCMAKE_CUDA_ARCHITECTURES=53 \
    -DCMAKE_BUILD_TYPE=Release