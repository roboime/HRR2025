#!/bin/bash

# Sincronização segura via rsync
rsync -avzP \
  --exclude='.git/' \
  --exclude='build/' \
  -e "ssh -i ~/.ssh/jetson_key" \
  ./src/ \
  developer@jetson-ip:/ros2_ws/src