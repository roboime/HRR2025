#!/bin/bash

# Monitoramento de recursos da Jetson
watch -n 1 -d \
  "echo '=== GPU ==='; \
   sudo tegrastats | grep -oP 'GR3D.*?%' && \
   echo '\n=== CPU ==='; \
   mpstat -P ALL 1 1 | awk '/Average:/ && \$2 ~ /[0-9]/ {print \"CPU\" \$2 \" Usage: \" \$3 \"%\"}' && \
   echo '\n=== Memory ==='; \
   free -m | awk '/Mem:/ {printf \"Mem: %.1f%% Used (%.1fG total)\n\", \$3/\$2*100, \$2/1024}'"