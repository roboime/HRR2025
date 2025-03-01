#!/bin/bash

# Executar container com acesso total ao hardware
docker run -it --rm \
  --runtime nvidia \
  --privileged \
  -v /dev:/dev \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v $(pwd):/ros2_ws/src \
  soccerbot:jp4