#!/bin/bash

# Executar container com acesso total ao hardware
docker run -it --rm \
  --runtime nvidia \
  --privileged \
  -v /dev:/dev \
  --network host --ipc=host \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v $(pwd):/ros2_ws/src \
  --device /dev/video0 \
  soccerbot:jp4