#!/bin/bash

# Variáveis de ambiente críticas para acesso a hardware no container
export NVARGUS_SOCKET=/tmp/argus_socket
export NVIDIA_DRIVER_CAPABILITIES=all,compute,utility,video,graphics
export NVIDIA_VISIBLE_DEVICES=all

# Garantir existência e permissão do socket Argus
mkdir -p /tmp/argus_socket 2>/dev/null
chmod 777 /tmp/argus_socket 2>/dev/null

# Tentar limpar processos GStreamer antigos (boa prática)
pkill -f "nvarguscamerasrc" 2>/dev/null || true
pkill -f "gst-launch-1.0.*nvarguscamerasrc" 2>/dev/null || true
sleep 0.5 # Pequena pausa após pkill

# Caminho para o script Python
SCRIPT_PATH="/ros2_ws/src/perception/perception/jetson_camera/jetson_camera_node.py"

# Executar o script Python diretamente, passando argumentos
# A verificação de existência do script pode ser feita pelo chamador ou tratada no erro do python3
python3 "$SCRIPT_PATH" "$@" 