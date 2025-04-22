#!/bin/bash

# Variáveis de ambiente essenciais
export DISPLAY=${DISPLAY:-:0}
export GST_GL_API=gles2
export GST_GL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export NVARGUS_SOCKET=/tmp/argus_socket
export NVIDIA_DRIVER_CAPABILITIES=all,compute,utility,video,graphics
export NVIDIA_VISIBLE_DEVICES=all

# Configurar socket Argus
mkdir -p /tmp/argus_socket 2>/dev/null
chmod 777 /tmp/argus_socket 2>/dev/null

# Liberar recursos
pkill -f "nvarguscamerasrc" 2>/dev/null || true
pkill -f "gst-launch-1.0.*nvarguscamerasrc" 2>/dev/null || true
sleep 1

# Caminho para o script Python
SCRIPT_PATH="/ros2_ws/src/perception/perception/jetson_camera/jetson_camera_node.py"

# Verificar script e executar
if [ -f "$SCRIPT_PATH" ]; then
    python3 "$SCRIPT_PATH" "$@"
else
    echo "ERRO: Script não encontrado: $SCRIPT_PATH"
    exit 1
fi 