#!/bin/bash

# Script wrapper para executar o nó da câmera Jetson diretamente
# Este script resolve o problema de executáveis não encontrados pelo ROS2 Eloquent

# Variáveis de ambiente para UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Variáveis de ambiente para CUDA e Argus em containers
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=${DISPLAY:-:0}
export GST_DEBUG=3
export GST_GL_API=gles2
export GST_GL_PLATFORM=egl
export GST_DEBUG_NO_COLOR=1

# Variáveis para nvarguscamerasrc
export NVARGUS_SOCKET=/tmp/argus_socket
export __GL_SYNC_TO_VBLANK=0
export GST_ARGUS_SENSOR_MODE=6  # Modo 6 = 1280x720@120fps

# Verificar e criar socket Argus se necessário
if [ ! -d /tmp/argus_socket ]; then
    echo "Criando diretório para socket Argus em /tmp/argus_socket"
    mkdir -p /tmp/argus_socket
    chmod 777 /tmp/argus_socket
fi

# Adicionar o diretório de bibliotecas Python ao PYTHONPATH
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Caminho para o script Python
SCRIPT_PATH="/ros2_ws/src/perception/perception/jetson_camera/jetson_camera_node.py"

# Verificar se o script existe
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERRO: Script não encontrado: $SCRIPT_PATH"
    exit 1
fi

# Limpar recursos existentes
echo "Liberando recursos da câmera em uso..."
pkill -f "nvarguscamerasrc" || true
sleep 1

# Verificar status do serviço nvargus-daemon
if command -v systemctl &> /dev/null; then
    if ! systemctl is-active --quiet nvargus-daemon; then
        echo "Tentando iniciar serviço nvargus-daemon..."
        systemctl start nvargus-daemon || true
        sleep 2
    else
        echo "Serviço nvargus-daemon está em execução."
    fi
fi

# Executar o script Python diretamente
echo "Iniciando nó da câmera..."
python3 "$SCRIPT_PATH" "$@" 