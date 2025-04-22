#!/bin/bash

# Script wrapper para executar o nó da câmera Jetson em ambiente containerizado
# Este script configura o ambiente necessário para acessar a câmera CSI dentro do container

# Exibir informações de debug mais detalhadas
set -x

# Variáveis de ambiente para UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Variáveis de ambiente para GStreamer e CUDA
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=${DISPLAY:-:0}
export GST_DEBUG=4
export GST_DEBUG_FILE=/tmp/gstreamer_debug.log
export GST_GL_API=gles2
export GST_GL_PLATFORM=egl
export GST_DEBUG_NO_COLOR=1

# Verificar ambiente containerizado
echo "Verificando ambiente containerizado..."
if [ ! -f /.dockerenv ] && [ ! -f /run/.containerenv ]; then
    echo "AVISO: Este script foi otimizado para ambiente containerizado, mas não parece estar executando em um container"
fi

# Variáveis específicas para ambiente containerizado
export NVIDIA_DRIVER_CAPABILITIES=all,compute,utility,video,graphics
export NVIDIA_VISIBLE_DEVICES=all
export __GL_SYNC_TO_VBLANK=0

# Configuração do socket Argus para nvarguscamerasrc
export NVARGUS_SOCKET=/tmp/argus_socket

# Configurar diretório do socket Argus
echo "Configurando diretório do socket Argus..."
mkdir -p /tmp/argus_socket 2>/dev/null || true
chmod 777 /tmp/argus_socket 2>/dev/null || true

# Verificar volume compartilhado entre host e container
if [ ! -d "/tmp/.X11-unix" ]; then
    echo "AVISO: Pasta /tmp/.X11-unix não encontrada - pode haver problemas com interface gráfica"
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

# Executar o script Python com variáveis de ambiente otimizadas para container
echo "Iniciando nó da câmera em ambiente containerizado..."
python3 "$SCRIPT_PATH" "$@" 