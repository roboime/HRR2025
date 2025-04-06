#!/bin/bash

# Script para testar diretamente o nó da câmera e monitorar os tópicos publicados

# Cores para mensagens
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Funções
function print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_header "Teste direto da câmera CSI e monitoramento de tópicos"

# Verificar dispositivos de vídeo
if ls -l /dev/video* &> /dev/null; then
    print_success "Dispositivos de vídeo encontrados:"
    ls -l /dev/video*
else
    print_error "Nenhum dispositivo de vídeo encontrado!"
    exit 1
fi

# Caminho para o nó da câmera
CAMERA_SCRIPT="/ros2_ws/src/perception/perception/jetson_camera/jetson_camera_node.py"

# Verificar se o script existe
if [ ! -f "$CAMERA_SCRIPT" ]; then
    print_error "Script da câmera não encontrado: $CAMERA_SCRIPT"
    exit 1
fi

# Dar permissão de execução
chmod +x "$CAMERA_SCRIPT"

# Instalar ferramenta de diagnóstico para ROS2
apt-get update && apt-get install -y python3-pip
pip3 install setuptools
pip3 install ros2topic ros2node ros2service

# Adicionar o diretório de bibliotecas Python ao PYTHONPATH
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Funções para monitorar tópicos em background
function monitor_topics() {
    while true; do
        echo -e "\n${YELLOW}=== Tópicos ativos: ===${NC}"
        ros2 topic list
        echo -e "${YELLOW}======================${NC}\n"
        sleep 5
    done
}

# Iniciar monitoramento em segundo plano
monitor_topics &
MONITOR_PID=$!

# Limpar ao finalizar
function cleanup() {
    kill $MONITOR_PID
    echo -e "\n${RED}Teste finalizado${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

print_info "Executando nó da câmera diretamente..."
print_info "Pressione Ctrl+C para encerrar"

# Executar nó da câmera diretamente
python3 "$CAMERA_SCRIPT" --ros-args \
  -p camera_mode:=2 \
  -p camera_fps:=120 \
  -p flip_method:=0 \
  -p exposure_time:=13333 \
  -p gain:=1.0 \
  -p awb_mode:=1 \
  -p brightness:=0 \
  -p saturation:=1.0 \
  -p enable_cuda:=true \
  -p enable_hdr:=false \
  -p enable_display:=true 