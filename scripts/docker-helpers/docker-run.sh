#!/bin/bash

# Script de Execução para Jetson Orin Nano Super Developer Kit
# JetPack 6.2 | ROS2 Humble | Ubuntu 22.04

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Verifica se já existe um container com o nome hrr
-cuda
CONTAINER_NAME="hrr-cuda"
CONTAINER_EXISTS=$(docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}")

# Opção: recriar container (usa: ./docker-run.sh --recreate)
RECREATE=0
for arg in "$@"; do
    if [ "$arg" = "--recreate" ] || [ "$arg" = "-r" ]; then
        RECREATE=1
    fi
done

if [ $RECREATE -eq 1 ] && [ "$CONTAINER_EXISTS" = "$CONTAINER_NAME" ]; then
    print_info "Recriando container existente: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true
    CONTAINER_EXISTS=""
fi

# Configuração específica para Jetson Orin Nano Super
JETSON_DEVICES=""
IS_JETSON=0
if [ -e "/dev/nvhost-ctrl" ] || [ -e "/dev/nvhost-gpu" ] || [ -e "/dev/nvmap" ] || \
   ( [ -f "/proc/device-tree/model" ] && grep -qi "jetson" /proc/device-tree/model ); then
    print_info "Detectando dispositivos Jetson Orin..."
    JETSON_DEVICES="--device /dev/nvhost-ctrl --device /dev/nvhost-ctrl-gpu --device /dev/nvhost-prof-gpu --device /dev/nvmap --device /dev/nvhost-gpu --device /dev/nvhost-as-gpu --device /dev/nvhost-vic --device /dev/tegra_dc_ctrl"
    IS_JETSON=1
    print_success "Dispositivos Jetson Orin configurados"
else
    print_info "Rodando em modo de desenvolvimento (não é uma Jetson)"
fi

if [ "$CONTAINER_EXISTS" = "$CONTAINER_NAME" ]; then
    # Se o container já existe mas não está rodando, inicia novamente
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" = "" ]; then
        print_info "Reiniciando container existente: $CONTAINER_NAME"
        docker start -ai $CONTAINER_NAME
    else
        # Se já está rodando, apenas entra no container
        print_success "Conectando ao container em execução: $CONTAINER_NAME"
        docker exec -it $CONTAINER_NAME bash
    fi
else
    # Verificar se a imagem existe com formato confiável
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^hrr:cuda$'; then
        print_error "Imagem hrr:cuda não encontrada!"
        print_info "Execute primeiro: ./scripts/docker-helpers/docker-build.sh"
        exit 1
    fi
    
    # Valida runtime NVIDIA (obrigatório somente na Jetson)
    HAS_NVIDIA_RUNTIME=0
    if docker info 2>/dev/null | grep -iq "Runtimes:.*nvidia"; then HAS_NVIDIA_RUNTIME=1; fi
    if command -v nvidia-container-runtime >/dev/null 2>&1; then HAS_NVIDIA_RUNTIME=1; fi
    if [ $IS_JETSON -eq 1 ] && [ $HAS_NVIDIA_RUNTIME -eq 0 ]; then
        print_error "Runtime NVIDIA ausente no Docker."
        print_info "Execute no host: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        exit 1
    fi

    # Args de GPU/Runtime (condicionais)
    RUNTIME_ARGS=""
    if [ $HAS_NVIDIA_RUNTIME -eq 1 ]; then
        RUNTIME_ARGS="--runtime nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all,compute,video,utility,graphics -e CUDA_VISIBLE_DEVICES=0"
    fi

    # Dispositivos e montagens dinâmicas
    VIDEO_DEVICES=""
    for dev in /dev/video*; do
        [ -e "$dev" ] && VIDEO_DEVICES="$VIDEO_DEVICES --device $dev"
    done
    ARGUS_MOUNT=""
    [ -e "/tmp/argus_socket" ] && ARGUS_MOUNT="-v /tmp/argus_socket:/tmp/argus_socket"
    CUDA_MOUNT=""
    [ -d "/usr/local/cuda-12.2" ] && CUDA_MOUNT="-v /usr/local/cuda-12.2:/usr/local/cuda-12.2"

    # Cria um novo container com nome específico
    print_info "Criando novo container: $CONTAINER_NAME"
    print_info "Imagem: hrr:cuda (JetPack 6.2 + ROS2 Humble)"
    
    # Montagem Xauthority apenas se existir no host, para evitar erro de bind
    XAUTH_MOUNT=""
    [ -f "$HOME/.Xauthority" ] && XAUTH_MOUNT="-v $HOME/.Xauthority:/root/.Xauthority:ro --env=\"XAUTHORITY=/root/.Xauthority\""

    # Montar tegrastats do host (se existir)
    TEGRASTATS_MOUNT=""
    [ -x "/usr/bin/tegrastats" ] && TEGRASTATS_MOUNT="-v /usr/bin/tegrastats:/usr/bin/tegrastats:ro"

    docker run -it \
      --name $CONTAINER_NAME \
      $RUNTIME_ARGS \
      --privileged \
      --entrypoint /entrypoint.sh \
      --env="DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      -v /dev:/dev \
      --network host --ipc=host \
      -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      $XAUTH_MOUNT \
      $TEGRASTATS_MOUNT \
      -v $(pwd):/ros2_ws \
      $ARGUS_MOUNT \
      $CUDA_MOUNT \
      $VIDEO_DEVICES \
      $JETSON_DEVICES \
      --workdir /ros2_ws \
      hrr:cuda \
      /bin/bash -i
fi