#!/bin/bash

# Verifica se já existe um container com o nome hsl-container
CONTAINER_EXISTS=$(docker ps -a --filter "name=hsl-docker" --format "{{.Names}}")

# Configuração para dispositivos Jetson
JETSON_DEVICES=""
if [ -d "/dev/nvhost-ctrl" ] || [ -d "/dev/nvhost-gpu" ] || [ -d "/dev/nvmap" ]; then
    echo "Configurando para dispositivos específicos da Jetson..."
    JETSON_DEVICES="--device /dev/nvhost-ctrl --device /dev/nvhost-ctrl-gpu --device /dev/nvhost-prof-gpu --device /dev/nvmap --device /dev/nvhost-gpu --device /dev/nvhost-as-gpu --device /dev/nvhost-vic"
fi

if [ "$CONTAINER_EXISTS" = "hsl-docker" ]; then
    # Se o container já existe mas não está rodando, inicia novamente
    if [ "$(docker ps -q -f name=hsl-docker)" = "" ]; then
        echo "Reiniciando container existente..."
        docker start -i hsl-docker
    else
        # Se já está rodando, apenas entra no container
        echo "Conectando ao container em execução..."
        docker exec -it hsl-docker bash
    fi
else
    # Cria um novo container com nome específico
    echo "Criando novo container..."
    docker run -it \
      --name hsl-docker \
      --runtime nvidia \
      --privileged \
      --gpus all \
      --env="DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=all,compute,video,utility,graphics \
      -v /dev:/dev \
      --network host --ipc=host \
      -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -v $HOME/.Xauthority:/root/.Xauthority:ro \
      --env="XAUTHORITY=/root/.Xauthority" \
      -v /tmp/argus_socket:/tmp/argus_socket \
      -v $(pwd):/ros2_ws \
      -v /usr/local/cuda:/usr/local/cuda \
      --device /dev/video0 \
      $JETSON_DEVICES \
      hsl:opencv \
      /bin/bash
fi