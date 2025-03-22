#!/bin/bash

# Verifica se já existe um container com o nome hsl-container
CONTAINER_EXISTS=$(docker ps -a --filter "name=hsl-container" --format "{{.Names}}")

if [ "$CONTAINER_EXISTS" = "hsl-container" ]; then
    # Se o container já existe mas não está rodando, inicia novamente
    if [ "$(docker ps -q -f name=hsl-container)" = "" ]; then
        echo "Reiniciando container existente..."
        docker start -i hsl-container
    else
        # Se já está rodando, apenas entra no container
        echo "Conectando ao container em execução..."
        docker exec -it hsl-container bash
    fi
else
    # Cria um novo container com nome específico
    echo "Criando novo container..."
    docker run -it \
      --name hsl-container \
      --runtime nvidia \
      --privileged \
      -v /dev:/dev \
      --network host --ipc=host \
      -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v /tmp/argus_socket:/tmp/argus_socket \
      -v $(pwd):/ros2_ws \
      --device /dev/video0 \
      hsl:latest
fi