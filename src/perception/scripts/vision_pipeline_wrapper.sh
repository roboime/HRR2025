#!/bin/bash

# Script wrapper para executar o nó vision_pipeline diretamente
# Este script resolve o problema de executáveis não encontrados pelo ROS2 Eloquent

# Variáveis de ambiente para UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Adicionar o diretório de bibliotecas Python ao PYTHONPATH
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Caminho para o script Python
SCRIPT_PATH="/ros2_ws/src/perception/perception/vision_pipeline.py"

# Verificar se o script existe
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERRO: Script não encontrado: $SCRIPT_PATH"
    exit 1
fi

# Executar o script Python diretamente
python3 "$SCRIPT_PATH" "$@" 