#!/bin/bash
set -e

# Configurar variáveis de ambiente ROS2
source /opt/ros/eloquent/setup.bash

# Configurar o ambiente CUDA e TensorRT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64:/usr/lib/aarch64-linux-gnu
export PATH=$PATH:/usr/local/cuda-10.2/bin

# Se existir um workspace ROS, configurar
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

# Exibir informações do ambiente
echo "==== Configuração do ROS2 Eloquent para Câmera IMX219 ===="
echo "Sistema: $(uname -a)"
echo "ROS Distro: $ROS_DISTRO"
echo "Python: $(python3 --version)"
echo "OpenCV: $(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Não disponível")"
echo "======================================================"

# Verificar dispositivos de câmera disponíveis
echo "Dispositivos de câmera disponíveis:"
ls -la /dev/video* 2>/dev/null || echo "Nenhum dispositivo de câmera encontrado"

# Executar o comando fornecido ou bash por padrão
exec "$@" 