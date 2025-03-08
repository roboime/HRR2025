#!/bin/bash
set -e

# Configuração do ambiente ROS2
source /opt/ros/eloquent/setup.bash
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

# Executar comando passado como argumento
exec "$@" 