#!/bin/bash
set -e

# Configuração do ambiente ROS2
source /opt/ros/$ROS_DISTRO/setup.bash

# Configuração do workspace, se existir
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

# Garante que as variáveis de ambiente necessárias estejam definidas
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONIOENCODING=utf-8

# Exibe informações úteis
echo "=================================================="
echo "Container ROS2 Eloquent para Jetson inicializado!"
echo "=================================================="
echo "ROS_DISTRO: $ROS_DISTRO"
echo "ROS_PYTHON_VERSION: $ROS_PYTHON_VERSION"
echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "================================================"
echo "Pacotes ROS2 disponíveis:"
ros2 pkg list | grep -E "roboime|perception" || echo "Nenhum pacote personalizado encontrado"
echo "================================================"
echo "Dispositivos de câmera disponíveis:"
ls -la /dev/video* 2>/dev/null || echo "Nenhum dispositivo de câmera encontrado"
echo "=================================================="
echo "Para instalar as dependências, execute:"
echo "  ./setup/install_dependencies.sh"
echo "=================================================="

# Executa o comando passado como argumento ou inicia um shell
exec "$@" || exec /bin/bash
