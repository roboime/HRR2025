#!/bin/bash
# Script de instalação para o sistema de visão da RoboIME na Jetson Nano
# Este script instala todas as dependências necessárias para o sistema de visão

set -e  # Sair em caso de erro

echo "Iniciando instalação do sistema de visão da RoboIME para Jetson Nano..."
echo "Este script deve ser executado como usuário normal (não root)"

# Verificar se está rodando como root
if [ "$(id -u)" -eq 0 ]; then
    echo "ERRO: Este script não deve ser executado como root"
    exit 1
fi

# Verificar se está em uma Jetson Nano
if ! grep -q "NVIDIA Jetson Nano" /proc/device-tree/model 2>/dev/null; then
    echo "AVISO: Este script é destinado à NVIDIA Jetson Nano"
    echo "Continuando mesmo assim..."
fi

# Atualizar repositórios
echo "Atualizando repositórios..."
sudo apt-get update

# Instalar dependências do sistema
echo "Instalando dependências do sistema..."
sudo apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools

# Instalar dependências Python
echo "Instalando dependências Python..."
pip3 install --user numpy opencv-python

# Reiniciar o serviço nvargus-daemon (necessário para a câmera CSI)
echo "Reiniciando o serviço nvargus-daemon..."
sudo systemctl restart nvargus-daemon

# Verificar se a câmera CSI está funcionando
echo "Verificando se a câmera CSI está funcionando..."
echo "Pressione Ctrl+C para sair após alguns segundos se a câmera estiver funcionando"
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1' ! nvvidconv flip-method=0 ! 'video/x-raw, width=640, height=480, format=BGRx' ! videoconvert ! 'video/x-raw, format=BGR' ! videoconvert ! xvimagesink -e

# Criar diretório para logs
echo "Criando diretório para logs..."
mkdir -p ~/ros2_logs

# Adicionar variáveis de ambiente ao .bashrc
echo "Adicionando variáveis de ambiente ao .bashrc..."
if ! grep -q "ROS_DOMAIN_ID=30" ~/.bashrc; then
    echo "# Configurações ROS 2 para RoboIME Vision" >> ~/.bashrc
    echo "export ROS_DOMAIN_ID=30" >> ~/.bashrc
    echo "export ROS_LOG_DIR=~/ros2_logs" >> ~/.bashrc
    echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
fi

echo ""
echo "Instalação concluída!"
echo "Para compilar o pacote, execute:"
echo "cd ~/ros2_ws"
echo "colcon build --packages-select roboime_vision"
echo ""
echo "Para iniciar o sistema de visão, execute:"
echo "ros2 launch roboime_vision jetson_vision.launch.py"
echo ""
echo "Reinicie o terminal ou execute 'source ~/.bashrc' para aplicar as alterações" 