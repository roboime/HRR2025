#!/bin/bash
# Script para configurar a Jetson Nano para o sistema de visão da RoboIME

echo "Configurando a Jetson Nano para o sistema de visão da RoboIME..."

# Verificar se está rodando como root
if [ "$EUID" -ne 0 ]; then
  echo "Por favor, execute este script como root (sudo)."
  exit 1
fi

# Atualizar o sistema
echo "Atualizando o sistema..."
apt-get update
apt-get upgrade -y

# Instalar dependências
echo "Instalando dependências..."
apt-get install -y \
  python3-pip \
  python3-opencv \
  python3-numpy \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  libopenblas-dev \
  libjpeg-dev \
  zlib1g-dev

# Instalar dependências Python
echo "Instalando dependências Python..."
pip3 install --upgrade pip
pip3 install \
  numpy \
  opencv-python \
  matplotlib \
  pillow

# Verificar se o TensorRT está instalado
if dpkg -l | grep -q tensorrt; then
  echo "TensorRT já está instalado."
else
  echo "AVISO: TensorRT não encontrado. Ele deve ser instalado como parte do JetPack 4.6.1."
  echo "Por favor, certifique-se de que o JetPack 4.6.1 está instalado corretamente."
fi

# Verificar se o TensorFlow está instalado
if pip3 list | grep -q tensorflow; then
  echo "TensorFlow já está instalado."
else
  echo "Instalando TensorFlow 2.4.0 para Jetson..."
  pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow==2.4.0+nv21.5
fi

# Reiniciar o serviço da câmera
echo "Reiniciando o serviço da câmera..."
systemctl restart nvargus-daemon

# Criar diretório para modelos
echo "Criando diretório para modelos..."
mkdir -p /opt/roboime/models
chmod 777 /opt/roboime/models

# Configurar variáveis de ambiente
echo "Configurando variáveis de ambiente..."
cat > /etc/profile.d/roboime_vision.sh << 'EOF'
export ROBOIME_VISION_MODELS=/opt/roboime/models
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=0
EOF

# Otimizar a Jetson para desempenho
echo "Otimizando a Jetson para desempenho..."
cat > /etc/systemd/system/jetson_performance.service << 'EOF'
[Unit]
Description=Jetson Performance Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable jetson_performance.service

# Testar a câmera
echo "Testando a câmera CSI..."
timeout 5s gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! fakesink -v

echo "Configuração concluída!"
echo "Por favor, reinicie a Jetson Nano para aplicar todas as configurações." 