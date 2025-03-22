#!/bin/bash

echo "Instalando dependências básicas do sistema..."
apt-get update && apt-get install -y build-essential cmake python3-dev libhdf5-dev \
    python3-opencv libopencv-dev python3-pip python3-setuptools

echo "Instalando scikit-build necessário para algumas dependências..."
pip3 install scikit-build wheel

echo "Definindo codificação UTF-8 para evitar problemas com caracteres especiais..."
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "Instalando dependências Python do requirements.txt..."
cat /setup/requirements.txt | tr -d "\r" > /setup/requirements_fixed.txt
pip3 install -r /setup/requirements_fixed.txt

echo "Verificando versão do Python..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Usando Python $PYTHON_VERSION"

echo "Instalando dependências específicas para ROS2..."
# Instalando dependências para Albumentations
apt-get update && apt-get install -y python3-pil
pip3 install scikit-image==0.17.2 scipy==1.5.4 qudida==0.0.4

# Instalando pacotes para suporte GUI do OpenCV
apt-get install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module

# Verificando se o OpenCV do sistema já está instalado e tem suporte GUI
python3 -c "import cv2; assert hasattr(cv2, 'imshow')" 2>/dev/null || {
    echo "Instalando OpenCV com suporte GUI completo..."
    pip3 install opencv-python
}

# Instalando Albumentations versão compatível com Python 3.6
pip3 install albumentations==0.5.2

# Corrigindo o rosdep para ament_python (necessário para o roboime_behavior)
if [ ! -d "/etc/ros/rosdep/sources.list.d" ]; then
    mkdir -p /etc/ros/rosdep/sources.list.d
fi

cat > /etc/ros/rosdep/sources.list.d/custom-rules.yaml << EOF
ament_python:
  ubuntu: [python3-pip]
python3-albumentations:
  ubuntu: [python-is-python3]
EOF

# Atualizar o rosdep com as novas regras
rosdep update || echo "rosdep update falhou, mas continuando mesmo assim"

# Instalação do TensorFlow específica para Jetson com verificação de compatibilidade
if [[ "$PYTHON_VERSION" == "3.6" ]]; then
    echo "Instalando TensorFlow compatível com Python 3.6 no Jetson..."
    
    # Instalar dependências específicas para TensorFlow
    apt-get update && apt-get install -y libhdf5-serial-dev hdf5-tools python3-h5py \
        zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    
    # Instalar Cython primeiro para evitar problemas com NumPy
    pip3 install cython
    
    # Instalar NumPy de forma segura
    pip3 install -U numpy==1.19.4
    
    # Instalar outras dependências do TensorFlow
    pip3 install -U future==0.18.2 mock==3.0.5 \
        keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 \
        protobuf pybind11 pkgconfig
    
    # Instalar h5py com flag especial para evitar problemas
    env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
    
    echo "Instalando TensorFlow otimizado para Jetson..."
    pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.6.2+nv21.12
    
    # Fallback para versão mais recente se a específica falhar
    if [ $? -ne 0 ]; then
        echo "Tentando versão alternativa do TensorFlow..."
        pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
    fi
else
    echo "Tentando instalar TensorFlow padrão, versão do Python é $PYTHON_VERSION..."
    pip3 install tensorflow
fi

echo "Verificando instalação de pacotes críticos..."
python3 -c "import tensorflow as tf; print(f\"TensorFlow versão: {tf.__version__}\")" || echo "TensorFlow não instalado corretamente"
python3 -c "import cv2; print(f\"OpenCV versão: {cv2.__version__}\"); print(f\"Com suporte GUI: {hasattr(cv2, 'imshow')}\")" || echo "OpenCV não instalado corretamente"
python3 -c "import numpy; print(f\"NumPy versão: {numpy.__version__}\")" || echo "NumPy não instalado corretamente"
python3 -c "import albumentations; print(f\"Albumentations versão: {albumentations.__version__}\")" || echo "Albumentations não instalado corretamente"

echo "Instalação concluída!" 