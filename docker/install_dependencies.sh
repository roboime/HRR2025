#!/bin/bash

# Definição de cores para melhorar a legibilidade
VERDE='\033[0;32m'
AZUL='\033[0;34m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
SEM_COR='\033[0m'

echo -e "${VERDE}Instalando dependências básicas do sistema...${SEM_COR}"
apt-get update && apt-get install -y build-essential cmake python3-dev libhdf5-dev \
    python3-opencv libopencv-dev python3-pip python3-setuptools \
    libhdf5-serial-dev hdf5-tools python3-h5py zlib1g-dev zip libjpeg8-dev \
    liblapack-dev libblas-dev gfortran pkg-config

echo -e "${VERDE}Atualizando pip e ferramentas essenciais...${SEM_COR}"
pip3 install --upgrade pip setuptools wheel

echo -e "${VERDE}Definindo codificação UTF-8 para evitar problemas com caracteres especiais...${SEM_COR}"
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Instalar dependências essenciais que precisam ser instaladas antes do requirements.txt
echo -e "${AZUL}Instalando dependências Python essenciais para Jetson...${SEM_COR}"
pip3 install cython
pip3 install pkgconfig
pip3 install testresources

# Verificando se o requirements.txt existe
if [ -f "/setup/requirements.txt" ]; then
    echo -e "${VERDE}Instalando dependências do requirements.txt...${SEM_COR}"
    # Removendo caracteres CR (^M) que podem causar problemas
    cat /setup/requirements.txt | tr -d "\r" > /setup/requirements_fixed.txt
    
    # Instalando NumPy primeiro, pois muitas outras dependências precisam dele
    echo -e "${AZUL}Instalando NumPy primeiro...${SEM_COR}"
    grep "^numpy==" /setup/requirements_fixed.txt | xargs pip3 install --no-deps || pip3 install --no-deps numpy==1.16.1
    
    # Instalando PyYAML, necessário para muitas bibliotecas
    echo -e "${AZUL}Instalando pyyaml...${SEM_COR}"
    grep "^pyyaml" /setup/requirements_fixed.txt | xargs pip3 install || pip3 install pyyaml>=5.1,<6.0
    
    # Instalando dependências científicas que precisam ser controladas
    echo -e "${AZUL}Instalando scipy...${SEM_COR}"
    grep "^scipy==" /setup/requirements_fixed.txt | xargs pip3 install || pip3 install scipy==1.4.1
    
    echo -e "${AZUL}Instalando Pillow...${SEM_COR}"
    grep "^pillow" /setup/requirements_fixed.txt | xargs pip3 install || pip3 install pillow>=7.0.0,<8.0.0
    
    echo -e "${AZUL}Instalando matplotlib...${SEM_COR}"
    grep "^matplotlib==" /setup/requirements_fixed.txt | xargs pip3 install || pip3 install matplotlib==3.2.2

    # Instalando h5py antes do TensorFlow
    echo -e "${AZUL}Instalando h5py...${SEM_COR}"
    env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==2.10.0
    
    # Instalando TensorFlow otimizado para Jetson
    echo -e "${AMARELO}Instalando TensorFlow otimizado para Jetson...${SEM_COR}"
    TF_VERSION=$(grep "^# tensorflow==" /setup/requirements_fixed.txt | sed 's/# //')
    if [ ! -z "$TF_VERSION" ]; then
        pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 $TF_VERSION
    else
        pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.5.0+nv21.8
    fi
    
    # Instalando scikit-image e outras dependências para Albumentations
    echo -e "${AZUL}Instalando dependências para Albumentations...${SEM_COR}"
    grep "^scikit-image==" /setup/requirements_fixed.txt | xargs pip3 install || pip3 install scikit-image==0.16.2
    pip3 install qudida==0.0.4

    # Instalando Albumentations
    echo -e "${AZUL}Instalando Albumentations...${SEM_COR}"
    grep "^albumentations==" /setup/requirements_fixed.txt | xargs pip3 install || pip3 install albumentations==0.4.6
    
    # Instalando protobuf com versão compatível
    echo -e "${AZUL}Instalando protobuf...${SEM_COR}"
    grep "^protobuf" /setup/requirements_fixed.txt | xargs pip3 install || pip3 install protobuf<=3.17.3
    
    # Instalando jetson-stats se disponível
    echo -e "${AZUL}Instalando jetson-stats...${SEM_COR}"
    grep "^jetson-stats" /setup/requirements_fixed.txt | xargs pip3 install || pip3 install jetson-stats>=3.1.0
    
    # Instalando as demais dependências do requirements.txt
    echo -e "${VERDE}Instalando as demais dependências do requirements.txt...${SEM_COR}"
    # Filtrar comentários e pacotes já instalados
    grep -v "^#" /setup/requirements_fixed.txt | grep -v "numpy==" | grep -v "scipy==" | grep -v "matplotlib==" | \
    grep -v "albumentations==" | grep -v "scikit-image==" | grep -v "protobuf" | grep -v "pillow" | \
    grep -v "pyyaml" | grep -v "jetson-stats" | xargs pip3 install || echo -e "${VERMELHO}Alguns pacotes não puderam ser instalados${SEM_COR}"
else
    echo -e "${VERMELHO}requirements.txt não encontrado. Usando valores padrão...${SEM_COR}"
    # Instala valores padrão para as dependências principais
    pip3 install --no-deps numpy==1.16.1
    pip3 install pyyaml>=5.1,<6.0
    pip3 install scipy==1.4.1
    pip3 install pillow>=7.0.0,<8.0.0
    pip3 install matplotlib==3.2.2
    env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==2.10.0
    pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.5.0+nv21.8
    pip3 install scikit-image==0.16.2
    pip3 install qudida==0.0.4
    pip3 install albumentations==0.4.6
    pip3 install protobuf<=3.17.3
    pip3 install jetson-stats>=3.1.0
fi

# Opção com suporte GUI do OpenCV
echo -e "${VERDE}Instalando suporte GUI para OpenCV...${SEM_COR}"
apt-get install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module

# Corrigindo o rosdep para ament_python (necessário para o roboime_behavior)
if [ ! -d "/etc/ros/rosdep/sources.list.d" ]; then
    mkdir -p /etc/ros/rosdep/sources.list.d
fi

echo -e "${VERDE}Configurando regras personalizadas para rosdep...${SEM_COR}"
cat > /etc/ros/rosdep/sources.list.d/custom-rules.yaml << EOF
ament_python:
  ubuntu: [python3-pip]
python3-albumentations:
  ubuntu: [python-is-python3]
EOF

# Atualizar o rosdep com as novas regras
echo -e "${VERDE}Atualizando rosdep...${SEM_COR}"
rosdep update || echo -e "${VERMELHO}rosdep update falhou, mas continuando mesmo assim${SEM_COR}"

echo -e "${AZUL}Verificando instalação de pacotes críticos...${SEM_COR}"
python3 -c "import tensorflow as tf; print(f\"${VERDE}TensorFlow versão: {tf.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}TensorFlow não instalado corretamente${SEM_COR}"
python3 -c "import cv2; print(f\"${VERDE}OpenCV versão: {cv2.__version__}${SEM_COR}\"); print(f\"${VERDE}Com suporte GUI: {hasattr(cv2, 'imshow')}${SEM_COR}\")" || echo -e "${VERMELHO}OpenCV não instalado corretamente${SEM_COR}"
python3 -c "import numpy; print(f\"${VERDE}NumPy versão: {numpy.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}NumPy não instalado corretamente${SEM_COR}"
python3 -c "import albumentations; print(f\"${VERDE}Albumentations versão: {albumentations.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}Albumentations não instalado corretamente${SEM_COR}"
python3 -c "import scipy; print(f\"${VERDE}SciPy versão: {scipy.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}SciPy não instalado corretamente${SEM_COR}"
python3 -c "import PIL; print(f\"${VERDE}Pillow versão: {PIL.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}Pillow não instalado corretamente${SEM_COR}"
python3 -c "import matplotlib; print(f\"${VERDE}Matplotlib versão: {matplotlib.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}Matplotlib não instalado corretamente${SEM_COR}"

echo -e "${VERDE}Instalação concluída!${SEM_COR}" 