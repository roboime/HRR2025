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

# Instalar dependências do sistema para matplotlib e outros pacotes
echo -e "${VERDE}Instalando dependências adicionais do sistema...${SEM_COR}"
apt-get install -y libfreetype6-dev libpng-dev pkg-config \
    libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module \
    python3-matplotlib python3-pil.imagetk python3-tk \
    python3-numpy python3-scipy python3-h5py

echo -e "${VERDE}Atualizando pip e ferramentas essenciais...${SEM_COR}"
python3 -m pip install --upgrade pip setuptools wheel

echo -e "${VERDE}Definindo codificação UTF-8 para evitar problemas com caracteres especiais...${SEM_COR}"
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Instalar dependências essenciais básicas
echo -e "${AZUL}Instalando dependências Python essenciais...${SEM_COR}"
python3 -m pip install cython
python3 -m pip install pkgconfig
python3 -m pip install testresources

# Verificar se as bibliotecas do sistema estão funcionando corretamente
echo -e "${AZUL}Verificando bibliotecas do sistema...${SEM_COR}"

# Verificar e corrigir NumPy se necessário
echo -e "${AZUL}Verificando NumPy...${SEM_COR}"
if ! python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null; then
    echo -e "${AMARELO}NumPy do sistema não está funcionando. Instalando versão alternativa...${SEM_COR}"
    apt-get remove -y python3-numpy
    python3 -m pip install --no-deps numpy==1.16.1
fi

# Verificar e corrigir SciPy se necessário
echo -e "${AZUL}Verificando SciPy...${SEM_COR}"
if ! python3 -c "import scipy; print(scipy.__version__)" 2>/dev/null; then
    echo -e "${AMARELO}SciPy do sistema não está funcionando. Instalando versão alternativa...${SEM_COR}"
    apt-get remove -y python3-scipy
    python3 -m pip install scipy==1.4.1
fi

# Verificar e corrigir Matplotlib se necessário
echo -e "${AZUL}Verificando Matplotlib...${SEM_COR}"
if ! python3 -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null; then
    echo -e "${AMARELO}Matplotlib do sistema não está funcionando. Instalando versão alternativa...${SEM_COR}"
    apt-get remove -y python3-matplotlib
    # Instalar dependências do Matplotlib
    python3 -m pip install kiwisolver==1.3.1 cycler==0.10.0 pyparsing==2.4.7 python-dateutil==2.8.1
    python3 -m pip install --no-build-isolation matplotlib==3.2.2
fi

# Verificar e corrigir h5py se necessário
echo -e "${AZUL}Verificando h5py...${SEM_COR}"
if ! python3 -c "import h5py; print(h5py.__version__)" 2>/dev/null; then
    echo -e "${AMARELO}h5py do sistema não está funcionando. Instalando versão alternativa...${SEM_COR}"
    apt-get remove -y python3-h5py
    H5PY_SETUP_REQUIRES=0 python3 -m pip install --no-build-isolation h5py==2.10.0
fi

# Verificar e corrigir OpenCV se necessário
echo -e "${AZUL}Verificando OpenCV...${SEM_COR}"
if ! python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null; then
    echo -e "${AMARELO}OpenCV do sistema não está funcionando. Instalando versão alternativa...${SEM_COR}"
    # Tentar instalar via pip se o pacote do sistema não estiver funcionando
    python3 -m pip install opencv-python==4.1.2.30
fi

# Instalando TensorFlow otimizado para Jetson
echo -e "${AMARELO}Instalando TensorFlow otimizado para Jetson...${SEM_COR}"
# Remover versões existentes que possam estar conflitando
python3 -m pip uninstall -y tensorflow tensorflow-gpu

# Tentar a instalação específica para Jetson
echo -e "${AZUL}Tentando instalar TensorFlow específico para Jetson...${SEM_COR}"
python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.5.0+nv21.8 || {
    echo -e "${AMARELO}Falha na instalação do TensorFlow para Jetson. Tentando alternativa...${SEM_COR}"
    python3 -m pip install tensorflow==2.4.0
}

# Instalando protobuf com versão compatível (necessário para TensorFlow)
echo -e "${AZUL}Instalando protobuf...${SEM_COR}"
python3 -m pip install protobuf<=3.17.3

# Instalando PyYAML (comum em configurações)
echo -e "${AZUL}Instalando pyyaml...${SEM_COR}"
python3 -m pip install --ignore-installed --no-warn-script-location pyyaml>=5.1,\<6.0

# Instalando Pillow (processamento de imagens)
echo -e "${AZUL}Instalando Pillow...${SEM_COR}"
python3 -m pip install --ignore-installed --no-warn-script-location pillow>=7.0.0,\<8.0.0

# Instalando jetson-stats
echo -e "${AZUL}Instalando jetson-stats...${SEM_COR}"
python3 -m pip install jetson-stats>=3.1.0

# Corrigindo o rosdep para ament_python (necessário para o roboime_behavior)
if [ ! -d "/etc/ros/rosdep/sources.list.d" ]; then
    mkdir -p /etc/ros/rosdep/sources.list.d
fi

echo -e "${VERDE}Configurando regras personalizadas para rosdep...${SEM_COR}"
cat > /etc/ros/rosdep/sources.list.d/custom-rules.yaml << EOF
ament_python:
  ubuntu: [python3-pip]
EOF

# Atualizar o rosdep com as novas regras
echo -e "${VERDE}Atualizando rosdep...${SEM_COR}"
rosdep update || echo -e "${VERMELHO}rosdep update falhou, mas continuando mesmo assim${SEM_COR}"

echo -e "${AZUL}Verificando instalação de pacotes críticos...${SEM_COR}"
python3 -c "import tensorflow as tf; print(f\"${VERDE}TensorFlow versão: {tf.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}TensorFlow não instalado corretamente${SEM_COR}"
python3 -c "import cv2; print(f\"${VERDE}OpenCV versão: {cv2.__version__}${SEM_COR}\"); print(f\"${VERDE}Com suporte GUI: {hasattr(cv2, 'imshow')}${SEM_COR}\")" || echo -e "${VERMELHO}OpenCV não instalado corretamente${SEM_COR}"
python3 -c "import numpy; print(f\"${VERDE}NumPy versão: {numpy.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}NumPy não instalado corretamente${SEM_COR}"
python3 -c "import scipy; print(f\"${VERDE}SciPy versão: {scipy.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}SciPy não instalado corretamente${SEM_COR}"
python3 -c "import PIL; print(f\"${VERDE}Pillow versão: {PIL.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}Pillow não instalado corretamente${SEM_COR}"
python3 -c "import matplotlib; print(f\"${VERDE}Matplotlib versão: {matplotlib.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}Matplotlib não instalado corretamente${SEM_COR}"
python3 -c "import h5py; print(f\"${VERDE}h5py versão: {h5py.__version__}${SEM_COR}\")" || echo -e "${VERMELHO}h5py não instalado corretamente${SEM_COR}"

echo -e "${VERDE}Instalação concluída!${SEM_COR}" 