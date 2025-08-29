#!/bin/bash

# Definição de cores para melhorar a legibilidade
VERDE='\033[0;32m'
AZUL='\033[0;34m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
SEM_COR='\033[0m'

echo -e "${VERDE}===============================================${SEM_COR}"
echo -e "${VERDE}  Instalação de Dependências - Jetson Orin Nano Super${SEM_COR}"
echo -e "${VERDE}  JetPack 6.2 | Ubuntu 22.04 | Python 3.10+${SEM_COR}"
echo -e "${VERDE}  ROS2 Humble | CUDA 12.2 | Versões Modernas${SEM_COR}"
echo -e "${VERDE}===============================================${SEM_COR}"

# Atualizar sistema base
echo -e "${VERDE}Atualizando sistema base...${SEM_COR}"
apt-get update && apt-get upgrade -y

echo -e "${VERDE}Instalando dependências básicas do sistema para Ubuntu 22.04...${SEM_COR}"
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    zlib1g-dev \
    zip \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    pkg-config \
    x11-apps

# Instalando dependências adicionais do sistema para Ubuntu 22.04
echo -e "${VERDE}Instalando dependências adicionais do sistema...${SEM_COR}"
apt-get install -y \
    libfreetype6-dev \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    python3-pil.imagetk \
    python3-tk \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    libopencv-dev \
    python3-opencv

echo -e "${VERDE}Atualizando pip e ferramentas essenciais...${SEM_COR}"
python3 -m pip install --upgrade pip setuptools wheel

echo -e "${VERDE}Definindo codificação UTF-8...${SEM_COR}"
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Função para verificar se o pacote já está instalado
check_package() {
    local package=$1
    local module=$2
    local min_version=$3
    
    echo -n "Verificando $package... "
    if python3 -c "import $module; from packaging import version; print(version.parse('$min_version') <= version.parse($module.__version__))" 2>/dev/null | grep -q "True"; then
        echo -e "${VERDE}✓ versão compatível instalada${SEM_COR}"
        return 0
    fi
    
    if python3 -c "import $module" 2>/dev/null; then
        echo -e "${AMARELO}⚠ versão incompatível, atualizando...${SEM_COR}"
        return 1
    fi
    
    echo -e "${AMARELO}✗ não encontrado, instalando...${SEM_COR}"
    return 2
}

# Instalação condicional
install_if_needed() {
    local package=$1
    local module=$2
    local min_version=$3
    local install_cmd=$4
    
    check_package "$package" "$module" "$min_version"
    local result=$?
    
    if [ $result -ne 0 ]; then
        echo -e "${AZUL}Instalando $package versão moderna...${SEM_COR}"
        eval $install_cmd
    fi
}

# Instalar packaging para verificação de versões
python3 -m pip install packaging

# === DEPENDÊNCIAS PYTHON MODERNAS ===
echo -e "${AZUL}Instalando dependências Python modernas (compatíveis com Python 3.10+)...${SEM_COR}"

# Dependências para Matplotlib modernas
echo -e "${AZUL}Instalando dependências para Matplotlib 3.7+...${SEM_COR}"
python3 -m pip install \
    kiwisolver==1.4.5 \
    cycler==0.12.1 \
    pyparsing==3.1.1 \
    python-dateutil==2.8.2 \
    fonttools==4.43.1

# NumPy e SciPy mais recentes compatíveis com Python 3.10+
install_if_needed "NumPy" "numpy" "1.24.0" "python3 -m pip install numpy==1.24.3"
install_if_needed "SciPy" "scipy" "1.11.0" "python3 -m pip install scipy==1.11.3"
install_if_needed "Matplotlib" "matplotlib" "3.7.0" "python3 -m pip install matplotlib==3.7.2"
install_if_needed "h5py" "h5py" "3.9.0" "python3 -m pip install h5py==3.9.0"
install_if_needed "Pillow" "PIL" "10.0.0" "python3 -m pip install pillow==10.0.0"
install_if_needed "PyYAML" "yaml" "6.0.0" "python3 -m pip install pyyaml==6.0.1"
install_if_needed "Protobuf" "google.protobuf" "4.24.0" "python3 -m pip install protobuf==4.24.4"

# Ferramentas modernas
python3 -m pip install \
    tqdm==4.66.1 \
    requests==2.31.0 \
    opencv-python==4.8.1.78

# === INSTALAR FRAMEWORKS DE ML MODERNOS PARA JETSON ORIN ===
echo -e "${AZUL}Instalando frameworks de ML modernos para Jetson Orin Nano Super...${SEM_COR}"

# TensorFlow para JetPack 6.2
echo -e "${VERDE}Instalando TensorFlow 2.15+ otimizado para JetPack 6.2...${SEM_COR}"
if ! python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null | grep -q "2.1[5-9]"; then
    echo -e "${AZUL}Instalando dependências modernas do TensorFlow...${SEM_COR}"
    
    # Dependências do TensorFlow modernas
    python3 -m pip install \
        keras-preprocessing==1.1.2 \
        gast==0.5.4 \
        six==1.16.0 \
        typing_extensions==4.8.0 \
        wrapt==1.15.0 \
        absl-py==2.0.0 \
        astunparse==1.6.3 \
        termcolor==2.3.0 \
        flatbuffers==23.5.26 \
        google-pasta==0.2.0 \
        opt-einsum==3.3.0 \
        ml-dtypes==0.2.0
    
    # TensorFlow otimizado para Jetson Orin com JetPack 6.2
    echo -e "${VERDE}Instalando TensorFlow para JetPack 6.2...${SEM_COR}"
    python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v62 \
        tensorflow==2.15.0+nv24.2 || {
        echo -e "${AMARELO}Tentando instalação alternativa do TensorFlow...${SEM_COR}"
        python3 -m pip install tensorflow==2.13.0 || {
            echo -e "${AMARELO}Instalando TensorFlow via pip padrão...${SEM_COR}"
            python3 -m pip install tensorflow==2.13.0
        }
    }
    
    # TensorBoard moderno
    python3 -m pip install tensorboard==2.15.0
else
    echo -e "${VERDE}TensorFlow moderno já instalado.${SEM_COR}"
fi

# PyTorch para Jetson Orin (alternativa moderna e mais eficiente)
echo -e "${VERDE}Instalando PyTorch 2.1+ para Jetson Orin...${SEM_COR}"
if ! python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.[1-9]"; then
    echo -e "${AZUL}Instalando PyTorch otimizado para Jetson Orin...${SEM_COR}"
    python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v62 \
        torch==2.1.0 \
        torchvision==0.16.0 \
        torchaudio==2.1.0 || {
        echo -e "${AMARELO}Instalando PyTorch via pip padrão...${SEM_COR}"
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
else
    echo -e "${VERDE}PyTorch moderno já instalado.${SEM_COR}"
fi

# === ULTRALYTICS YOLOv8 (SUBSTITUTO MODERNO DO YOLOv4-TINY) ===
echo -e "${AZUL}Instalando Ultralytics YOLOv8/v9 (substituto moderno do YOLOv4-Tiny)...${SEM_COR}"
python3 -m pip install ultralytics==8.0.196

echo -e "${VERDE}✓ YOLOv8 instalado! Muito mais rápido e preciso que YOLOv4-Tiny${SEM_COR}"

# === GSTREAMER 1.20+ PARA UBUNTU 22.04 ===
echo -e "${VERDE}Instalando GStreamer 1.20+ atualizado para Ubuntu 22.04...${SEM_COR}"
apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-x \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# Dependências de vídeo atualizadas para Ubuntu 22.04
apt-get install -y \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    libcanberra-gtk3-module \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev

echo -e "${VERDE}✓ GStreamer 1.20+ instalado para Ubuntu 22.04${SEM_COR}"

# === CONFIGURAR CUDA 12.2 ===
echo -e "${VERDE}Verificando CUDA 12.2 (JetPack 6.2)...${SEM_COR}"
if [ -d "/usr/local/cuda-12.2" ]; then
    echo -e "${VERDE}✓ CUDA 12.2 encontrado!${SEM_COR}"
    export CUDA_HOME=/usr/local/cuda-12.2
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
elif [ -d "/usr/local/cuda" ]; then
    echo -e "${AMARELO}⚠ Usando CUDA padrão (possível versão diferente)${SEM_COR}"
    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    echo -e "${VERMELHO}✗ CUDA não encontrado!${SEM_COR}"
fi

# Configurar TensorRT (vem com JetPack 6.2)
echo -e "${VERDE}Configurando TensorRT 8.6+...${SEM_COR}"
python3 -m pip install pycuda

# === FERRAMENTAS DE MONITORAMENTO JETSON ===
echo -e "${AZUL}Instalando ferramentas de monitoramento Jetson...${SEM_COR}"
python3 -m pip install \
    jetson-stats \
    jtop

# === CONFIGURAR VARIÁVEIS DE AMBIENTE PERMANENTES ===
echo -e "${VERDE}Configurando variáveis de ambiente para Jetson Orin Nano Super...${SEM_COR}"
cat >> /root/.bashrc << 'EOF'

# === Configurações Jetson Orin Nano Super (JetPack 6.2) ===
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Aliases úteis para Jetson Orin Nano Super
alias enable-super='sudo nvpmodel -m 2 && echo "Super Mode ativado: 67 TOPS!"'
alias disable-super='sudo nvpmodel -m 0 && echo "Modo normal ativado"'
alias check-perf='jtop'
alias gpu-info='nvidia-smi'
alias check-model='sudo nvpmodel -q'

# ROS2 Humble
source /opt/ros/humble/setup.bash
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

echo "🚀 Jetson Orin Nano Super configurado!"
echo "📊 Use 'enable-super' para 67 TOPS"
echo "📈 Use 'jtop' para monitorar performance"
EOF

# === CONFIGURAR ROSDEP PARA ROS2 HUMBLE ===
echo -e "${VERDE}Configurando rosdep para ROS2 Humble...${SEM_COR}"
if [ ! -d "/etc/ros/rosdep/sources.list.d" ]; then
    mkdir -p /etc/ros/rosdep/sources.list.d
fi

cat > /etc/ros/rosdep/sources.list.d/custom-rules.yaml << 'EOF'
ament_python:
  ubuntu: [python3-pip]
ultralytics:
  ubuntu: [python3-pip]
EOF

# Atualizar rosdep
rosdep update || echo -e "${AMARELO}rosdep update falhou, mas continuando${SEM_COR}"

# === VERIFICAÇÃO FINAL DE PACOTES ===
echo -e "${AZUL}Verificando instalação de pacotes críticos modernos...${SEM_COR}"

verify_package() {
    local package=$1
    local module=$2
    echo -n "Verificando $package... "
    if python3 -c "import $module; print(f'${VERDE}✓ Versão: {$module.__version__}${SEM_COR}')" 2>/dev/null; then
        return 0
    else
        echo -e "${VERMELHO}✗ Falhou!${SEM_COR}"
        return 1
    fi
}

echo -e "\n${AZUL}=== Verificação Final das Versões Modernas ===${SEM_COR}"
verify_package "NumPy" "numpy"
verify_package "OpenCV" "cv2"
verify_package "Ultralytics YOLOv8" "ultralytics"

# Verificar TensorFlow ou PyTorch
if python3 -c "import tensorflow" 2>/dev/null; then
    verify_package "TensorFlow" "tensorflow"
elif python3 -c "import torch" 2>/dev/null; then
    verify_package "PyTorch" "torch"
fi

verify_package "SciPy" "scipy"
verify_package "h5py" "h5py"
verify_package "Matplotlib" "matplotlib"
verify_package "Pillow" "PIL"
verify_package "PyYAML" "yaml"

echo -e "\n${VERDE}===============================================${SEM_COR}"
echo -e "${VERDE}  ✅ Instalação Concluída para Jetson Orin Nano Super!${SEM_COR}"
echo -e "${VERDE}===============================================${SEM_COR}"

echo -e "\n${AMARELO}🚀 Próximos Passos:${SEM_COR}"
echo -e "${AMARELO}1. Use 'enable-super' para ativar Super Mode (67 TOPS)${SEM_COR}"
echo -e "${AMARELO}2. Monitor com 'jtop' para verificar performance${SEM_COR}"
echo -e "${AMARELO}3. Rebuild workspace: 'colcon build --packages-select perception'${SEM_COR}"