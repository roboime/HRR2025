#!/bin/bash

# Definição de cores para melhorar a legibilidade
VERDE='\033[0;32m'
AZUL='\033[0;34m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
SEM_COR='\033[0m'

echo -e "${VERDE}Instalando dependências básicas do sistema...${SEM_COR}"
apt-get update && apt-get install -y build-essential cmake python3-dev libhdf5-dev \
    libopencv-dev python3-pip python3-setuptools \
    libhdf5-serial-dev hdf5-tools zlib1g-dev zip libjpeg8-dev \
    liblapack-dev libblas-dev gfortran pkg-config

# Removendo pacotes problemáticos para evitar conflitos
echo -e "${AMARELO}Removendo pacotes problemáticos do sistema...${SEM_COR}"
apt-get remove -y python3-numpy python3-scipy python3-matplotlib python3-h5py python3-opencv

# Instalando dependências adicionais do sistema
echo -e "${VERDE}Instalando dependências adicionais do sistema...${SEM_COR}"
apt-get install -y libfreetype6-dev libpng-dev pkg-config \
    libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module \
    python3-pil.imagetk python3-tk

echo -e "${VERDE}Atualizando pip e ferramentas essenciais...${SEM_COR}"
python3 -m pip install --upgrade pip setuptools wheel

echo -e "${VERDE}Definindo codificação UTF-8 para evitar problemas com caracteres especiais...${SEM_COR}"
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Instalando dependências essenciais básicas
echo -e "${AZUL}Instalando dependências Python essenciais...${SEM_COR}"
python3 -m pip install cython==0.29.24
python3 -m pip install pkgconfig==1.5.5
python3 -m pip install testresources==2.0.1

# Instalando pacotes com versões específicas e seguras para Jetson
echo -e "${VERDE}Instalando dependências Python com versões específicas...${SEM_COR}"

# NumPy - versão estável para aarch64
echo -e "${AZUL}Instalando NumPy...${SEM_COR}"
python3 -m pip install numpy==1.19.4

# SciPy - versão estável para aarch64
echo -e "${AZUL}Instalando SciPy...${SEM_COR}"
python3 -m pip install scipy==1.5.4

# Matplotlib precisa de dependências específicas
echo -e "${AZUL}Instalando dependências para Matplotlib...${SEM_COR}"
python3 -m pip install kiwisolver==1.3.1 
python3 -m pip install cycler==0.10.0 
python3 -m pip install pyparsing==2.4.7 
python3 -m pip install python-dateutil==2.8.1

echo -e "${AZUL}Instalando Matplotlib...${SEM_COR}"
python3 -m pip install matplotlib==3.3.4

# h5py pre-compilada para evitar problemas de compilação
echo -e "${AZUL}Instalando h5py...${SEM_COR}"
python3 -m pip install h5py==2.10.0 --no-build-isolation

# OpenCV com versão específica
echo -e "${AZUL}Instalando OpenCV...${SEM_COR}"
python3 -m pip install opencv-python==4.5.3.56

# Instalando PyYAML
echo -e "${AZUL}Instalando pyyaml...${SEM_COR}"
python3 -m pip install --ignore-installed --no-warn-script-location pyyaml==5.4.1

# Instalando Pillow
echo -e "${AZUL}Instalando Pillow...${SEM_COR}"
python3 -m pip install --ignore-installed --no-warn-script-location pillow==8.3.2

# Instalando protobuf (necessário para TensorFlow)
echo -e "${AZUL}Instalando protobuf...${SEM_COR}"
python3 -m pip install protobuf==3.17.3

# Tentando instalar TensorFlow compatível com Jetson
echo -e "${AMARELO}Instalando TensorFlow compatível com Jetson...${SEM_COR}"
python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.5.0+nv21.8 || {
    echo -e "${AMARELO}Falha na instalação do TensorFlow especifico para Jetson.${SEM_COR}"
    echo -e "${AZUL}Tentando instalar TensorFlow alternativo...${SEM_COR}"
    
    # Alguns Jetson funcionam com estas versões específicas
    python3 -m pip install 'tensorflow<2.0' || {
        echo -e "${VERMELHO}Não foi possível instalar o TensorFlow. Verifique manualmente a versão compatível com sua Jetson.${SEM_COR}"
    }
}

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

# Verificação com captura de erros
echo -e "${AZUL}Verificando instalação de pacotes críticos...${SEM_COR}"

verify_package() {
    local package=$1
    local module=$2
    echo -n "Verificando $package... "
    if python3 -c "import $module; print(f'${VERDE}Versão: {$module.__version__}${SEM_COR}')" 2>/dev/null; then
        echo -e "${VERDE}OK!${SEM_COR}"
        return 0
    else
        echo -e "${VERMELHO}Falhou!${SEM_COR}"
        return 1
    fi
}

verify_package "TensorFlow" "tensorflow.compat.v1"
verify_package "NumPy" "numpy"
verify_package "SciPy" "scipy"
verify_package "OpenCV" "cv2"
verify_package "h5py" "h5py"
verify_package "Matplotlib" "matplotlib"
verify_package "Pillow" "PIL"
verify_package "PyYAML" "yaml"

echo -e "${VERDE}Instalação concluída!${SEM_COR}" 