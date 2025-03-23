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

# Instalando dependências Python essenciais sem sobrescrever pacotes do sistema
echo -e "${AZUL}Instalando dependências Python essenciais...${SEM_COR}"
python3 -m pip install --no-deps cython==0.29.24
python3 -m pip install --no-deps pkgconfig==1.5.5
python3 -m pip install --no-deps testresources==2.0.1

# Detecção de versões do sistema para pacotes críticos para evitar reinstalação
echo -e "${VERDE}Verificando pacotes do sistema...${SEM_COR}"

# Função para verificar se o pacote já está instalado e funcionando corretamente
check_package() {
    local package=$1
    local module=$2
    local min_version=$3
    
    if python3 -c "import $module; from pkg_resources import parse_version; print(parse_version('$min_version') <= parse_version($module.__version__))" 2>/dev/null | grep -q "True"; then
        echo -e "${VERDE}$package já está instalado com versão compatível.${SEM_COR}"
        return 0  # Já instalado com versão compatível
    fi
    
    # Tenta importar apenas, sem verificar versão
    if python3 -c "import $module" 2>/dev/null; then
        echo -e "${AMARELO}$package está instalado, mas versão pode ser incompatível. Instalando versão específica...${SEM_COR}"
        return 1  # Instalado mas versão incompatível
    fi
    
    echo -e "${AMARELO}$package não encontrado. Instalando...${SEM_COR}"
    return 2  # Não instalado
}

# Instalação de pacotes apenas se necessário
install_if_needed() {
    local package=$1
    local module=$2
    local min_version=$3
    local install_cmd=$4
    
    check_package "$package" "$module" "$min_version"
    local result=$?
    
    if [ $result -ne 0 ]; then
        echo -e "${AZUL}Instalando $package...${SEM_COR}"
        eval $install_cmd
    fi
}

# Instalando dependências para Matplotlib 
echo -e "${AZUL}Instalando dependências para Matplotlib...${SEM_COR}"
python3 -m pip install --no-deps kiwisolver==1.3.1 
python3 -m pip install --no-deps cycler==0.10.0 
python3 -m pip install --no-deps pyparsing==2.4.7 
python3 -m pip install --no-deps python-dateutil==2.8.1

# Instalação de pacotes essenciais
install_if_needed "NumPy" "numpy" "1.16.0" "python3 -m pip install --no-deps numpy==1.19.4"
install_if_needed "SciPy" "scipy" "1.4.0" "python3 -m pip install --no-deps scipy==1.5.4"
install_if_needed "Matplotlib" "matplotlib" "3.2.0" "python3 -m pip install --no-deps matplotlib==3.3.4"
install_if_needed "h5py" "h5py" "2.10.0" "python3 -m pip install --no-deps h5py==2.10.0"
install_if_needed "OpenCV" "cv2" "4.0.0" "python3 -m pip install --no-deps opencv-python==4.5.3.56"
install_if_needed "Pillow" "PIL" "7.0.0" "python3 -m pip install --no-deps pillow==8.3.2"
install_if_needed "PyYAML" "yaml" "5.1" "python3 -m pip install --no-deps pyyaml==5.4.1"
install_if_needed "Protobuf" "google.protobuf" "3.10.0" "python3 -m pip install --no-deps protobuf==3.17.3"

# Modificação para o TensorFlow
echo -e "${AMARELO}Instalando TensorFlow compatível com Jetson...${SEM_COR}"
# Verificar se o TensorFlow já está instalado
if ! python3 -c "import tensorflow" 2>/dev/null; then
    echo -e "${AZUL}TensorFlow não encontrado. Instalando...${SEM_COR}"
    
    # Instalar dependências críticas do TensorFlow primeiro
    python3 -m pip install --no-deps keras-preprocessing==1.1.2
    python3 -m pip install --no-deps gast==0.4.0
    python3 -m pip install --no-deps six==1.15.0
    
    # Instalar TensorFlow sem dependências
    python3 -m pip install --no-deps --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.5.0+nv21.8 || {
        echo -e "${AMARELO}Falha na instalação do TensorFlow específico para Jetson.${SEM_COR}"
        echo -e "${AZUL}Tentando instalar TensorFlow alternativo...${SEM_COR}"
        
        # Tentar versão mais antiga compatível com Jetson
        python3 -m pip install --no-deps 'tensorflow<2.0' || {
            echo -e "${VERMELHO}Não foi possível instalar o TensorFlow. Verifique manualmente a versão compatível com sua Jetson.${SEM_COR}"
        }
    }
else
    echo -e "${VERDE}TensorFlow já está instalado.${SEM_COR}"
fi

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

# Verificação final de pacotes
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

verify_package "TensorFlow" "tensorflow"
verify_package "NumPy" "numpy"
verify_package "SciPy" "scipy"
verify_package "OpenCV" "cv2"
verify_package "h5py" "h5py"
verify_package "Matplotlib" "matplotlib"
verify_package "Pillow" "PIL"
verify_package "PyYAML" "yaml"

echo -e "${VERDE}Instalação concluída!${SEM_COR}" 