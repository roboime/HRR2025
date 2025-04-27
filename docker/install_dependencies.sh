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
    liblapack-dev libblas-dev gfortran pkg-config x11-apps

# Instalando dependências adicionais do sistema
echo -e "${VERDE}Instalando dependências adicionais do sistema...${SEM_COR}"
apt-get install -y libfreetype6-dev libpng-dev pkg-config \
    libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module \
    python3-pil.imagetk python3-tk ros-eloquent-rqt ros-eloquent-rqt-common-plugins

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
install_if_needed "Pillow" "PIL" "7.0.0" "python3 -m pip install --no-deps pillow==8.3.2"
install_if_needed "PyYAML" "yaml" "5.1" "python3 -m pip install --no-deps pyyaml==5.4.1"
install_if_needed "Protobuf" "google.protobuf" "3.10.0" "python3 -m pip install --no-deps protobuf==3.17.3"

# --- Seção de Instalação do OpenCV ---
echo -e "${AZUL}Instalando OpenCV com GStreamer no Jetson Nano (JetPack 4.6)...${SEM_COR}"
echo -e "${AMARELO}Este processo pode demorar mais de uma hora!${SEM_COR}"

# Criar script temporário de instalação do OpenCV
cat > /tmp/install_opencv_jetson.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Instalação do OpenCV com GStreamer no Jetson Nano (JetPack 4.6) ==="

# 1. Instalar dependências via apt
echo "[1/5] Instalando pacotes de dependência..."
sudo apt-get update 
sudo apt-get install -y build-essential cmake git unzip pkg-config gfortran
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev zlib1g-dev
sudo apt-get install -y libgtk-3-dev libcanberra-gtk*  # GTK para HighGUI (ignorar avisos de '*' no pacote)
sudo apt-get install -y libv4l-dev v4l-utils 
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev gstreamer1.0-tools
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
sudo apt-get install -y libx264-dev libxvidcore-dev libvorbis-dev libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install -y libtbb-dev libeigen3-dev libatlas-base-dev libopenblas-dev liblapack-dev liblapacke-dev
sudo apt-get install -y libhdf5-dev libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev
sudo apt-get install -y python3-dev python3-numpy python3-pip

# (Opcional) Remover OpenCV pré-existente para evitar conflitos
# sudo apt-get purge -y python3-opencv libopencv* 

# 2. Configurar swap de 4GB (se já não existir)
echo "[2/5] Verificando swap..."
SWAPFILE="/swapfile"
if [ ! -f $SWAPFILE ]; then
  echo "Criando swapfile de 4GB em $SWAPFILE..."
  sudo fallocate -l 4G $SWAPFILE 
  sudo chmod 600 $SWAPFILE 
  sudo mkswap $SWAPFILE 
  sudo swapon $SWAPFILE 
fi
echo "Swap atual:"
free -h

# 3. Baixar código-fonte do OpenCV e opencv_contrib
echo "[3/5] Baixando OpenCV 4.6.0 e opencv_contrib..."
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.6.0.zip 
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.6.0.zip 
unzip -q opencv.zip && unzip -q opencv_contrib.zip
mv opencv-4.6.0 opencv && mv opencv_contrib-4.6.0 opencv_contrib
rm opencv.zip opencv_contrib.zip

# 4. Configurar e compilar o OpenCV
echo "[4/5] Configurando build do OpenCV via CMake..."
cd ~/opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D WITH_GSTREAMER=ON -D WITH_FFMPEG=ON \
      -D WITH_V4L=ON -D WITH_LIBV4L=ON \
      -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON \
      -D CUDA_ARCH_BIN="5.3" -D CUDA_ARCH_PTX="" \
      -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D ENABLE_NEON=ON \
      -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_OPENMP=ON \
      -D WITH_TBB=ON -D BUILD_TBB=ON \
      -D BUILD_TESTS=OFF -D BUILD_EXAMPLES=OFF \
      -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON \
      -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON .. 

echo "[4/5] Compilando OpenCV... isso pode demorar ~1 hora ou mais."
make -j4

echo "[5/5] Instalando OpenCV no sistema..."
sudo make install
sudo ldconfig

echo ">>> OpenCV 4.6.0 instalado com sucesso. Verificando suporte ao GStreamer..."
python3 -c "import cv2; info=cv2.getBuildInformation(); print(info[info.find('GStreamer'):info.find('GStreamer')+30])"
echo "Concluído!"
EOF

# Tornar o script executável
chmod +x /tmp/install_opencv_jetson.sh

# Executar o script
echo -e "${VERDE}Executando script de instalação do OpenCV...${SEM_COR}"
/tmp/install_opencv_jetson.sh || {
    echo -e "${VERMELHO}Falha durante a instalação do OpenCV!${SEM_COR}"
    exit 1
}

# Configuração para garantir que o OpenCV instalado seja usado pelo colcon
echo -e "${VERDE}Configurando OpenCV para ser usado pelo colcon...${SEM_COR}"
# Criar/atualizar arquivo de configuração pkg-config
cat > /usr/local/lib/pkgconfig/opencv4.pc << EOF
# Package Information for pkg-config

prefix=/usr
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include/opencv4

Name: OpenCV
Description: Open Source Computer Vision Library
Version: 4.6.0
Libs: -L\${libdir} -lopencv_dnn -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
Libs.private: -ldl -lm -lpthread -lrt
Cflags: -I\${includedir}
EOF

# Garantir que o pkg-config encontre o OpenCV instalado
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
echo "export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:\$PKG_CONFIG_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo -e "${VERDE}Instalação do OpenCV concluída. Configurado para ser usado como principal durante compilação colcon.${SEM_COR}"
# --- Fim da Seção de Instalação do OpenCV ---

# Modificação para o TensorFlow
echo -e "${AMARELO}Instalando TensorFlow compatível com Jetson...${SEM_COR}"
# Verificar se o TensorFlow já está instalado
if ! python3 -c "import tensorflow" 2>/dev/null; then
    echo -e "${AZUL}TensorFlow não encontrado. Instalando...${SEM_COR}"
    
    # Instalar dependências críticas do TensorFlow primeiro
    python3 -m pip install --no-deps keras-preprocessing==1.1.2
    python3 -m pip install --no-deps gast==0.4.0
    python3 -m pip install --no-deps six==1.15.0
    python3 -m pip install --no-deps typing_extensions==3.7.4.3
    python3 -m pip install --no-deps wrapt==1.12.1
    python3 -m pip install --no-deps absl-py==0.12.0
    python3 -m pip install --no-deps astunparse==1.6.3
    python3 -m pip install --no-deps termcolor==1.1.0
    python3 -m pip install --no-deps flatbuffers==1.12
    python3 -m pip install --no-deps google-pasta==0.2.0
    python3 -m pip install --no-deps opt-einsum==3.3.0
    
    # Instalar bibliotecas CUDA e cuDNN para Jetson
    echo -e "${VERDE}Instalando bibliotecas CUDA e cuDNN...${SEM_COR}"
    apt-get update
    apt-get install -y --no-install-recommends \
        cuda-cudart-10-2 \
        cuda-cublas-10-2 \
        cuda-cufft-10-2 \
        cuda-curand-10-2 \
        cuda-cusolver-10-2 \
        cuda-cusparse-10-2 \
        libcudnn8
        
    #Instalação de plugins gstreamer
    apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-x \
    libgstreamer1.0-dev \

    apt-get update && apt-get install -y \
        libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        libgstreamer-plugins-base1.0-dev libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev \
        libcanberra-gtk-module libv4l-dev libxvidcore-dev libx264-dev
    
    # Configurar variáveis de ambiente para CUDA
    export CUDA_HOME=/usr/local/cuda-10.2
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-10.2/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
    
    # Instalar TensorFlow sem dependências
    python3 -m pip install --no-deps --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.5.0+nv21.8 || {
        echo -e "${AMARELO}Falha na instalação do TensorFlow específico para Jetson.${SEM_COR}"
        echo -e "${AZUL}Tentando instalar TensorFlow alternativo...${SEM_COR}"
        
        # Tentar versão mais antiga compatível com Jetson
        python3 -m pip install --no-deps 'tensorflow<2.0' || {
            echo -e "${VERMELHO}Não foi possível instalar o TensorFlow. Verifique manualmente a versão compatível com sua Jetson.${SEM_COR}"
        }
    }
    
    # Instalar TensorBoard
    echo -e "${VERDE}Instalando TensorBoard...${SEM_COR}"
    python3 -m pip install tensorboard==2.5.0
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
echo -e "${AMARELO}ATENÇÃO: Como o OpenCV foi compilado do código-fonte, pode ser necessário reconstruir seu workspace ROS (colcon build) para que pacotes como cv_bridge utilizem a nova versão.${SEM_COR}" 