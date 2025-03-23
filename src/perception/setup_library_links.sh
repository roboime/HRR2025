#!/bin/bash
set -e

# Cores para formatação
VERDE='\033[0;32m'
VERMELHO='\033[0;31m'
AMARELO='\033[1;33m'
AZUL='\033[0;34m'
SEM_COR='\033[0m'

# Verifica se está sendo executado como root
if [ "$EUID" -ne 0 ]; then
  echo -e "${VERMELHO}Este script precisa ser executado como root.${SEM_COR}"
  echo -e "Use: ${AMARELO}sudo ./setup_library_links.sh${SEM_COR}"
  exit 1
fi

echo -e "${AZUL}=== Configurando links para bibliotecas Python ===${SEM_COR}"

# Diretórios importantes
SYSTEM_PYTHON_DIR="/usr/local/lib/python3.6/dist-packages"
ROS_PYTHON_DIR="/opt/ros/eloquent/lib/python3.6/site-packages"
WORKSPACE_PYTHON_DIR="/ros2_ws/install/perception/lib/python3.6/site-packages"

# Verifica e cria diretório de saída para TensorFlow
echo -e "${AZUL}Verificando diretórios...${SEM_COR}"
mkdir -p ${ROS_PYTHON_DIR}
mkdir -p ${WORKSPACE_PYTHON_DIR}

# Bibliotecas para vincular
LIBRARIES=(
  "tensorflow"
  "tensorflow_core"
  "tensorflow_estimator"
  "tensorboard"
  "cv2"
  "numpy"
  "h5py"
  "PIL"
  "scipy"
  "matplotlib"
  "yaml"
)

# Função para criar link simbólico
create_link() {
  local lib=$1
  local src_dir=$2
  local dest_dir=$3

  if [ -d "${src_dir}/${lib}" ] || [ -f "${src_dir}/${lib}.py" ]; then
    if [ -d "${src_dir}/${lib}" ]; then
      # Para diretórios
      if [ ! -e "${dest_dir}/${lib}" ]; then
        ln -sf "${src_dir}/${lib}" "${dest_dir}/${lib}"
        echo -e "${VERDE}✓${SEM_COR} Link criado: ${lib} -> ${dest_dir}/${lib}"
      else
        echo -e "${AMARELO}!${SEM_COR} Link já existe: ${lib}"
      fi
    elif [ -f "${src_dir}/${lib}.py" ]; then
      # Para arquivos .py
      if [ ! -e "${dest_dir}/${lib}.py" ]; then
        ln -sf "${src_dir}/${lib}.py" "${dest_dir}/${lib}.py"
        echo -e "${VERDE}✓${SEM_COR} Link criado: ${lib}.py -> ${dest_dir}/${lib}.py"
      else
        echo -e "${AMARELO}!${SEM_COR} Link já existe: ${lib}.py"
      fi
    fi
  else
    echo -e "${VERMELHO}✗${SEM_COR} Biblioteca não encontrada: ${lib} em ${src_dir}"
  fi
}

# Criar links do sistema para o diretório do ROS
echo -e "\n${AZUL}Criando links para o diretório do ROS...${SEM_COR}"
for lib in "${LIBRARIES[@]}"; do
  create_link "$lib" "$SYSTEM_PYTHON_DIR" "$ROS_PYTHON_DIR"
done

# Criar links do sistema para o diretório do workspace
echo -e "\n${AZUL}Criando links para o diretório do workspace...${SEM_COR}"
for lib in "${LIBRARIES[@]}"; do
  create_link "$lib" "$SYSTEM_PYTHON_DIR" "$WORKSPACE_PYTHON_DIR"
done

# Verificar TensorFlow e OpenCV no ROS Python
echo -e "\n${AZUL}Verificando instalação...${SEM_COR}"
echo -e "${AMARELO}Verificando TensorFlow em ROS Python${SEM_COR}"
PYTHONPATH="$ROS_PYTHON_DIR:$PYTHONPATH" python3 -c "import tensorflow; print(f'✓ TensorFlow {tensorflow.__version__} encontrado')" || echo -e "${VERMELHO}✗ TensorFlow não encontrado${SEM_COR}"

echo -e "${AMARELO}Verificando OpenCV em ROS Python${SEM_COR}"
PYTHONPATH="$ROS_PYTHON_DIR:$PYTHONPATH" python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__} encontrado')" || echo -e "${VERMELHO}✗ OpenCV não encontrado${SEM_COR}"

echo -e "${AMARELO}Verificando TensorFlow e OpenCV no ambiente Python padrão${SEM_COR}"
python3 -c "import tensorflow, cv2; print(f'✓ TensorFlow {tensorflow.__version__}, OpenCV {cv2.__version__} encontrados')" || echo -e "${VERMELHO}✗ Erro ao carregar bibliotecas${SEM_COR}"

echo -e "\n${AZUL}Configuração concluída!${SEM_COR}"
echo -e "Agora você pode executar: ${VERDE}./src/perception/test_perception.sh${SEM_COR}"

# Criar diretório de modelos se não existir
MODEL_DIR="./src/perception/resources/models"
if [ ! -d "$MODEL_DIR" ]; then
  mkdir -p "$MODEL_DIR"
  echo -e "${VERDE}✓${SEM_COR} Diretório de modelos criado: $MODEL_DIR"
fi

echo -e "\n${AZUL}Nota:${SEM_COR} Se o erro persistir, você pode adicionar a seguinte linha ao seu .bashrc:"
echo -e "${AMARELO}export PYTHONPATH=\"$SYSTEM_PYTHON_DIR:\$PYTHONPATH\"${SEM_COR}" 