#!/bin/bash

# Script para configurar links simbólicos entre as bibliotecas Python do sistema e o ambiente ROS2
# Isso resolve problemas de importação de bibliotecas em ambientes ROS2

# Cores para mensagens
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Funções
function print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Verificar se está sendo executado como root
if [ "$EUID" -ne 0 ]; then
    print_error "Este script precisa ser executado como root (sudo)."
    exit 1
fi

print_header "Configurando links simbólicos para bibliotecas Python"

# Definir diretórios
SYSTEM_PYTHON_DIR="/usr/local/lib/python3.6/dist-packages"
ROS_PYTHON_DIR="/opt/ros/eloquent/lib/python3.6/site-packages"

# Verificar se os diretórios existem
if [ ! -d "$SYSTEM_PYTHON_DIR" ]; then
    print_error "Diretório Python do sistema não encontrado: $SYSTEM_PYTHON_DIR"
    exit 1
fi

if [ ! -d "$ROS_PYTHON_DIR" ]; then
    print_error "Diretório Python do ROS2 não encontrado: $ROS_PYTHON_DIR"
    exit 1
fi

print_info "Diretório Python do sistema: $SYSTEM_PYTHON_DIR"
print_info "Diretório Python do ROS2: $ROS_PYTHON_DIR"

# Função para criar link simbólico para um pacote Python
function create_symlink() {
    local package=$1
    local sys_path="$SYSTEM_PYTHON_DIR/$package"
    local ros_path="$ROS_PYTHON_DIR/$package"
    
    if [ -e "$sys_path" ]; then
        if [ -L "$ros_path" ] || [ -e "$ros_path" ]; then
            print_info "Removendo link/diretório existente: $ros_path"
            rm -rf "$ros_path"
        fi
        
        print_info "Criando link simbólico para $package..."
        ln -s "$sys_path" "$ros_path"
        if [ $? -eq 0 ]; then
            print_success "Link para $package criado com sucesso!"
        else
            print_error "Falha ao criar link para $package."
        fi
    else
        print_error "Pacote $package não encontrado em $SYSTEM_PYTHON_DIR"
    fi
}

# Bibliotecas para criar links
LIBRARIES=(
    "tensorflow"
    "tensorflow-2.5.0+nv21.8.dist-info"
    "cv2"
    "numpy"
    "h5py"
    "matplotlib"
    "PIL"
    "scipy"
    "yaml"
    "wrapt"
    "absl"
    "typing_extensions.py"
    "gast"
    "astunparse"
    "termcolor.py"
    "flatbuffers"
    "google"
    "opt_einsum"
)

# Criar links para cada biblioteca
for lib in "${LIBRARIES[@]}"; do
    create_symlink "$lib"
done

# Verificar se os links foram criados corretamente
print_header "Verificando instalação das bibliotecas no ambiente ROS2"

# Salvar estado atual do PATH Python
ORIGINAL_PYTHONPATH=$PYTHONPATH

# Configurar para usar apenas o ambiente ROS2
export PYTHONPATH="$ROS_PYTHON_DIR"

# Verificar TensorFlow
print_info "Verificando TensorFlow..."
if python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null; then
    print_success "TensorFlow está funcionando corretamente no ambiente ROS2!"
else
    print_error "TensorFlow ainda não está funcionando no ambiente ROS2."
fi

# Verificar OpenCV
print_info "Verificando OpenCV..."
if python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>/dev/null; then
    print_success "OpenCV está funcionando corretamente no ambiente ROS2!"
else
    print_error "OpenCV ainda não está funcionando no ambiente ROS2."
fi

# Restaurar PYTHONPATH original
export PYTHONPATH=$ORIGINAL_PYTHONPATH

print_header "Instruções"
echo "1. Agora você pode executar o sistema de percepção sem modificar o PYTHONPATH."
echo "2. Para testar, execute: ./src/perception/test_perception.sh"
echo "3. Se ainda houver problemas, tente reiniciar o terminal ou o sistema."
echo "4. Este script pode ser executado novamente se necessário."
echo
print_success "Configuração concluída!" 