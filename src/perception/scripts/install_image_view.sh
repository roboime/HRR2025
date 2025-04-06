#!/bin/bash

# Script para instalar o pacote image_view para visualização

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

print_header "Instalando pacote image_view para visualização de imagens"

# Atualizar lista de pacotes
print_info "Atualizando repositórios..."
apt-get update

# Buscar informações sobre o pacote
print_info "Buscando informações sobre o pacote image_view..."
apt-cache search ros-eloquent-image-view

# Tentar instalar image_view
print_info "Tentando instalar ros-eloquent-image-view..."
apt-get install -y ros-eloquent-image-view

# Verificar se a instalação foi bem-sucedida
if [ $? -eq 0 ]; then
    print_success "Pacote image_view instalado com sucesso!"
else
    print_error "Falha ao instalar image_view. Tentando método alternativo..."
    
    # Tentar instalar image_pipeline que contém image_view
    print_info "Tentando instalar ros-eloquent-image-pipeline..."
    apt-get install -y ros-eloquent-image-pipeline
    
    if [ $? -eq 0 ]; then
        print_success "Pacote image_pipeline instalado com sucesso!"
    else
        print_error "Falha ao instalar pacotes pré-compilados"
        
        # Como último recurso, tentar instalar do código fonte
        print_info "Tentando instalar do código fonte..."
        
        # Criar diretório para o código fonte
        mkdir -p /tmp/image_view_ws/src
        cd /tmp/image_view_ws/src
        
        # Clonar repositório
        git clone https://github.com/ros-perception/image_pipeline.git -b eloquent
        
        # Compilar apenas image_view
        cd /tmp/image_view_ws
        source /opt/ros/eloquent/setup.bash
        colcon build --packages-select image_view
        
        if [ $? -eq 0 ]; then
            print_success "Pacote image_view compilado com sucesso!"
            echo "source /tmp/image_view_ws/install/setup.bash" >> ~/.bashrc
        else
            print_error "Não foi possível compilar image_view"
            
            # Mostrar alternativa
            print_info "Alternativa: Use a visualização interna da câmera com enable_display:=true"
        fi
    fi
fi

print_header "Testar image_view"

print_info "Para testar se a instalação funcionou, execute:"
echo "ros2 run image_view image_view --ros-args -r image:=/camera/image_raw"

print_info "Se não funcionar, use o parâmetro enable_display:=true no nó da câmera:"
echo "ros2 run perception/scripts/direct_camera_test.sh" 