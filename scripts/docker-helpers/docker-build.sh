#!/bin/bash

# Script de Build para Jetson Orin Nano Super Developer Kit
# JetPack 6.2 | ROS2 Humble | Ubuntu 22.04

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Função para print colorido
print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_header "BUILD DOCKER IMAGE - JETSON ORIN NANO SUPER"

# Verificar se estamos na raiz do projeto
if [ ! -f "docker/Dockerfile.jetson" ]; then
    print_error "Execute este script da raiz do projeto HSL2025!"
    exit 1
fi

# Verificar se o Docker está rodando
if ! docker info >/dev/null 2>&1; then
    print_error "Docker não está rodando. Inicie o Docker primeiro."
    exit 1
fi

# Nome da imagem
IMAGE_NAME="hsl:orin-super"
DOCKERFILE_PATH="docker/Dockerfile.jetson"

print_info "Construindo imagem Docker: $IMAGE_NAME"
print_info "Usando Dockerfile: $DOCKERFILE_PATH"
print_info "Target: Jetson Orin Nano Super (JetPack 6.2)"

# Build da imagem
echo -e "\n${YELLOW}Iniciando build da imagem...${NC}"
echo -e "${YELLOW}Isso pode demorar 10-20 minutos na primeira vez.${NC}\n"

docker build \
    -f "$DOCKERFILE_PATH" \
    -t "$IMAGE_NAME" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    . || {
    print_error "Falha no build da imagem Docker!"
    exit 1
}

print_success "Imagem Docker construída com sucesso: $IMAGE_NAME"

# Verificar se a imagem foi criada
if docker images | grep -q "hsl.*orin-super"; then
    print_success "Verificação da imagem: OK"
    
    # Mostrar informações da imagem
    echo -e "\n${BLUE}Informações da Imagem:${NC}"
    docker images | head -1
    docker images | grep "hsl.*orin-super"
    
    # Mostrar tamanho
    IMAGE_SIZE=$(docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep "hsl.*orin-super" | awk '{print $3}')
    print_info "Tamanho da imagem: $IMAGE_SIZE"
    
else
    print_error "Imagem não foi criada corretamente!"
    exit 1
fi

print_header "BUILD CONCLUÍDO COM SUCESSO!"

echo -e "${GREEN}Próximos passos:${NC}"
echo -e "${YELLOW}1. Execute o container:${NC} ./scripts/docker-helpers/docker-run.sh"
echo -e "${YELLOW}2. Dentro do container, instale dependências:${NC} /setup/install_dependencies.sh"
echo -e "${YELLOW}3. Habilite Super Mode:${NC} sudo nvpmodel -m 2"
echo -e "${YELLOW}4. Build o workspace:${NC} colcon build --packages-select perception"
echo -e "${YELLOW}5. Monitor performance:${NC} jtop"

print_success "Pronto para desenvolvimento na Jetson Orin Nano Super!"
