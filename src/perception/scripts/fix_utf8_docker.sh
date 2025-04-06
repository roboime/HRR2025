#!/bin/bash

# Script para corrigir permanentemente problemas de codificação UTF-8 no Docker
# Este script deve ser executado como root dentro do contêiner

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

print_header "Corrigindo problemas de codificação UTF-8 no ROS2 (Docker)"

# Verificar permissões de root
if [ "$(id -u)" -ne 0 ]; then
    print_error "Este script precisa ser executado como root."
    print_info "Execute: sudo bash $(basename $0)"
    exit 1
fi

# 1. Modificar o executável ros2
print_info "Modificando o executável ros2..."

# Localizar o executável ros2
ROS2_EXECUTABLE=$(which ros2)

if [ -z "$ROS2_EXECUTABLE" ]; then
    print_error "Executável ros2 não encontrado. Verifique se ROS2 está instalado corretamente."
    exit 1
fi

print_info "Executável ros2 encontrado em: $ROS2_EXECUTABLE"

# Criar backup do executável original
cp "$ROS2_EXECUTABLE" "${ROS2_EXECUTABLE}.bak"
print_success "Backup criado: ${ROS2_EXECUTABLE}.bak"

# Criar novo executável com UTF-8 embutido
cat > "$ROS2_EXECUTABLE" << 'EOF'
#!/bin/bash

# ROS2 wrapper com configurações UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Obter o diretório do script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Executar o binário original com as configurações corretas
"${SCRIPT_DIR}/ros2.bak" "$@"
EOF

chmod +x "$ROS2_EXECUTABLE"
print_success "Executável ros2 modificado com configurações UTF-8"

# 2. Modificar arquivos de configuração do sistema
print_info "Modificando arquivos de configuração do sistema..."

# Criar arquivo de ambiente
cat > /etc/profile.d/ros2_utf8.sh << 'EOF'
#!/bin/bash

# Configurações UTF-8 para ROS2
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"
EOF

chmod +x /etc/profile.d/ros2_utf8.sh
print_success "Arquivo de ambiente criado: /etc/profile.d/ros2_utf8.sh"

# Adicionar ao /etc/environment
if ! grep -q "PYTHONIOENCODING=utf8" /etc/environment; then
    echo "PYTHONIOENCODING=utf8" >> /etc/environment
    echo "LANG=C.UTF-8" >> /etc/environment
    echo "LC_ALL=C.UTF-8" >> /etc/environment
    print_success "Variáveis adicionadas ao /etc/environment"
else
    print_info "Variáveis já existem em /etc/environment"
fi

# 3. Corrigir os módulos Python do ROS2
print_info "Localizando e corrigindo módulos Python do ROS2..."

# Encontrar diretório de instalação do ROS2
ROS2_PYTHON_DIR=$(python3 -c "import rclpy; print(rclpy.__path__[0])" 2>/dev/null)

if [ -z "$ROS2_PYTHON_DIR" ]; then
    print_error "Não foi possível encontrar os módulos Python do ROS2."
    print_info "Continuando com outras correções..."
else
    ROS2_ROOT_DIR=$(dirname $(dirname "$ROS2_PYTHON_DIR"))
    print_info "Diretório de instalação do ROS2 encontrado: $ROS2_ROOT_DIR"
    
    # Adicionar encoding UTF-8 nos arquivos Python principais
    find "$ROS2_ROOT_DIR" -name "*.py" -type f -exec grep -l "^#!/usr/bin/env python3" {} \; | xargs -I {} sed -i '1s|^#!/usr/bin/env python3.*|#!/usr/bin/env python3\n# -*- coding: utf-8 -*-|' {}
    print_success "Arquivos Python principais do ROS2 corrigidos"
    
    # Corrigir launch_service.py que pode causar o erro específico
    LAUNCH_FILES=$(find "$ROS2_ROOT_DIR" -name "launch_service.py" -type f)
    for file in $LAUNCH_FILES; do
        # Adicionar configuração de codificação no início
        sed -i '1s|^#!/usr/bin/env python3.*|#!/usr/bin/env python3\n# -*- coding: utf-8 -*-|' "$file"
        # Corrigir os métodos de codificação/decodificação
        sed -i 's/\.decode(/\.decode("utf-8",/g' "$file"
        sed -i 's/\.encode(/\.encode("utf-8",/g' "$file"
        print_success "Arquivo corrigido: $file"
    done
fi

# 4. Corrigir o launch do ROS2
print_info "Corrigindo módulo launch do ROS2..."

LAUNCH_CMD=$(which ros2-launch 2>/dev/null)
if [ -z "$LAUNCH_CMD" ]; then
    LAUNCH_CMD=$(find /opt -name "ros2launch" -type f 2>/dev/null | head -n 1)
fi

if [ -n "$LAUNCH_CMD" ]; then
    cp "$LAUNCH_CMD" "${LAUNCH_CMD}.bak"
    
    sed -i '1s|^#!/usr/bin/env python3.*|#!/usr/bin/env python3\n# -*- coding: utf-8 -*-|' "$LAUNCH_CMD"
    
    # Adicionar configurações UTF-8 após o shebang
    sed -i '2a import os\nos.environ["PYTHONIOENCODING"] = "utf8"\nos.environ["LANG"] = "C.UTF-8"\nos.environ["LC_ALL"] = "C.UTF-8"' "$LAUNCH_CMD"
    
    print_success "Comando launch corrigido: $LAUNCH_CMD"
else
    print_info "Comando ros2-launch não encontrado, tentando localizar o módulo Python"
    
    # Encontrar e corrigir o módulo de launch
    PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    
    if [ -d "$PYTHON_SITE_PACKAGES/launch" ]; then
        find "$PYTHON_SITE_PACKAGES/launch" -name "*.py" -type f -exec sed -i '1s|^#!/usr/bin/env python3.*|#!/usr/bin/env python3\n# -*- coding: utf-8 -*-|' {} \;
        find "$PYTHON_SITE_PACKAGES/launch" -name "*.py" -type f -exec sed -i '2a import os\nos.environ["PYTHONIOENCODING"] = "utf8"\nos.environ["LANG"] = "C.UTF-8"\nos.environ["LC_ALL"] = "C.UTF-8"' {} \;
        print_success "Módulos de launch Python corrigidos"
    else
        print_info "Diretório de launch não encontrado em $PYTHON_SITE_PACKAGES"
    fi
fi

# 5. Criar script de inicialização para o Docker
print_info "Criando script de inicialização..."

cat > /docker-entrypoint-utf8.sh << 'EOF'
#!/bin/bash

# Configurar variáveis UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Executar o comando original
exec "$@"
EOF

chmod +x /docker-entrypoint-utf8.sh
print_success "Script de inicialização criado: /docker-entrypoint-utf8.sh"

print_info "Você pode usar esse script como ponto de entrada para o Docker:"
print_info "ENTRYPOINT [\"/docker-entrypoint-utf8.sh\"]"
print_info "CMD [\"bash\"]"

# 6. Aplicar configurações na sessão atual
print_info "Aplicando configurações na sessão atual..."
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Verificação final
print_header "Correção UTF-8 concluída"
print_info "As configurações UTF-8 foram aplicadas permanentemente."
print_info "Teste executando: ros2 launch perception perception.launch.py camera_src:=csi enable_display:=true"
print_info "Se ainda encontrar problemas, reinicie o contêiner ou use o script /docker-entrypoint-utf8.sh como um wrapper." 