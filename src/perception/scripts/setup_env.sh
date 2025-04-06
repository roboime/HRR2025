#!/bin/bash

# Script para configurar variáveis de ambiente UTF-8 permanentemente
# Resolve problemas de codificação ASCII ao executar nós ROS2

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

print_header "Configurando variáveis de ambiente UTF-8 permanentemente"

# Verificar se o script está sendo executado como root
if [ "$(id -u)" -ne 0 ]; then
    print_error "Este script precisa ser executado como root (sudo)."
    print_info "Use: sudo bash $(basename "$0")"
    exit 1
fi

# Caminho para o arquivo de configuração ROS
ROS_SETUP_FILE="/opt/ros/eloquent/setup.bash"
ENV_FILE="/etc/environment"
BASHRC_FILE="/etc/bash.bashrc"

# Verificar se o arquivo de setup ROS existe
if [ ! -f "$ROS_SETUP_FILE" ]; then
    print_error "Arquivo de setup do ROS não encontrado: $ROS_SETUP_FILE"
    print_info "Verificando alternativas..."
    # Procurar alternativas
    ROS_SETUP_FILE=$(find /opt/ros -name setup.bash | head -n 1)
    if [ -z "$ROS_SETUP_FILE" ]; then
        print_error "Nenhum arquivo de setup do ROS encontrado."
        exit 1
    else
        print_success "Encontrado arquivo alternativo: $ROS_SETUP_FILE"
    fi
fi

# Adicionar configurações de codificação ao arquivo de ambiente do sistema
print_info "Adicionando configurações UTF-8 ao $ENV_FILE..."

# Remover configurações antigas se existirem
sed -i '/PYTHONIOENCODING/d' $ENV_FILE
sed -i '/LANG=C.UTF-8/d' $ENV_FILE
sed -i '/LC_ALL=C.UTF-8/d' $ENV_FILE

# Adicionar novas configurações
echo "PYTHONIOENCODING=utf8" >> $ENV_FILE
echo "LANG=C.UTF-8" >> $ENV_FILE
echo "LC_ALL=C.UTF-8" >> $ENV_FILE
print_success "Configurações UTF-8 adicionadas ao $ENV_FILE"

# Criar um novo arquivo de configuração para o ROS que include as variáveis UTF-8
print_info "Criando arquivo de configuração para o ROS com UTF-8..."

ROS_ENV_FILE="/etc/profile.d/ros_utf8.sh"
cat > $ROS_ENV_FILE << EOF
#!/bin/bash

# Configurações UTF-8 para o ROS
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:\$PYTHONPATH"

# Incluir setup ROS original
if [ -f "$ROS_SETUP_FILE" ]; then
    source "$ROS_SETUP_FILE"
fi
EOF

chmod +x $ROS_ENV_FILE
print_success "Arquivo de configuração $ROS_ENV_FILE criado com sucesso"

# Modificar a inicialização do shell para incluir as configurações
print_info "Atualizando $BASHRC_FILE..."

# Verificar se as configurações já existem no arquivo
if ! grep -q "# ROS2 UTF-8 Configurations" $BASHRC_FILE; then
    # Adicionar configurações ao final do arquivo
    cat >> $BASHRC_FILE << EOF

# ROS2 UTF-8 Configurations
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:\$PYTHONPATH"
EOF
    print_success "Configurações adicionadas ao $BASHRC_FILE"
else
    print_info "Configurações já existem no $BASHRC_FILE"
fi

# Criar um wrapper para o comando ros2
print_info "Criando wrapper para o comando ros2..."

ROS2_WRAPPER="/usr/local/bin/ros2_wrapper"
cat > $ROS2_WRAPPER << EOF
#!/bin/bash

# Wrapper para o comando ros2 que inclui configurações UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:\$PYTHONPATH"

# Executar o comando ros2 com os argumentos passados
ros2 "\$@"
EOF

chmod +x $ROS2_WRAPPER
print_success "Wrapper $ROS2_WRAPPER criado com sucesso"

# Criar um alias para o comando ros2
print_info "Criando arquivo de alias para o comando ros2..."

ALIAS_FILE="/etc/profile.d/ros2_alias.sh"
cat > $ALIAS_FILE << EOF
#!/bin/bash

# Alias para o comando ros2 com configurações UTF-8
alias ros2='PYTHONIOENCODING=utf8 LANG=C.UTF-8 LC_ALL=C.UTF-8 ros2'
EOF

chmod +x $ALIAS_FILE
print_success "Arquivo de alias $ALIAS_FILE criado com sucesso"

print_header "Configuração concluída com sucesso!"
print_info "As variáveis de ambiente UTF-8 estão configuradas permanentemente."
print_info "Reinicie o sistema ou execute 'source $ENV_FILE' para aplicar as alterações."
print_info "Para testar, execute: ros2 launch perception perception.launch.py camera_src:=csi" 