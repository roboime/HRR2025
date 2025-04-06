#!/bin/bash

# Script para configurar variáveis de ambiente UTF-8 permanentemente em ambiente Docker
# Deve ser executado como parte do Dockerfile ou como primeiro comando no contêiner

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

print_header "Configurando ambiente UTF-8 para ROS2 no Docker"

# Arquivos de configuração
ENV_FILE="/etc/environment"
BASHRC_FILE="/etc/bash.bashrc"
PROFILE_FILE="/etc/profile.d/ros_utf8.sh"

# Adicionar ao arquivo de ambiente
print_info "Configurando $ENV_FILE..."
grep -q "PYTHONIOENCODING" $ENV_FILE || echo "PYTHONIOENCODING=utf8" >> $ENV_FILE
grep -q "LANG=C.UTF-8" $ENV_FILE || echo "LANG=C.UTF-8" >> $ENV_FILE
grep -q "LC_ALL=C.UTF-8" $ENV_FILE || echo "LC_ALL=C.UTF-8" >> $ENV_FILE
print_success "Arquivo $ENV_FILE configurado"

# Adicionar ao bashrc global
print_info "Configurando $BASHRC_FILE..."
if ! grep -q "# ROS2 UTF-8 Configurations" $BASHRC_FILE; then
    cat >> $BASHRC_FILE << EOF

# ROS2 UTF-8 Configurations
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:\$PYTHONPATH"
EOF
    print_success "Configurações adicionadas a $BASHRC_FILE"
else
    print_info "Configurações já existem em $BASHRC_FILE"
fi

# Criar arquivo no profile.d
print_info "Criando arquivo $PROFILE_FILE..."
cat > $PROFILE_FILE << EOF
#!/bin/bash

# Configurações UTF-8 para ROS2
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:\$PYTHONPATH"
EOF
chmod +x $PROFILE_FILE
print_success "Arquivo $PROFILE_FILE criado"

# Criar wrapper ros2 para garantir que todas as execuções do ROS2 usem UTF-8
print_info "Criando wrapper para o comando ROS2..."
ROS2_WRAPPER="/usr/local/bin/ros2_utf8"
cat > $ROS2_WRAPPER << EOF
#!/bin/bash

# Wrapper para o comando ros2 com configurações UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:\$PYTHONPATH"

# Executa o comando ros2 original com os parâmetros passados
ros2 "\$@"
EOF
chmod +x $ROS2_WRAPPER
print_success "Wrapper $ROS2_WRAPPER criado"

# Criar um alias para o comando ros2
print_info "Criando alias para o comando ros2..."
ALIAS_FILE="/etc/profile.d/ros2_alias.sh"
cat > $ALIAS_FILE << EOF
#!/bin/bash

# Alias para o comando ros2 com configurações UTF-8
alias ros2='PYTHONIOENCODING=utf8 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:\$PYTHONPATH" ros2'
EOF
chmod +x $ALIAS_FILE
print_success "Arquivo de alias $ALIAS_FILE criado"

# Modificar os scripts wrapper existentes para garantir que usem UTF-8
print_info "Atualizando scripts wrapper existentes..."

for wrapper in $(find /ros2_ws/src -name "*wrapper.sh" -type f); do
    if [ -f "$wrapper" ]; then
        # Verificar se já tem as configurações
        if ! grep -q "PYTHONIOENCODING=utf8" "$wrapper"; then
            # Adicionar configurações UTF-8 logo após a linha do shebang
            sed -i '1a\
# Configurações UTF-8\
export PYTHONIOENCODING=utf8\
export LANG=C.UTF-8\
export LC_ALL=C.UTF-8\
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"
' "$wrapper"
            print_success "Script $wrapper atualizado"
        else
            print_info "Script $wrapper já está configurado"
        fi
    fi
done

print_header "Configuração concluída com sucesso!"
print_info "As variáveis de ambiente UTF-8 estão configuradas permanentemente."
print_info "Execute 'source $BASHRC_FILE' para aplicar as configurações na sessão atual."
print_info "Ou use diretamente o comando 'ros2_utf8' para executar comandos ROS2 com UTF-8." 