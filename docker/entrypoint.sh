#!/bin/bash
set -e

# Cores para melhorar a visualização
VERDE='\033[0;32m'
AZUL='\033[0;34m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
CIANO='\033[0;36m'
SEM_COR='\033[0m'

# Função para exibir cabeçalhos
exibir_cabecalho() {
    echo -e "${AZUL}============================================================${SEM_COR}"
    echo -e "${VERDE}$1${SEM_COR}"
    echo -e "${AZUL}============================================================${SEM_COR}"
}

# Função para exibir informações com formatação
exibir_info() {
    echo -e "${CIANO}$1:${SEM_COR} $2"
}

# Função para exibir comandos
exibir_comando() {
    echo -e "  ${AMARELO}$1${SEM_COR}"
}

# Função para verificar disponibilidade de recursos
verificar_disponibilidade() {
    local recurso=$1
    local comando=$2
    local mensagem_erro=$3
    
    echo -e "${CIANO}$recurso:${SEM_COR}"
    eval $comando || echo -e "  ${VERMELHO}$mensagem_erro${SEM_COR}"
}

# Configuração do ambiente ROS2
source /opt/ros/$ROS_DISTRO/setup.bash

# Configuração do workspace, se existir
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

# Garante que as variáveis de ambiente necessárias estejam definidas
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONIOENCODING=utf-8

# Exibe informações do sistema
exibir_cabecalho "🤖 Container ROS2 $ROS_DISTRO para Jetson inicializado!"

# Exibe informações do ambiente ROS
exibir_info "Versão ROS" "$ROS_DISTRO"
exibir_info "Versão Python" "$ROS_PYTHON_VERSION"
exibir_info "Implementação RMW" "$RMW_IMPLEMENTATION"
exibir_info "Domain ID" "$ROS_DOMAIN_ID"

# Verifica pacotes ROS disponíveis
echo ""
verificar_disponibilidade "Pacotes ROS2 do projeto" \
    "ros2 pkg list | grep -E 'roboime|perception'" \
    "Nenhum pacote personalizado encontrado. Execute o build primeiro."

# Verifica GPU NVIDIA
echo ""
verificar_disponibilidade "Status da GPU NVIDIA" \
    "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory --format=csv,noheader" \
    "GPU NVIDIA não detectada ou drivers não instalados."

# Verifica dispositivos de câmera
echo ""
verificar_disponibilidade "Câmeras disponíveis" \
    "ls -la /dev/video* 2>/dev/null" \
    "Nenhum dispositivo de câmera encontrado. Verifique as conexões."

# Exibe informações de uso
echo ""
exibir_cabecalho "📋 Comandos úteis"

echo -e "${CIANO}Para instalar as dependências:${SEM_COR}"
exibir_comando "/setup/install_dependencies.sh"

echo -e "${CIANO}Para compilar o workspace:${SEM_COR}"
exibir_comando "cd /ros2_ws && colcon build --symlink-install"

echo -e "${CIANO}Para visualizar tópicos disponíveis:${SEM_COR}"
exibir_comando "ros2 topic list"

echo -e "${AZUL}============================================================${SEM_COR}"

# Executa o comando passado como argumento ou inicia um shell
exec "$@" || exec /bin/bash
