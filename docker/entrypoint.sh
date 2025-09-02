#!/bin/bash
set +e

# Cores para melhorar a visualização
VERDE='\033[0;32m'
AZUL='\033[0;34m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
CIANO='\033[0;36m'
MAGENTA='\033[0;35m'
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

# Caminhos de bibliotecas do JetPack (necessários para CUDA/cuDNN/TensorRT)
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH

# Configuração CUDA se existir
if [ -d "/usr/local/cuda-12.2" ]; then
    export CUDA_HOME=/usr/local/cuda-12.2
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-12.2/bin:$PATH
elif [ -d "/usr/local/cuda-10.2" ]; then
    export CUDA_HOME=/usr/local/cuda-10.2
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-10.2/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-10.2/bin:$PATH
fi

# Exibe informações do sistema
exibir_cabecalho "🤖 Container ROS2 $ROS_DISTRO para Jetson inicializado!"

# Exibe informações do ambiente ROS
exibir_info "Versão ROS" "$ROS_DISTRO"
exibir_info "Versão Python" "$ROS_PYTHON_VERSION"
exibir_info "Implementação RMW" "$RMW_IMPLEMENTATION"
exibir_info "Domain ID" "$ROS_DOMAIN_ID"

# Verifica ambiente CUDA
echo ""
exibir_cabecalho "🖥️ Ambiente CUDA e GPU"

if [ -n "$CUDA_HOME" ]; then
    exibir_info "CUDA Home" "$CUDA_HOME"
    exibir_info "CUDA Path" "$(which nvcc 2>/dev/null || echo 'Não encontrado')"
    exibir_info "CUDA Versão" "$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'Não detectado')"
else
    exibir_info "CUDA" "${VERMELHO}Não configurado${SEM_COR}"
fi

# Verifica bibliotecas CUDA
echo ""
echo -e "${CIANO}Bibliotecas CUDA:${SEM_COR}"
ldconfig -p | grep -E "libcuda|libcudart|libcublas" | while read -r lib; do
    echo -e "  ${VERDE}✓${SEM_COR} $lib"
done

# Verifica GPU NVIDIA (robusto)
echo ""
echo -e "${CIANO}Status da GPU Jetson:${SEM_COR}"
if [ -e "/dev/nvhost-gpu" ] || [ -e "/dev/nvhost-as-gpu" ] || [ -e "/dev/nvhost-ctrl-gpu" ] || [ -e "/dev/nvmap" ]; then
    echo -e "  ${VERDE}✓${SEM_COR} GPU Jetson detectada"
    ls -1 /dev/nvhost* /dev/nvmap 2>/dev/null | sed 's/^/    - /'
else
    if command -v tegrastats &> /dev/null; then
        verificar_disponibilidade "Status da GPU (tegrastats)" \
            "timeout 2s tegrastats --interval 1000 | head -n 1" \
            "GPU não detectada ou drivers não instalados."
    else
        echo -e "  ${VERMELHO}GPU NVIDIA não detectada${SEM_COR}"
    fi
fi

# Verificação de dispositivos de câmera
echo ""
verificar_disponibilidade "Câmeras disponíveis" \
    "ls -la /dev/video* 2>/dev/null" \
    "Nenhum dispositivo de câmera encontrado. Verifique as conexões."

# Verifica pacotes ROS disponíveis
echo ""
verificar_disponibilidade "Pacotes ROS2 do projeto" \
    "ros2 pkg list | grep -E 'roboime|perception'" \
    "Nenhum pacote personalizado encontrado. Execute o build primeiro."

echo -e "${AZUL}============================================================${SEM_COR}"

# Executa o comando passado como argumento ou inicia um shell interativo
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    # Força um shell que fica ativo - solução definitiva
    echo -e "${VERDE}Shell interativo iniciado. Digite 'exit' para sair.${SEM_COR}"
    cd /ros2_ws
    export PS1="\[\033[1;32m\]root@hsl-container\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]# "
    
    # Garante que stdin está conectado e força modo interativo
    if [ -t 0 ]; then
        exec /bin/bash
    else
        # Fallback para quando stdin não é um TTY
        exec /bin/bash < /dev/tty
    fi
fi
