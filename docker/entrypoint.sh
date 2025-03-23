#!/bin/bash
set -e

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

# Função para verificar status de bibliotecas Python
verificar_biblioteca() {
    local nome=$1
    local modulo=$2
    
    echo -en "${CIANO}$nome:${SEM_COR} "
    if python3 -c "import $modulo; print('OK (' + $modulo.__version__ + ')')" 2>/dev/null; then
        return 0
    else
        echo -e "${VERMELHO}Não instalado${SEM_COR}"
        return 1
    fi
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

# Configuração CUDA se existir
if [ -d "/usr/local/cuda-10.2" ]; then
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

# Verifica GPU NVIDIA
echo ""
if command -v tegrastats &> /dev/null; then
    verificar_disponibilidade "Status da GPU Jetson" \
        "tegrastats --interval 1 --count 1" \
        "GPU Jetson não detectada ou drivers não instalados."
else
    echo -e "${CIANO}Status da GPU Jetson:${SEM_COR}"
    if [ -e "/dev/nvhost-ctrl" ]; then
        echo -e "  ${VERDE}✓${SEM_COR} GPU Jetson detectada (/dev/nvhost-ctrl)"
    else
        echo -e "  ${VERMELHO}GPU Jetson não detectada${SEM_COR}"
    fi
fi

# Verifica TensorFlow e status da GPU
echo ""
echo -e "${CIANO}Status do TensorFlow:${SEM_COR}"
python3 -c "
import sys
try:
    import tensorflow as tf
    print('  ${VERDE}✓${SEM_COR} TensorFlow ' + tf.__version__)
    print('  ${VERDE}✓${SEM_COR} Suporte CUDA: ' + ('Sim' if tf.test.is_built_with_cuda() else 'Não'))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print('  ${VERDE}✓${SEM_COR} GPU disponível: ' + gpu.name)
    else:
        print('  ${AMARELO}!${SEM_COR} Nenhuma GPU detectada pelo TensorFlow')
except ImportError:
    print('  ${VERMELHO}✗${SEM_COR} TensorFlow não instalado')
except Exception as e:
    print('  ${VERMELHO}✗${SEM_COR} Erro: ' + str(e))
" 2>/dev/null || echo -e "  ${VERMELHO}✗${SEM_COR} Erro ao verificar TensorFlow"

# Verifica dispositivos de câmera
echo ""
verificar_disponibilidade "Câmeras disponíveis" \
    "ls -la /dev/video* 2>/dev/null" \
    "Nenhum dispositivo de câmera encontrado. Verifique as conexões."

# Verifica pacotes ROS disponíveis
echo ""
verificar_disponibilidade "Pacotes ROS2 do projeto" \
    "ros2 pkg list | grep -E 'roboime|perception'" \
    "Nenhum pacote personalizado encontrado. Execute o build primeiro."

# Verifica bibliotecas Python
echo ""
exibir_cabecalho "📦 Bibliotecas Python para IA e Visão"

verificar_biblioteca "TensorFlow" "tensorflow"
verificar_biblioteca "NumPy" "numpy"
verificar_biblioteca "SciPy" "scipy"
verificar_biblioteca "OpenCV" "cv2"
verificar_biblioteca "h5py" "h5py"
verificar_biblioteca "Matplotlib" "matplotlib"
verificar_biblioteca "Pillow" "PIL"
verificar_biblioteca "PyYAML" "yaml"

# Exibe informações de uso
echo ""
exibir_cabecalho "📋 Comandos úteis"

echo -e "${CIANO}Para instalar as dependências:${SEM_COR}"
exibir_comando "/setup/install_dependencies.sh"

echo -e "${CIANO}Para compilar o workspace:${SEM_COR}"
exibir_comando "cd /ros2_ws && colcon build --symlink-install"

echo -e "${CIANO}Para executar um nó ROS2:${SEM_COR}"
exibir_comando "ros2 run <package_name> <node_name>"

echo -e "${CIANO}Para visualizar tópicos disponíveis:${SEM_COR}"
exibir_comando "ros2 topic list"

echo -e "${CIANO}Para testar as bibliotecas Python:${SEM_COR}"
exibir_comando "python3 -c \"import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU disponível:', tf.config.list_physical_devices('GPU'))\""

echo -e "${AZUL}============================================================${SEM_COR}"

# Executa o comando passado como argumento ou inicia um shell
exec "$@" || exec /bin/bash
