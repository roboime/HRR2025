#!/bin/bash

# Script simplificado para testar o sistema de percepção da RoboIME
# Este script torna mais fácil testar diferentes configurações do sistema de percepção

# Configurar ambiente com codificação UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Adicionar caminhos de bibliotecas importantes
# Usar o caminho completo para o Python do sistema e adicionar ao PYTHONPATH
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/targets/aarch64-linux/lib:$LD_LIBRARY_PATH"

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

# Verificar se o script está sendo executado na raiz do workspace
SCRIPT_DIR=$(dirname "$0")
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

if [[ "$PWD" != "$WORKSPACE_DIR" ]]; then
    print_info "Mudando para o diretório raiz do workspace: $WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"
fi

# Verificar instalação do ROS 2
print_header "Verificando ambiente"

if ! command -v ros2 &> /dev/null; then
    print_error "ROS 2 não encontrado. Certifique-se de que o ROS 2 está instalado e configurado."
    exit 1
fi
print_success "ROS 2 encontrado."

# Verificar workspace
if [ ! -f "./install/setup.bash" ]; then
    print_error "Arquivo setup.bash não encontrado. Execute 'colcon build' primeiro."
    exit 1
fi
print_success "Workspace encontrado."

# Verificar pacote de percepção
if [ ! -d "./src/perception" ]; then
    print_error "Pacote 'perception' não encontrado."
    exit 1
fi
print_success "Pacote 'perception' encontrado."

# Verificar se o pacote foi construído
if [ ! -d "./install/perception" ]; then
    print_error "O pacote 'perception' não foi construído. Execute 'colcon build --packages-select perception'."
    exit 1
fi
print_success "Pacote 'perception' construído."

# Configurar ambiente ROS
print_info "Configurando ambiente ROS 2..."
source ./install/setup.bash

# Adicionar caminho das bibliotecas Python do sistema ao PYTHONPATH de forma mais abrangente
print_info "Configurando caminhos para bibliotecas Python..."
export PYTHONPATH="/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"
print_success "PYTHONPATH configurado: $PYTHONPATH"

# Verificar dependências
print_header "Verificando dependências"

# Verificar TensorFlow com tratamento de erros mais robusto
echo "Verificando TensorFlow..."
if python3 -c "import tensorflow as tf; print(f'TensorFlow versão: {tf.__version__}')" 2>tensorflow_error.log; then
    print_success "TensorFlow está funcionando."
    # Verificar GPU no TensorFlow
    if python3 -c "import tensorflow as tf; print(f'GPU disponível: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')" 2>tensorflow_gpu_error.log; then
        print_success "GPU disponível para TensorFlow."
    else
        print_info "GPU não disponível para TensorFlow. O sistema funcionará mais lento."
    fi
else
    print_error "TensorFlow não encontrado ou com erro. Tentando instalar..."
    pip3 install tensorflow==2.4.0
    
    # Verificar novamente após instalação
    if python3 -c "import tensorflow as tf; print(f'TensorFlow versão: {tf.__version__}')" 2>/dev/null; then
        print_success "TensorFlow instalado com sucesso."
    else
        print_error "Não foi possível instalar o TensorFlow. O detector YOLOv4-Tiny não funcionará."
    fi
fi

# Verificar OpenCV com tratamento de erros mais robusto
echo "Verificando OpenCV..."
if python3 -c "import cv2; print(f'OpenCV versão: {cv2.__version__}')" 2>opencv_error.log; then
    print_success "OpenCV está funcionando."
else
    print_error "OpenCV não encontrado ou com erro. Tentando instalar..."
    pip3 install opencv-python
    
    # Verificar novamente após instalação
    if python3 -c "import cv2; print(f'OpenCV versão: {cv2.__version__}')" 2>/dev/null; then
        print_success "OpenCV instalado com sucesso."
    else
        print_error "Não foi possível instalar o OpenCV. Os detectores tradicionais não funcionarão."
    fi
fi

# Verificar se o modelo YOLOv4-Tiny existe
echo "Verificando modelo YOLOv4-Tiny..."
MODEL_PATH="./src/perception/resources/models/yolov4_tiny.h5"
if [ -f "$MODEL_PATH" ]; then
    print_success "Modelo YOLOv4-Tiny encontrado em $MODEL_PATH"
else
    print_error "Modelo YOLOv4-Tiny não encontrado em $MODEL_PATH"
    # Verificar diretório alternativo
    ALT_MODEL_PATH="./install/perception/share/perception/resources/models/yolov4_tiny.h5"
    if [ -f "$ALT_MODEL_PATH" ]; then
        print_success "Modelo YOLOv4-Tiny encontrado em $ALT_MODEL_PATH"
    else
        print_info "O diretório de modelos será criado se não existir"
        mkdir -p "$(dirname "$MODEL_PATH")"
    fi
fi

# Configurar permissões de execução para os scripts Python
print_header "Configurando permissões para scripts Python"

# Verificar e definir permissões para arquivos Python
find_python_files() {
    find "$1" -name "*.py" -type f
}

# Listar e configurar permissões para scripts Python
PYTHON_FILES=$(find_python_files "./src/perception")
if [ -n "$PYTHON_FILES" ]; then
    for py_file in $PYTHON_FILES; do
        if [ -f "$py_file" ]; then
            print_info "Configurando permissão para: $py_file"
            chmod +x "$py_file"
        fi
    done
    print_success "Permissões configuradas para scripts Python."
else
    print_info "Nenhum arquivo Python encontrado."
fi

# Reconstruir o pacote para atualizar os entry points
print_info "Reconstruindo o pacote perception para atualizar os entry points..."
colcon build --packages-select perception

# Verificar se o pacote foi construído corretamente
if [ ! -d "./install/perception" ]; then
    print_error "Falha ao construir o pacote perception."
    exit 1
fi
print_success "Pacote perception reconstruído com sucesso."

# Verificar entry points
if [ -f "./install/perception/lib/perception/vision_pipeline" ]; then
    print_success "Entry point vision_pipeline encontrado."
else
    print_error "Entry point vision_pipeline não encontrado. Verificando arquivos instalados:"
    find ./install/perception -type f -name "*vision*" | while read file; do
        print_info "Arquivo encontrado: $file"
    done
fi

# Adicionar opção para debug de lançamento
print_header "Verificando configuração"

# Verificar entry points instalados
print_info "Entry points instalados:"
find /ros2_ws/install/perception/lib -type f -executable | while read file; do
    echo "- $file"
done

# Verificar se os arquivos Python foram instalados
print_info "Arquivos Python instalados:"
find /ros2_ws/install/perception/lib/python* -name "*.py" 2>/dev/null | grep -v "__pycache__" | while read file; do
    echo "- $file"
done

# Menu interativo
print_header "Menu de Teste do Sistema de Percepção"
echo "Selecione uma opção para testar:"
echo ""
echo "1. Iniciar sistema completo (YOLOv4-Tiny + detectores tradicionais)"
echo "2. Iniciar apenas detector YOLOv4-Tiny"
echo "3. Iniciar apenas detectores tradicionais"
echo "4. Testar detecção de bola (YOLOv4-Tiny)"
echo "5. Testar detecção de bola (tradicional)"
echo "6. Testar detecção de gol"
echo "7. Testar detecção de robôs"
echo "8. Carregar webcam local"
echo "9. Executar testes de benchmark"
echo "10. Informações e status do sistema"
echo ""
echo "0. Sair"
echo ""

read -p "Escolha uma opção: " option

case $option in
    1)
        print_header "Iniciando sistema completo (YOLOv4-Tiny + tradicional)"
        ros2 launch perception perception.launch.py
        ;;
    2)
        print_header "Iniciando apenas detector YOLOv4-Tiny"
        ros2 launch perception perception.launch.py mode:=yolo
        ;;
    3)
        print_header "Iniciando apenas detectores tradicionais"
        ros2 launch perception perception.launch.py mode:=traditional
        ;;
    4)
        print_header "Testando detecção de bola (YOLOv4-Tiny)"
        ros2 launch perception perception.launch.py mode:=yolo detector_ball:=yolo
        ;;
    5)
        print_header "Testando detecção de bola (tradicional)"
        ros2 launch perception perception.launch.py mode:=traditional detector_ball:=traditional
        ;;
    6)
        print_header "Testando detecção de gol"
        echo "1. YOLOv4-Tiny"
        echo "2. Tradicional"
        read -p "Escolha: " goal_option
        if [ "$goal_option" -eq 1 ]; then
            ros2 launch perception perception.launch.py mode:=yolo detector_goal:=yolo
        else
            ros2 launch perception perception.launch.py mode:=traditional detector_goal:=traditional
        fi
        ;;
    7)
        print_header "Testando detecção de robôs"
        echo "1. YOLOv4-Tiny"
        echo "2. Tradicional"
        read -p "Escolha: " robot_option
        if [ "$robot_option" -eq 1 ]; then
            ros2 launch perception perception.launch.py mode:=yolo detector_robot:=yolo
        else
            ros2 launch perception perception.launch.py mode:=traditional detector_robot:=traditional
        fi
        ;;
    8)
        print_header "Carregando webcam local"
        ros2 launch perception perception.launch.py camera_src:=usb
        ;;
    9)
        print_header "Executando testes de benchmark"
        print_info "Teste 1: YOLOv4-Tiny não visualizado (5 segundos)"
        timeout 5 ros2 launch perception perception.launch.py mode:=yolo debug_image:=false
        
        print_info "Teste 2: Apenas YOLOv4-Tiny (5 segundos)"
        timeout 5 ros2 launch perception perception.launch.py mode:=yolo
        
        print_info "Teste 3: Apenas tradicional (5 segundos)"
        timeout 5 ros2 launch perception perception.launch.py mode:=traditional
        
        print_info "Teste 4: Detecção de bola YOLOv4-Tiny (5 segundos)"
        timeout 5 ros2 launch perception perception.launch.py mode:=yolo detector_ball:=yolo
        
        print_success "Testes concluídos!"
        ;;
    10)
        print_header "Informações e status do sistema"
        
        # Verificar status do ROS
        echo "Status do ROS 2:"
        ros2 node list 2>/dev/null || echo "Nenhum nó ROS em execução"
        
        # Verificar status dos tópicos
        echo -e "\nTópicos disponíveis:"
        ros2 topic list 2>/dev/null || echo "Nenhum tópico disponível"
        
        # Verificar status dos modelos
        echo -e "\nStatus dos modelos:"
        if [ -f "$MODEL_PATH" ]; then
            echo "YOLOv4-Tiny: Disponível ($(du -h "$MODEL_PATH" | cut -f1))"
        else
            echo "YOLOv4-Tiny: Não encontrado em $MODEL_PATH"
        fi
        
        # Verificar versões
        echo -e "\nVersões instaladas:"
        python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>/dev/null || echo "OpenCV: Não instalado"
        python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null || echo "TensorFlow: Não instalado"
        
        # Verificar recursos de hardware
        echo -e "\nRecursos de hardware:"
        python3 -c "import tensorflow as tf; print(f'GPUs disponíveis: {len(tf.config.list_physical_devices(\"GPU\"))}')" 2>/dev/null || echo "GPUs disponíveis: Desconhecido"
        echo "CPUs: $(nproc)"
        echo "Memória total: $(free -h | grep -i 'mem' | awk '{print $2}')"
        ;;
    0)
        print_info "Saindo..."
        exit 0
        ;;
    *)
        print_error "Opção inválida!"
        ;;
esac

# Opção para reiniciar o script
echo ""
read -p "Deseja executar o script novamente? (s/n): " run_again
if [[ "$run_again" == "s" ]] || [[ "$run_again" == "S" ]]; then
    exec "$0"
fi

print_info "Script finalizado." 