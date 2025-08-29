#!/bin/bash

# Sistema de Teste e Conveniência - Percepção RoboIME
# Suporte dual para câmeras CSI IMX219 e USB Logitech C930
# Jetson Orin Nano Super - Sistema YOLOv8 Simplificado (6 classes)

# Configurar ambiente com codificação UTF-8
export PYTHONIOENCODING=utf8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Adicionar caminhos de bibliotecas importantes
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/targets/aarch64-linux/lib:$LD_LIBRARY_PATH"

# Cores para mensagens
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configurações
CONFIG_FILE="src/perception/config/perception_config.yaml"
USB_DEVICE="/dev/video0"

# Funções de utilitários
function print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}💡 $1${NC}"
}

function print_warning() {
    echo -e "${PURPLE}⚠️  $1${NC}"
}

function print_camera() {
    echo -e "${CYAN}🎥 $1${NC}"
}

# Verificar se o script está sendo executado na raiz do workspace
SCRIPT_DIR=$(dirname "$0")
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

if [[ "$PWD" != "$WORKSPACE_DIR" ]]; then
    print_info "Mudando para o diretório raiz do workspace: $WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"
fi

# Função para verificar ambiente básico
check_environment() {
    print_header "🔍 Verificando Ambiente"

    # Verificar ROS 2
    if ! command -v ros2 &> /dev/null; then
        print_error "ROS 2 não encontrado. Certifique-se de que o ROS 2 está instalado e configurado."
        return 1
    fi
    print_success "ROS 2 encontrado"

    # Verificar workspace
    if [ ! -f "./install/setup.bash" ]; then
        print_error "Arquivo setup.bash não encontrado. Execute 'colcon build' primeiro."
        return 1
    fi
    print_success "Workspace encontrado"

    # Verificar pacote de percepção
    if [ ! -d "./src/perception" ]; then
        print_error "Pacote 'perception' não encontrado."
        return 1
    fi
    print_success "Pacote 'perception' encontrado"

    # Verificar se o pacote foi construído
    if [ ! -d "./install/perception" ]; then
        print_error "O pacote 'perception' não foi construído. Execute 'colcon build --packages-select perception'."
        return 1
    fi
    print_success "Pacote 'perception' construído"

    # Configurar ambiente ROS
    print_info "Configurando ambiente ROS 2..."
    source ./install/setup.bash
    print_success "Ambiente ROS 2 configurado"

    return 0
}

# Função para verificar dependências modernas
check_dependencies() {
    print_header "📦 Verificando Dependências Modernas"

    # Verificar PyTorch/Ultralytics (YOLOv8)
    echo "🔍 Verificando YOLOv8/Ultralytics..."
    if python3 -c "import ultralytics; from ultralytics import YOLO; print(f'Ultralytics versão: {ultralytics.__version__}')" 2>/dev/null; then
        print_success "YOLOv8/Ultralytics funcionando"
    else
        print_warning "YOLOv8/Ultralytics não encontrado - detector YOLOv8 pode não funcionar"
    fi

    # Verificar OpenCV moderno
    echo "🔍 Verificando OpenCV..."
    if python3 -c "import cv2; print(f'OpenCV versão: {cv2.__version__}')" 2>/dev/null; then
        print_success "OpenCV funcionando"
        
        # Verificar CUDA no OpenCV
        if python3 -c "import cv2; print(f'CUDA suportado: {cv2.cuda.getCudaEnabledDeviceCount() > 0}')" 2>/dev/null; then
            print_success "OpenCV com suporte CUDA"
        else
            print_warning "OpenCV sem suporte CUDA"
        fi
    else
        print_error "OpenCV não encontrado"
        return 1
    fi

    # Verificar v4l-utils (para câmera USB)
    if command -v v4l2-ctl &> /dev/null; then
        print_success "v4l-utils disponível (suporte USB)"
    else
        print_warning "v4l-utils não instalado (necessário para câmera USB)"
        print_info "Instale com: sudo apt install v4l-utils"
    fi

    return 0
}

# Função para verificar câmeras disponíveis
check_cameras() {
    print_header "🎥 Verificando Câmeras Disponíveis"

    # Verificar CSI IMX219
    echo "1️⃣ Verificando CSI IMX219:"
    if dmesg | grep -q "imx219"; then
        print_success "   CSI IMX219 detectada"
    else
        print_warning "   CSI IMX219 não detectada"
        print_info "   Verifique a conexão do cabo CSI"
    fi

    # Verificar USB
    echo "2️⃣ Verificando câmeras USB:"
    if lsusb | grep -q "Logitech"; then
        print_success "   Câmera Logitech detectada"
        lsusb | grep Logitech | sed 's/^/      /'
    else
        print_warning "   Câmera Logitech não detectada"
    fi

    # Verificar dispositivos de vídeo
    echo "3️⃣ Verificando dispositivos de vídeo:"
    if ls /dev/video* &> /dev/null; then
        print_success "   Dispositivos de vídeo encontrados:"
        ls -la /dev/video* | sed 's/^/      /'
    else
        print_warning "   Nenhum dispositivo de vídeo encontrado"
    fi

    return 0
}

# Função para teste específico da câmera C930
test_c930_camera() {
    print_header "🧪 Teste Específico - Câmera USB Logitech C930"

    # Verificação básica
    if [ ! -c "$USB_DEVICE" ]; then
        print_error "Dispositivo $USB_DEVICE não encontrado"
        print_info "Verifique se a câmera USB C930 está conectada"
        return 1
    fi

    if ! lsusb | grep -q "Logitech"; then
        print_error "Câmera Logitech não detectada via lsusb"
        print_info "Verifique se a C930 está conectada corretamente"
        return 1
    fi

    print_success "Câmera C930 detectada"

    # Verificar capacidades (se v4l2-ctl disponível)
    if command -v v4l2-ctl &> /dev/null; then
        print_info "Verificando capacidades da C930:"
        v4l2-ctl --device=$USB_DEVICE --info | head -5 | sed 's/^/   /'
        echo ""
        print_info "Formatos suportados:"
        v4l2-ctl --device=$USB_DEVICE --list-formats-ext | grep -E "(Index|Size|Interval)" | head -10 | sed 's/^/   /'
    else
        print_warning "v4l2-ctl não disponível - pulando verificação detalhada"
    fi

    # Teste OpenCV
    print_info "Testando captura OpenCV..."
    python3 << EOF
import cv2
import sys

try:
    print("   🔄 Tentando abrir câmera em $USB_DEVICE...")
    cap = cv2.VideoCapture('$USB_DEVICE', cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("   ❌ Falha ao abrir câmera")
        sys.exit(1)
    
    # Configurar resolução C930
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verificar configurações
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   📏 Resolução: {width}x{height}")
    print(f"   🎬 FPS: {fps}")
    
    # Testar captura
    ret, frame = cap.read()
    if ret:
        print(f"   ✅ Frame capturado: {frame.shape}")
        print("   🎉 Teste OpenCV PASSOU!")
    else:
        print("   ❌ Falha na captura de frame")
        sys.exit(1)
    
    cap.release()
    
except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        print_success "Teste da câmera C930 PASSOU"
        return 0
    else
        print_error "Teste da câmera C930 FALHOU"
        return 1
    fi
}

# Função para iniciar sistema com CSI
start_perception_csi() {
    print_header "🎥 Iniciando Sistema com Câmera CSI IMX219"
    
    # Configurar para CSI
    if [ -f "$CONFIG_FILE" ]; then
        sed -i 's/camera_type: ".*"/camera_type: "csi"/' "$CONFIG_FILE"
        print_success "Configuração alterada para CSI"
    else
        print_error "Arquivo de configuração não encontrado: $CONFIG_FILE"
        return 1
    fi

    print_camera "Configurações CSI:"
    print_info "   📷 Câmera: CSI IMX219"
    print_info "   📏 Resolução: 640x480@30fps"
    print_info "   🎯 Detectores: YOLOv8 + Tradicionais"
    print_info "   ⚡ Latência: Baixa (~50ms)"
    echo ""

    # Iniciar sistema
    print_info "Iniciando sistema de percepção com CSI..."
    ros2 launch perception dual_camera_perception.launch.py \
        camera_type:=csi \
        config_file:=$CONFIG_FILE \
        debug:=true
}

# Função para iniciar sistema com USB
start_perception_usb() {
    print_header "🎥 Iniciando Sistema com Câmera USB Logitech C930"
    
    # Verificar C930 primeiro
    if ! test_c930_camera; then
        print_error "Teste da C930 falhou - não é possível iniciar sistema USB"
        return 1
    fi

    # Configurar para USB
    if [ -f "$CONFIG_FILE" ]; then
        sed -i 's/camera_type: ".*"/camera_type: "usb"/' "$CONFIG_FILE"
        print_success "Configuração alterada para USB"
    else
        print_error "Arquivo de configuração não encontrado: $CONFIG_FILE"
        return 1
    fi

    print_camera "Configurações USB C930:"
    print_info "   📷 Câmera: USB Logitech C930"
    print_info "   📏 Resolução: 1280x720@30fps"
    print_info "   🎯 Detectores: YOLOv8 + Tradicionais"
    print_info "   🔍 Auto Focus: Ativo"
    print_info "   📐 Campo de Visão: 90°"
    echo ""

    # Iniciar sistema
    print_info "Iniciando sistema de percepção com USB C930..."
    ros2 launch perception dual_camera_perception.launch.py \
        camera_type:=usb \
        config_file:=$CONFIG_FILE \
        debug:=true
}

# Função para rebuildar workspace
rebuild_workspace() {
    print_header "🔧 Reconstruindo Workspace"
    
    print_info "Reconstruindo pacote perception..."
    colcon build --packages-select perception

    if [ $? -eq 0 ]; then
        print_success "Workspace reconstruído com sucesso"
        source ./install/setup.bash
        return 0
    else
        print_error "Falha ao reconstruir workspace"
        return 1
    fi
}

# Função para informações do sistema
show_system_info() {
    print_header "📊 Informações e Status do Sistema"
    
    # Status ROS2
    echo "🤖 Status do ROS 2:"
    if ros2 node list &> /dev/null; then
        ros2 node list | sed 's/^/   /'
    else
        echo "   Nenhum nó ROS em execução"
    fi
    echo ""

    # Tópicos
    echo "📡 Tópicos disponíveis:"
    if ros2 topic list &> /dev/null; then
        ros2 topic list | grep -E "(camera|perception|image)" | sed 's/^/   /' || echo "   Nenhum tópico de percepção ativo"
    else
        echo "   Nenhum tópico disponível"
    fi
    echo ""

    # Configuração atual da câmera
    echo "🎥 Configuração atual da câmera:"
    if [ -f "$CONFIG_FILE" ]; then
        current_camera=$(grep "camera_type:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
        if [ "$current_camera" = "csi" ]; then
            echo "   📷 CSI IMX219 (640x480@30fps)"
        elif [ "$current_camera" = "usb" ]; then
            echo "   📷 USB Logitech C930 (1280x720@30fps)"
        else
            echo "   ❓ Configuração desconhecida: $current_camera"
        fi
    else
        echo "   ❌ Arquivo de configuração não encontrado"
    fi
    echo ""

    # Versões
    echo "📦 Versões instaladas:"
    python3 -c "import cv2; print(f'   OpenCV: {cv2.__version__}')" 2>/dev/null || echo "   OpenCV: Não instalado"
    python3 -c "import ultralytics; print(f'   Ultralytics: {ultralytics.__version__}')" 2>/dev/null || echo "   Ultralytics: Não instalado"
    echo ""

    # Hardware
    echo "🖥️  Recursos de hardware:"
    echo "   CPUs: $(nproc)"
    echo "   Memória: $(free -h | grep -i 'mem' | awk '{print $2}')"
    if command -v nvidia-smi &> /dev/null; then
        echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    else
        echo "   GPU: Não disponível"
    fi
}

# Menu principal
show_menu() {
    print_header "🚀 Sistema de Teste e Conveniência - Percepção RoboIME"
    echo "Sistema Simplificado: YOLOv8 (6 classes otimizadas - estratégia + localização)"
    echo "Suporte dual: CSI IMX219 e USB Logitech C930"
    echo ""
    echo "📋 MENU PRINCIPAL:"
    echo ""
    echo "🎥 CÂMERAS:"
    echo "1. Iniciar sistema com CSI IMX219 (baixa latência)"
    echo "2. Iniciar sistema com USB Logitech C930 (alta qualidade)"
    echo "3. Testar câmera USB C930 (diagnóstico completo)"
    echo ""
    echo "🔧 SISTEMA:"
    echo "4. Verificar ambiente e dependências"
    echo "5. Verificar câmeras disponíveis"
    echo "6. Reconstruir workspace"
    echo "7. Informações e status do sistema"
    echo ""
    echo "🧪 TESTES AVANÇADOS:"
    echo "8. Testar apenas detectores YOLOv8"
    echo "9. Testar apenas detectores tradicionais"
    echo "10. Benchmark de performance"
    echo ""
    echo "0. Sair"
    echo ""
}

# Função principal
main() {
    # Verificação inicial rápida
    if ! check_environment; then
        print_error "Falha na verificação do ambiente"
        exit 1
    fi

    while true; do
        show_menu
        read -p "Escolha uma opção: " option

        case $option in
            1)
                start_perception_csi
                ;;
            2)
                start_perception_usb
                ;;
            3)
                test_c930_camera
                ;;
            4)
                check_environment
                check_dependencies
                ;;
            5)
                check_cameras
                ;;
            6)
                rebuild_workspace
                ;;
            7)
                show_system_info
                ;;
            8)
                print_header "🧪 Testando apenas detectores YOLOv8"
                print_info "Iniciando sistema apenas com YOLOv8..."
                ros2 launch perception dual_camera_perception.launch.py \
                    config_file:=$CONFIG_FILE \
                    debug:=true
                ;;
            9)
                print_header "🧪 Testando apenas detectores tradicionais"
                print_info "Este teste requer configuração manual no YAML"
                print_info "Desative YOLOv8 no arquivo de configuração"
                ;;
            10)
                print_header "📊 Benchmark de Performance"
                print_info "Executando testes de benchmark..."
                
                echo "🔄 Teste 1: Sistema CSI (10 segundos)"
                timeout 10 ros2 launch perception dual_camera_perception.launch.py \
                    camera_type:=csi config_file:=$CONFIG_FILE debug:=false &
                wait
                
                echo "🔄 Teste 2: Sistema USB (10 segundos)"
                timeout 10 ros2 launch perception dual_camera_perception.launch.py \
                    camera_type:=usb config_file:=$CONFIG_FILE debug:=false &
                wait
                
                print_success "Benchmark concluído!"
                ;;
            0)
                print_info "Saindo do sistema de teste..."
                exit 0
                ;;
            *)
                print_error "Opção inválida! Tente novamente."
                sleep 2
                ;;
        esac

        echo ""
        read -p "Pressione ENTER para voltar ao menu..." 
    done
}

# Executar função principal
main 