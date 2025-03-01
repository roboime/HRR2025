#!/bin/bash
# Script para testar o sistema de visão da RoboIME na Jetson Nano

set -e  # Sair em caso de erro

echo "Iniciando testes do sistema de visão da RoboIME..."

# Verificar se o ROS 2 está instalado
if ! command -v ros2 &> /dev/null; then
    echo "ERRO: ROS 2 não encontrado. Verifique se o ROS 2 está instalado e configurado corretamente."
    exit 1
fi

# Verificar se o pacote está instalado
if ! ros2 pkg list | grep -q roboime_vision; then
    echo "ERRO: Pacote roboime_vision não encontrado. Verifique se o pacote foi compilado corretamente."
    echo "Execute: cd ~/ros2_ws && colcon build --packages-select roboime_vision"
    exit 1
fi

# Função para testar um nó
test_node() {
    local node_name=$1
    local node_executable=$2
    local timeout=$3
    
    echo "Testando $node_name..."
    
    # Iniciar o nó em segundo plano
    ros2 run roboime_vision $node_executable &
    local pid=$!
    
    # Aguardar um pouco para o nó iniciar
    sleep 2
    
    # Verificar se o nó está em execução
    if ros2 node list | grep -q $node_name; then
        echo "✓ $node_name está em execução"
        
        # Verificar tópicos publicados
        echo "  Tópicos publicados:"
        ros2 topic list | grep -i "$node_name\|camera" | while read topic; do
            echo "    - $topic"
        done
        
        # Aguardar o tempo especificado
        echo "  Aguardando $timeout segundos..."
        sleep $timeout
    else
        echo "✗ $node_name não está em execução"
    fi
    
    # Encerrar o nó
    kill $pid 2>/dev/null || true
    sleep 1
}

# Testar câmera da Jetson
echo "=== Teste da Câmera da Jetson ==="
echo "Iniciando câmera da Jetson..."
ros2 run roboime_vision jetson_camera_node.py &
camera_pid=$!

# Aguardar um pouco para a câmera iniciar
sleep 3

# Verificar se a câmera está publicando
if ros2 topic list | grep -q "/camera/image_raw"; then
    echo "✓ Câmera está publicando imagens"
    
    # Verificar taxa de publicação
    echo "Taxa de publicação da câmera:"
    ros2 topic hz /camera/image_raw --window 5
else
    echo "✗ Câmera não está publicando imagens"
fi

# Encerrar a câmera
kill $camera_pid 2>/dev/null || true
sleep 1

# Testar cada detector individualmente
echo ""
echo "=== Testes dos Detectores ==="

# Iniciar a câmera novamente para os testes dos detectores
ros2 run roboime_vision jetson_camera_node.py &
camera_pid=$!
sleep 3

# Testar cada detector
test_node "ball_detector" "ball_detector.py" 10
test_node "field_detector" "field_detector.py" 10
test_node "line_detector" "line_detector.py" 10
test_node "goal_detector" "goal_detector.py" 10
test_node "obstacle_detector" "obstacle_detector.py" 10

# Encerrar a câmera
kill $camera_pid 2>/dev/null || true

echo ""
echo "Testes concluídos!"
echo "Para executar o sistema completo, use:"
echo "ros2 launch roboime_vision jetson_vision.launch.py" 