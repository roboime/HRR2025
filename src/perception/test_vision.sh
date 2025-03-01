#!/bin/bash
# Script para testar o sistema de visão da RoboIME

echo "Testando o sistema de visão da RoboIME..."

# Verificar se o ROS 2 está instalado
if ! command -v ros2 &> /dev/null; then
    echo "ERRO: ROS 2 não encontrado. Certifique-se de que o ROS 2 está instalado e configurado."
    exit 1
fi

# Verificar se o workspace está configurado
if [ ! -d "$(pwd)/install" ]; then
    echo "ERRO: Diretório de instalação não encontrado. Execute este script do diretório raiz do workspace."
    exit 1
fi

# Fonte o workspace
echo "Configurando o ambiente ROS 2..."
source install/setup.bash

# Verificar se o pacote está instalado
if ! ros2 pkg list | grep -q roboime_vision; then
    echo "ERRO: Pacote roboime_vision não encontrado. Certifique-se de que o pacote está compilado."
    exit 1
fi

# Função para testar um nó
test_node() {
    local node_name=$1
    local executable=$2
    local timeout=$3
    
    echo "Testando $node_name..."
    timeout $timeout ros2 run roboime_vision $executable &
    local pid=$!
    sleep 2
    
    if ps -p $pid > /dev/null; then
        echo "✓ $node_name está funcionando."
        kill $pid
        wait $pid 2>/dev/null
        return 0
    else
        echo "✗ $node_name falhou ao iniciar."
        return 1
    fi
}

# Função para testar um tópico
test_topic() {
    local topic=$1
    local timeout=$2
    
    echo "Verificando tópico $topic..."
    if timeout $timeout ros2 topic echo $topic --once &>/dev/null; then
        echo "✓ Tópico $topic está publicando mensagens."
        return 0
    else
        echo "✗ Tópico $topic não está publicando mensagens."
        return 1
    fi
}

# Testar a câmera
echo "Iniciando a câmera..."
if [ -f "/dev/video0" ] || [ -d "/dev/nvhost-vi" ]; then
    echo "Dispositivo de câmera encontrado."
    
    # Iniciar a câmera em segundo plano
    ros2 run roboime_vision jetson_camera_node.py &
    camera_pid=$!
    
    # Aguardar a câmera iniciar
    sleep 3
    
    # Verificar se a câmera está publicando
    if test_topic "/camera/image_raw" 5; then
        echo "Câmera está funcionando corretamente."
    else
        echo "AVISO: Câmera não está publicando imagens."
    fi
    
    # Testar os detectores
    test_node "Detector de campo" "field_detector.py" 5
    test_node "Detector de bola" "ball_detector.py" 5
    test_node "Detector de linhas" "line_detector.py" 5
    test_node "Detector de gols" "goal_detector.py" 5
    test_node "Detector de obstáculos" "obstacle_detector.py" 5
    test_node "Detector YOEO" "yoeo_detector" 5
    
    # Parar a câmera
    kill $camera_pid
    wait $camera_pid 2>/dev/null
else
    echo "AVISO: Nenhum dispositivo de câmera encontrado."
fi

# Testar o TensorFlow e TensorRT
echo "Testando TensorFlow..."
if python3 -c "import tensorflow as tf; print('TensorFlow versão:', tf.__version__); print('GPU disponível:', tf.config.list_physical_devices('GPU'))" 2>/dev/null; then
    echo "✓ TensorFlow está funcionando."
else
    echo "✗ Erro ao carregar TensorFlow."
fi

echo "Testando TensorRT..."
if python3 -c "from tensorflow.python.compiler.tensorrt import trt_convert as trt; print('TensorRT disponível')" 2>/dev/null; then
    echo "✓ TensorRT está disponível."
else
    echo "✗ TensorRT não está disponível ou não está configurado corretamente."
fi

# Testar o lançamento completo
echo "Testando o lançamento completo do sistema de visão..."
echo "Pressione Ctrl+C após alguns segundos para encerrar o teste."
ros2 launch roboime_vision vision.launch.py use_ball_detector:=true use_field_detector:=true use_yoeo_detector:=true

echo "Testes concluídos!" 