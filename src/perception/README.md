# RoboIME Vision

Sistema de visão computacional para o robô de futebol da RoboIME, adaptado do sistema de visão dos [Hamburg Bit-Bots](https://github.com/bit-bots/bitbots_main/tree/main/bitbots_vision).

## Visão Geral

Este pacote contém módulos para detecção de elementos importantes em um jogo de futebol de robôs:

- **Bola**: Detecção da bola de futebol usando segmentação por cor e transformada de Hough
- **Campo**: Detecção do campo de futebol e sua fronteira
- **Linhas**: Detecção das linhas brancas do campo
- **Gols**: Detecção dos postes do gol
- **Obstáculos**: Detecção de outros robôs e obstáculos
- **YOEO**: Detecção de múltiplos objetos (bola, gol, robôs, árbitro) usando redes neurais
- **Câmera Jetson**: Suporte para câmera CSI e USB na Jetson Nano

## Requisitos

- ROS 2 Eloquent
- NVIDIA Jetson Nano com Jetpack 4.6.1
- Python 3.6+
- OpenCV 4.x
- NumPy
- TensorFlow 2.x
- TensorRT (para otimização na Jetson)
- GStreamer (para câmera CSI da Jetson)

## Instalação

1. Clone este repositório no seu workspace ROS 2:
```bash
cd ~/ros2_ws/src
git clone https://github.com/seu-usuario/RoboIME-HSL2025.git
```

2. Instale as dependências:
```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-numpy libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
pip3 install --user numpy opencv-python tensorflow==2.4.0
```

3. Para a Jetson Nano, instale o TensorRT:
```bash
# O TensorRT já vem instalado com o JetPack 4.6.1
# Verifique a instalação com:
dpkg -l | grep tensorrt
```

4. Compile o pacote:
```bash
cd ~/ros2_ws
colcon build --packages-select roboime_vision
```

5. Fonte o workspace:
```bash
source ~/ros2_ws/install/setup.bash
```

## Uso

### Iniciar o sistema de visão completo:

```bash
ros2 launch roboime_vision vision.launch.py
```

### Iniciar o sistema de visão na Jetson Nano:

```bash
ros2 launch roboime_vision jetson_vision.launch.py
```

### Iniciar apenas a câmera da Jetson Nano:

```bash
# Câmera CSI (padrão)
ros2 launch roboime_vision jetson_camera.launch.py

# Câmera CSI com parâmetros personalizados
ros2 launch roboime_vision jetson_camera.launch.py camera_width:=1280 camera_height:=720 camera_fps:=60

# Câmera USB
ros2 launch roboime_vision jetson_camera.launch.py camera_type:=usb camera_index:=0
```

### Iniciar componentes individuais:

```bash
# Detector de bola
ros2 run roboime_vision ball_detector.py

# Detector de campo
ros2 run roboime_vision field_detector.py

# Detector de linhas
ros2 run roboime_vision line_detector.py

# Detector de gols
ros2 run roboime_vision goal_detector.py

# Detector de obstáculos
ros2 run roboime_vision obstacle_detector.py

# Detector YOEO
ros2 run roboime_vision yoeo_detector

# Câmera da Jetson Nano
ros2 run roboime_vision jetson_camera_node.py
```

### Treinar o modelo YOEO:

```bash
# Treinar com dados personalizados
python3 src/perception/src/yoeo/train_yoeo.py --train_annotations=caminho/para/anotacoes.json --train_images=caminho/para/imagens --output_dir=models
```

### Converter modelo para TensorRT:

```bash
# Converter modelo para TensorRT (otimizado para Jetson Nano)
python3 src/perception/src/yoeo/tensorrt_converter.py --model_path=models/yoeo_model_final.h5 --output_dir=models/tensorrt --precision=FP16
```

## Configuração

Os parâmetros do sistema de visão podem ser configurados no arquivo `config/vision_params.yaml`. Você pode ajustar os parâmetros de detecção de cores, limiares, etc.

### Parâmetros da Câmera da Jetson

- `camera_type`: Tipo de câmera ('csi' ou 'usb')
- `camera_index`: Índice da câmera (geralmente 0)
- `camera_width`: Largura da imagem da câmera
- `camera_height`: Altura da imagem da câmera
- `camera_fps`: Taxa de quadros da câmera
- `display_width`: Largura da janela de exibição
- `display_height`: Altura da janela de exibição
- `enable_display`: Habilitar exibição da imagem da câmera

### Parâmetros do YOEO

- `model_path`: Caminho para o modelo YOEO
- `input_width`: Largura da entrada do modelo
- `input_height`: Altura da entrada do modelo
- `confidence_threshold`: Limiar de confiança para detecções
- `iou_threshold`: Limiar de IoU para non-maximum suppression
- `use_tensorrt`: Usar modelo otimizado com TensorRT

## Tópicos ROS

### Entradas
- `/camera/image_raw` (sensor_msgs/Image): Imagem da câmera
- `/camera/camera_info` (sensor_msgs/CameraInfo): Informações de calibração da câmera

### Saídas
- `/ball_position` (geometry_msgs/Pose2D): Posição da bola
- `/field_mask` (sensor_msgs/Image): Máscara do campo
- `/field_boundary` (sensor_msgs/Image): Fronteira do campo
- `/lines_image` (sensor_msgs/Image): Imagem com as linhas detectadas
- `/goal_posts` (geometry_msgs/PoseArray): Posições dos postes do gol
- `/obstacles` (geometry_msgs/PoseArray): Posições dos obstáculos
- `/vision_debug` (sensor_msgs/Image): Imagem de debug do pipeline de visão
- `/ball_detection_debug` (sensor_msgs/Image): Imagem de debug da detecção de bola
- `/field_detection_debug` (sensor_msgs/Image): Imagem de debug da detecção de campo
- `/line_detection_debug` (sensor_msgs/Image): Imagem de debug da detecção de linhas
- `/goal_detection_debug` (sensor_msgs/Image): Imagem de debug da detecção de gols
- `/obstacle_detection_debug` (sensor_msgs/Image): Imagem de debug da detecção de obstáculos

### Tópicos YOEO
- `/yoeo/detections` (vision_msgs/Detection2DArray): Todas as detecções do YOEO
- `/yoeo/ball_detections` (vision_msgs/Detection2DArray): Detecções de bola
- `/yoeo/goal_detections` (vision_msgs/Detection2DArray): Detecções de gol
- `/yoeo/robot_detections` (vision_msgs/Detection2DArray): Detecções de robôs
- `/yoeo/referee_detections` (vision_msgs/Detection2DArray): Detecções de árbitro
- `/yoeo/debug_image` (sensor_msgs/Image): Imagem de debug do YOEO

## Solução de Problemas

### Câmera CSI da Jetson Nano

Se você estiver tendo problemas com a câmera CSI da Jetson Nano, verifique:

1. Se o cabo da câmera está conectado corretamente
2. Se a câmera está habilitada no Jetson:
   ```bash
   sudo systemctl restart nvargus-daemon
   ```
3. Teste a câmera com o GStreamer:
   ```bash
   gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink
   ```

### TensorRT na Jetson Nano

Se você estiver tendo problemas com o TensorRT na Jetson Nano:

1. Verifique se o TensorRT está instalado corretamente:
   ```bash
   dpkg -l | grep tensorrt
   ```
2. Verifique se o TensorFlow está configurado para usar o TensorRT:
   ```bash
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
3. Tente reduzir a precisão para FP16 ou INT8 para melhor desempenho.

## Licença

Este projeto é licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Agradecimentos

Este pacote é baseado no trabalho do [Hamburg Bit-Bots](https://github.com/bit-bots/bitbots_main/tree/main/bitbots_vision), adaptado para uso com ROS 2 Eloquent e Jetson Nano. 