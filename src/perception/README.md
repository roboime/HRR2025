# RoboIME Vision

Sistema de visão computacional para o robô de futebol da RoboIME, adaptado do sistema de visão dos [Hamburg Bit-Bots](https://github.com/bit-bots/bitbots_main/tree/main/bitbots_vision).

## Visão Geral

Este pacote contém módulos para detecção de elementos importantes em um jogo de futebol de robôs:

- **Bola**: Detecção da bola de futebol usando segmentação por cor e transformada de Hough
- **Campo**: Detecção do campo de futebol e sua fronteira
- **Linhas**: Detecção das linhas brancas do campo
- **Gols**: Detecção dos postes do gol
- **Obstáculos**: Detecção de outros robôs e obstáculos
- **Câmera Jetson**: Suporte para câmera CSI e USB na Jetson Nano

## Requisitos

- ROS 2 Eloquent
- NVIDIA Jetson Nano com Jetpack 4.6.1
- Python 3.6+
- OpenCV 4.x
- NumPy
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
pip3 install --user numpy opencv-python
```

3. Compile o pacote:
```bash
cd ~/ros2_ws
colcon build --packages-select roboime_vision
```

4. Fonte o workspace:
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

# Câmera da Jetson Nano
ros2 run roboime_vision jetson_camera_node.py
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

## Licença

Este projeto é licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Agradecimentos

Este pacote é baseado no trabalho do [Hamburg Bit-Bots](https://github.com/bit-bots/bitbots_main/tree/main/bitbots_vision), adaptado para uso com ROS 2 Eloquent e Jetson Nano. 