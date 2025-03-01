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

# Sistema de Visão YOEO (You Only Encode Once)

Este diretório contém a implementação de um sistema de visão para robôs da categoria SSL (Small Size League), baseado na arquitetura YOEO (You Only Encode Once) inspirada na implementação do time Bit-Bots.

## Estrutura do Sistema

O sistema de visão YOEO é organizado de forma modular, permitindo diferentes configurações e combinações de abordagens para detecção e segmentação de elementos em campo.

### Estrutura de Diretórios

```
src/perception/
├── config/                 # Arquivos de configuração
│   └── vision_params.yaml  # Parâmetros para os componentes de visão
├── launch/                 # Arquivos de lançamento ROS
│   ├── jetson_vision.launch.py    # Lançamento para Jetson Nano
│   ├── vision.launch.py           # Lançamento para computador normal
│   └── yoeo_detector.launch.py    # Lançamento específico para YOEO
├── src/                    # Código-fonte
│   ├── ball_detector.py    # Detector tradicional de bola
│   ├── field_detector.py   # Detector tradicional de campo
│   ├── goal_detector.py    # Detector tradicional de gol
│   ├── line_detector.py    # Detector tradicional de linhas
│   ├── obstacle_detector.py # Detector tradicional de obstáculos
│   └── yoeo/               # Módulo YOEO
│       ├── __init__.py     # Inicialização do módulo
│       ├── yoeo_detector.py      # Nó ROS para integração com YOEO
│       ├── yoeo_handler.py       # Manipulador do modelo YOEO
│       ├── yoeo_model.py         # Definição da arquitetura do modelo
│       └── components/           # Componentes modulares
│           ├── __init__.py       # Inicialização dos componentes
│           ├── ball_component.py      # Componente de detecção de bola
│           ├── field_component.py     # Componente de segmentação de campo
│           ├── goal_component.py      # Componente de detecção de gol
│           ├── line_component.py      # Componente de segmentação de linhas
│           ├── referee_component.py   # Componente de detecção de árbitro
│           └── robot_component.py     # Componente de detecção de robôs
└── README.md               # Este arquivo
```

## Arquitetura do Sistema

### Abordagem YOEO

O YOEO é uma arquitetura neural unificada que combina:

1. **Detecção de Objetos**: Identificação e localização de elementos como bola, robôs, postes de gol e árbitro.
2. **Segmentação Semântica**: Classificação de cada pixel da imagem, principalmente para campo e linhas.

Essa abordagem oferece vantagens significativas:

- **Eficiência Computacional**: Um único modelo realiza múltiplas tarefas.
- **Inferência Rápida**: Otimizado para hardware especializado (Jetson Nano).
- **Robustez**: Melhor desempenho em condições variáveis de iluminação e ambientes.

### Estrutura Modular

O sistema segue uma arquitetura modular com os seguintes componentes principais:

#### 1. YOEOModel

Define a arquitetura neural do modelo YOEO, baseada em uma rede convolucional com:
- Backbone para extração de características
- Heads de detecção para objetos em diferentes escalas
- Heads de segmentação para campo e linhas

#### 2. YOEOHandler

Gerencia o modelo YOEO, incluindo:
- Carregamento e otimização do modelo (com suporte a TensorRT)
- Pré-processamento de imagens
- Execução de inferência
- Pós-processamento de resultados

#### 3. Componentes Especializados

Cada componente é responsável por um tipo específico de detecção ou segmentação:

- **BallDetectionComponent**: Detecção de bola e cálculo de posição 3D
- **FieldSegmentationComponent**: Segmentação do campo e extração de fronteira
- **LineSegmentationComponent**: Segmentação das linhas e extração de características
- **GoalDetectionComponent**: Detecção de postes de gol e cálculo de posição 3D
- **RobotDetectionComponent**: Detecção de robôs, com possível classificação por time
- **RefereeDetectionComponent**: Detecção de árbitro

#### 4. YOEODetector

Nó ROS que integra todos os componentes e fornece uma interface unificada para o sistema de visão:
- Gerencia os parâmetros de configuração
- Processa imagens da câmera
- Coordena os componentes ativos
- Publica resultados para tópicos ROS
- Suporta fallback para detectores tradicionais

## Integração com Detectores Tradicionais

Uma característica importante do sistema é a possibilidade de integração com detectores tradicionais baseados em processamento de imagem convencional:

- **Fallback Automático**: Se o YOEO não detectar um objeto com confiança suficiente, o sistema pode usar automaticamente um detector tradicional.
- **Validação Cruzada**: Os resultados do YOEO podem ser validados comparando com os resultados de detectores tradicionais.
- **Desenvolvimento Incremental**: Permite migração gradual de métodos tradicionais para abordagem neural.

## Configuração e Uso

### Parâmetros Configuráveis

Os parâmetros do sistema são definidos no arquivo `config/vision_params.yaml` e incluem:

- **Parâmetros do Modelo**: Caminho do modelo, dimensões de entrada, limiares de confiança
- **Ativação de Componentes**: Habilitar/desabilitar detecções específicas
- **Fallback**: Configurar uso de detectores tradicionais como backup
- **Debug**: Habilitar visualizações para depuração

### Lançamento do Sistema

O sistema pode ser iniciado usando um dos arquivos de lançamento:

```bash
# Para computadores normais
ros2 launch perception vision.launch.py

# Para Jetson Nano (otimizado)
ros2 launch perception jetson_vision.launch.py

# Apenas o detector YOEO
ros2 launch perception yoeo_detector.launch.py
```

### Tópicos ROS Publicados

O sistema publica os resultados nos seguintes tópicos:

- **/ball_position**: Posição da bola detectada
- **/field_mask**: Máscara de segmentação do campo
- **/field_boundary**: Fronteira do campo detectada
- **/lines_image**: Máscara de segmentação das linhas
- **/goal_posts**: Array de poses dos postes de gol
- **/robots**: Array de poses dos robôs detectados
- **/referee_position**: Pose do árbitro
- **/yoeo_detection_debug**: Imagem com visualizações de depuração

## Extensão e Desenvolvimento

### Adicionando Novos Componentes

O sistema foi projetado para ser facilmente estendido com novos componentes:

1. Crie um novo arquivo de componente em `src/yoeo/components/`
2. Implemente a classe de componente seguindo o padrão existente
3. Adicione a importação e inicialização no `yoeo_detector.py`
4. Atualize a configuração em `vision_params.yaml`

### Treinamento do Modelo

O treinamento do modelo YOEO não está incluído neste repositório, mas pode ser realizado com as seguintes etapas:

1. Coletar um conjunto de dados rotulado para seus objetos específicos
2. Adaptar a arquitetura em `yoeo_model.py` para suas classes
3. Treinar o modelo usando TensorFlow/Keras
4. Converter para TensorRT para otimização em Jetson (opcional)
5. Salvar o modelo no formato adequado e atualizar o caminho em `vision_params.yaml`

## Referências

Este trabalho é baseado na implementação da equipe Hamburg Bit-Bots:

- [Documentação Bit-Bots YOEO](https://robots.bit-bots.de/en/research/vision/yoeo/)
- [Repositório GitHub Bit-Bots](https://github.com/bit-bots/bitbots_vision)

A arquitetura YOEO é inspirada nas redes YOLO (You Only Look Once) para detecção de objetos, estendidas para incluir segmentação semântica. 